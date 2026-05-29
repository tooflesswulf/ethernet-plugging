from collections import namedtuple
from typing import Literal
import rtde_control
import rtde_receive
import numpy as np
import threading
import pathlib
import h5py
import copy
import time
import cv2
import os

from util import URPose, blend, episode_index
from camera import Camera
import wsg


class RobotObs(namedtuple('RobotObs', ('time', 'actual_pose', 'actual_force'))):
    pass


class GripperObs(namedtuple('GripperObs', ('time', 'gripper_width', 'gripper_force'))):
    pass


class CameraObs(namedtuple('CameraObs', ('time', 'rgb'))):
    pass


class Env:
    """
    Minimal robot environment wrapper for:
        - Teleoperation
        - Robot policy evaluation
        - Data collection

    Components:
        - Single RGB camera
        - UR robot arm
        - WSG gripper
    """

    def __init__(
        self,
        robot_ip="192.168.0.100",  # could be 101 or 100 depending on your setup
        gripper_ip="192.168.0.20",
        camera_crop_mode=1,  # crop on the right half of the image to focus on the workspace
        servo_frequency=500,
        gripper_query_frequency=250,
        max_position_step=(0.008, 0.008, 0.008),
        max_orientation_step=0.02,
        lookahead_time=0.1,
        servo_gain=500,
        obs_mode: Literal['latest', 'mean'] = 'latest',
        dataset_path=None,
        save_interval=0.1,
    ):
        # ============================================================
        # Internal states
        # ============================================================
        self.t0 = None
        self.open_width = 35
        self.home_pose = URPose(-0.125, 0.545, 0.305, 2.44, 2.44, 0.653)
        self.gripper_state = 0  # 0=open, 1=closed
        self.des_pose, self.des_gripper_state = self.home_pose, self.gripper_state

        # ============================================================
        # Camera
        # ============================================================
        self.camera = Camera(crop_mode=camera_crop_mode)

        # ============================================================
        # Robot interfaces
        # ============================================================
        self.robot_ip = robot_ip
        self.gripper_ip = gripper_ip
        self.ctrl = rtde_control.RTDEControlInterface(robot_ip)
        self.recv = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.gripper = wsg.WSG(ip=gripper_ip)
        self.gripper_query_frequency = gripper_query_frequency

        # ============================================================
        # Servo parameters
        # ============================================================
        self.servo_frequency = servo_frequency
        self.dt = 1.0 / servo_frequency
        self.max_position_step = np.array(max_position_step)
        self.max_orientation_step = max_orientation_step
        self.lookahead_time = lookahead_time
        self.servo_gain = servo_gain

        print("Initializing environment...")
        print(f"Robot IP:   {robot_ip}")
        print(f"Gripper IP: {gripper_ip}")
        print(f"Servo  {self.home_pose} frequency: {servo_frequency} Hz")

        # ----------------------------
        # threading
        # ----------------------------
        self.stop_flag = False
        self.threads = []
        # self.control_thread = None
        # self.camera_thread = None
        # self.gripper_thread = None
        # self.logger_thread = None
        self.obs_mode = obs_mode

        # ----------------------------
        # observation buffer
        # ----------------------------
        self.dataset_path = dataset_path
        self.save_interval = save_interval  # save thread loop interval in seconds
        self.robot_obs: list[RobotObs] = []
        self.gripper_obs: list[GripperObs] = []
        self.camera_obs: list[CameraObs] = []

    def wait_for_obs(self):
        while len(self.camera_obs) == 0 or len(self.robot_obs) == 0 or len(self.gripper_obs) == 0:
            time.sleep(0.01)

    def get_obs(self):
        # Assume obs is populated with at least one entry
        if self.obs_mode == 'latest':
            obs = {
                'rgb': self.camera_obs[-1].rgb,
                'state': {
                    'pose': self.robot_obs[-1].actual_pose,
                    'force': self.robot_obs[-1].actual_force,
                    'gripper_width': self.gripper_obs[-1].gripper_width,
                    'gripper_force': self.gripper_obs[-1].gripper_force,
                }
            }
            return obs

        elif self.obs_mode == 'mean':
            raise NotImplementedError("Mean obs mode not implemented yet")

    def start(self):
        if self.dataset_path is not None:
            prefix = 'episode'
            ix = episode_index(self.dataset_path, prefix=prefix)
            self.epi_path = pathlib.Path(self.dataset_path) / f'{prefix}{ix:06d}'
            self.epi_path.mkdir(parents=True, exist_ok=True)
            os.makedirs(self.epi_path / 'images', exist_ok=True)

        self.stop_flag = False
        self.threads = [
            threading.Thread(target=self._control_loop, daemon=True,),
            threading.Thread(target=self._camera_loop, daemon=True,),
            threading.Thread(target=self._gripper_loop, daemon=True,),
            threading.Thread(target=self._logger_loop, daemon=True,),
        ]
        for thread in self.threads:
            thread.start()

        self.t0 = time.time()

    def reset(self, home_pose=None):
        """
        Reset environment:
            1. Open/home the gripper
            2. Move robot to home pose
            3. Set gripper to default width

        Args:
            home_pose: URPose
        """
        if home_pose is None:
            home_pose = self.home_pose

        print('Resetting environment...')
        self.stop_flag = True
        for thr in self.threads:
            thr.join()
        self.save_data()

        # ============================================================
        # Home / open gripper
        # ============================================================
        g = self.gripper.home()
        g.ack.wait()

        # ============================================================
        # Move robot home (blocking)
        # ============================================================
        self.ctrl.moveL(home_pose, 0.1, 0.1)
        self.des_pose = home_pose  # Ensure robot doesn't move after homing

        # Wait for gripper homing to finish
        g.finished.wait()

        # ============================================================
        # Move gripper to default open width
        # ============================================================
        g = self.gripper.move(position=self.open_width, speed=50)
        g.finished.wait()
        self.gripper_state = 0

        # ============================================================
        # Reset observations
        # ============================================================
        self.robot_obs: list[RobotObs] = []
        self.gripper_obs: list[GripperObs] = []
        self.camera_obs: list[CameraObs] = []
        self.start()

        print('Environment reset complete.')
        self.wait_for_obs()
        return self.get_obs()

    def close(self):
        self.stop_flag = True
        for thr in self.threads:
            thr.join()
        self.camera.close()
        self.save_data()

    def _control_loop(self):
        while not self.stop_flag:
            t_start = self.ctrl.initPeriod()
            actual_pose = URPose(*self.recv.getActualTCPPose())
            actual_force = URPose(*self.recv.getActualTCPForce())
            self.robot_obs.append(RobotObs(time=time.time() - self.t0,
                                  actual_pose=actual_pose, actual_force=actual_force))

            des_pose = self.des_pose
            des_gripper_state = self.des_gripper_state
            gripper_state = self.gripper_state

            # ----------------------------
            # gripper logic (non-blocking preferred)
            # ----------------------------
            if gripper_state != des_gripper_state:
                if gripper_state == 0:
                    self.gripper.grip(force=40, width=20, speed=50)
                    self.gripper_state = 1
                else:
                    self.gripper.release(pullback=10, speed=50)
                    self.gripper_state = 0

            # ----------------------------
            # blend + servo
            # ----------------------------
            command = blend(
                actual_pose,
                des_pose,
                self.max_position_step,
                self.max_orientation_step,
            )

            self.ctrl.servoL(
                command,
                0.0,
                0.0,
                self.dt,
                self.lookahead_time,
                self.servo_gain,
            )

            self.ctrl.waitPeriod(t_start)

    def _camera_loop(self):
        while not self.stop_flag:
            rgb = self.camera.get_rgb()
            self.camera_obs.append(CameraObs(time=time.time() - self.t0, rgb=rgb))

    def _gripper_loop(self):
        while not self.stop_flag:
            t0 = time.perf_counter()
            force = self.gripper.force()
            pos = self.gripper.position()

            self.gripper_obs.append(GripperObs(time=time.time() - self.t0,
                                               gripper_width=force.value,
                                               gripper_force=pos.value))
            sleep_dur = max(0, 1.0 / self.gripper_query_frequency - (time.perf_counter() - t0))
            time.sleep(sleep_dur)

    def _logger_loop(self):
        image_path = self.epi_path / 'images'
        pose_list = []
        force_list = []
        gpos_list = []
        gforce_list = []
        self.wait_for_obs()  # Ensure we have at least one obs before starting logging

        image_idx = 0
        while not self.stop_flag:
            t0 = time.perf_counter()
            obs = self.get_obs()

            im_path = pathlib.Path(image_path) / f'{image_idx:06d}.png'
            cv2.imwrite(im_path, cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR))
            pose_list.append(obs['state']['pose'])
            force_list.append(obs['state']['force'])
            gpos_list.append(obs['state']['gripper_width'])
            gforce_list.append(obs['state']['gripper_force'])
            image_idx += 1

            sleep_time = max(0, self.save_interval - (time.perf_counter() - t0))
            time.sleep(sleep_time)

        # IMPORTANT: logger must exit via stop_flag for data to be saved
        np.savez_compressed(
            self.epi_path / 'states.npz',
            pose=np.array(pose_list),
            force=np.array(force_list),
            gripper_width=np.array(gpos_list),
            gripper_force=np.array(gforce_list),
        )

    def save_data(self):
        # Save collected RAW data to HDF5.
        print(f'Saving data to {self.epi_path}...')
        path = self.epi_path / 'rawdata.h5'
        with h5py.File(path, 'w') as f:
            f.create_dataset('robot_obs/time', data=[obs.time for obs in self.robot_obs])
            f.create_dataset('robot_obs/actual_pose', data=[obs.actual_pose for obs in self.robot_obs])
            f.create_dataset('robot_obs/actual_force', data=[obs.actual_force for obs in self.robot_obs])

            f.create_dataset('gripper_obs/time', data=[obs.time for obs in self.gripper_obs])
            f.create_dataset('gripper_obs/gripper_width', data=[obs.gripper_width for obs in self.gripper_obs])
            f.create_dataset('gripper_obs/gripper_force', data=[obs.gripper_force for obs in self.gripper_obs])

            f.create_dataset('camera_obs/time', data=[obs.time for obs in self.camera_obs])
            f.create_dataset('camera_obs/rgb', data=[obs.rgb for obs in self.camera_obs])
        print(f'Data saved to {path}')
