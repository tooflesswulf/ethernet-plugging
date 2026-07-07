from scipy.spatial.transform import Rotation as R, Slerp
from collections import namedtuple
from typing import Literal
import rtde_control
import rtde_receive
import numpy as np
import threading
import pathlib
import h5py
import time
import cv2
import os

from util import URPose, blend, slerp, episode_index, dict2hdf5
from camera import Camera
import wsg


class RobotObs(namedtuple('RobotObs', ('time', 'actual_pose', 'actual_force', 'filtered_force'))):
    pass


class GripperObs(namedtuple('GripperObs', ('time', 'gripper_width', 'gripper_force'))):
    pass


class CameraObs(namedtuple('CameraObs', ('time', 'image'))):
    pass


class Command(namedtuple('Command', ('time', 'des_pose', 'des_gripper', 'adaptive_mode', 'des_zforce', 'controller_state'))):
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
        control_frequency=20,
        servo_frequency=500,
        gripper_query_frequency=250,
        max_position_step=(0.008, 0.008, 0.008),
        max_orientation_step=0.05,
        lookahead_time=0.1,
        servo_gain=500,
        obs_mode: Literal['latest', 'mean'] = 'latest',
        dataset_path=None,
        save_interval=0.1,
        save_eps=1e-3,
        gwidth=20,
        gforce=40,
        gspeed=50,
        gpullback=10,
        payload_mass=1.7,  # measured gripper assembly [kg]
        payload_cog=(0.004, 0.001, 0.039),  # measured CoG, tool flange frame [m]
        gravity_cal_path=None,
        metadata={}
    ):
        # ============================================================
        # Internal states
        # ============================================================
        self.t0 = None
        self.open_width = gwidth + 2 * gpullback
        self.home_pose = URPose(-0.125, 0.545, 0.305, 2.44, 2.44, 0.653)
        self.gripper_state = 0  # 0=open, 1=closed
        self.des_pose, self.des_gripper_state = self.home_pose, self.gripper_state
        self.des_zforce = 0.
        self.adaptive_mode = False
        self.last_step_t = time.perf_counter()

        # getActualTCPForce() compensates the configured payload internally, so the
        # correct payload config (set in reset()) removes the bulk of the
        # orientation-dependent drift. What leaks through — sensor gain/crosstalk
        # acting on the payload weight — is fitted by calibrate_gravity_residual()
        # as a 3x3 matrix B mapping the change in tool-frame gravity direction to a
        # tool-frame force residual.
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.grav_cal_path = pathlib.Path(gravity_cal_path) if gravity_cal_path \
            else pathlib.Path(__file__).parent / 'gravity_residual.npz'
        self.grav_residual_B = np.zeros((3, 3))  # tool-frame force residual model [N]
        self.grav_residual_cog = np.zeros(3)     # residual lever arm, tool frame [m]
        if self.grav_cal_path.exists():
            cal = np.load(self.grav_cal_path)
            if 'residual_B' in cal:
                self.grav_residual_B = cal['residual_B']
                self.grav_residual_cog = cal['residual_cog']
                print(f'Loaded gravity residual calibration from {self.grav_cal_path}')
            else:
                print(f'Ignoring outdated calibration {self.grav_cal_path}; re-run scripts/calibrate_gravity.py')
        self._ft_zero_rot = None  # TCP orientation when the FT sensor was zeroed

        # ============================================================
        # Control parameters
        # ============================================================
        self.g_force = gforce
        self.g_width = gwidth
        self.g_speed = gspeed
        self.g_pullback = gpullback

        # ============================================================
        # Camera
        # ============================================================
        self.camera_crop_mode = camera_crop_mode

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
        self.input_frequency = control_frequency
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
        self.obs_mode = obs_mode

        # ----------------------------
        # observation buffer
        # ----------------------------
        self.dataset_path = dataset_path
        self.save_interval = save_interval  # save thread loop interval in seconds
        self.robot_obs: list[RobotObs] = []
        self.gripper_obs: list[GripperObs] = []
        self.camera_obs: list[CameraObs] = []
        self.commands: list[Command] = []
        self.save_eps = save_eps
        self.image_idx = 0
        self.metadata = metadata

    def wait_for_obs(self):
        while len(self.camera_obs) == 0 or len(self.robot_obs) == 0 or len(self.gripper_obs) == 0:
            time.sleep(0.01)

    def get_obs(self):
        # Assume obs is populated with at least one entry
        if self.obs_mode == 'latest':
            obs = {
                'image': self.camera_obs[-1].image,
                'state': {
                    'actual_pose': self.robot_obs[-1].actual_pose,
                    'actual_force': self.robot_obs[-1].actual_force,
                    'filtered_force': self.robot_obs[-1].filtered_force,
                    'gripper_width': self.gripper_obs[-1].gripper_width,
                    'gripper_force': self.gripper_obs[-1].gripper_force,
                }
            }
            return obs

        elif self.obs_mode == 'mean':
            raise NotImplementedError("Mean obs mode not implemented yet")

    def step(self, des_pose, des_gripper_state, des_zforce=0., adaptive_mode=False, dualsense=None):
        """Args:
            des_pose: URPose
            des_gripper_state: int (0=open, 1=closed)
            des_zforce: float (desired z-force in N)
            adaptive_mode: bool (whether to use adaptive z-force control)
        """
        log_cmd = Command(time=time.time() - self.t0,
                          des_pose=des_pose, des_gripper=des_gripper_state,
                          adaptive_mode=adaptive_mode, des_zforce=des_zforce,
                          controller_state=dualsense)
        self.commands.append(log_cmd)

        if self.adaptive_mode and not adaptive_mode:
            # Transitioning adaptive -> position
            self.last_step_t = time.perf_counter()
            self.last_step_end = des_pose
        else:
            self.last_step_t = time.perf_counter()
            self.last_step_end = self.des_pose

        self.des_pose = des_pose
        self.des_gripper_state = des_gripper_state
        self.des_zforce = des_zforce
        self.adaptive_mode = adaptive_mode
        return self.get_obs()

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
            threading.Thread(target=self._gripper_loop, daemon=True,)]
        if self.dataset_path is not None:
            self.threads.append(threading.Thread(target=self._logger_loop, daemon=True,))

        self.t0 = time.time()
        for thread in self.threads:
            thread.start()
        self.wait_for_obs()

    def reset(self, home_pose):
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
        if len(self.threads) > 0:
            self.stop_flag = True
            for thr in self.threads:
                thr.join()
            if self.dataset_path is not None:
                self.save_data()
        self.camera = Camera(crop_mode=self.camera_crop_mode)

        # ============================================================
        # Home / open gripper
        # ============================================================
        if self.gripper.gripstate().value != wsg.GripperState.IDLE.value:
            self.gripper.stop().wait()
            while self.gripper.gripstate().value != wsg.GripperState.IDLE.value:
                time.sleep(.1)

        g = self.gripper.home()
        g.ack.wait()

        # ============================================================
        # Move robot home (blocking)
        # ============================================================
        self.ctrl.moveL(home_pose, 0.1, 0.1)
        self.des_pose = home_pose  # Ensure robot doesn't move after homing
        self.last_step_t = -1

        # Wait for gripper homing to finish
        g.finished.wait()
        g = self.gripper.home()
        g.wait()

        # ============================================================
        # Move gripper to default open width
        # ============================================================
        g = self.gripper.move(position=self.open_width, speed=self.g_speed)
        g.finished.wait()
        self.gripper_state = 0

        # ============================================================
        # Reset observations
        # ============================================================
        self.robot_obs: list[RobotObs] = []
        self.gripper_obs: list[GripperObs] = []
        self.camera_obs: list[CameraObs] = []
        self.commands: list[Command] = []
        if self.payload_mass is not None:
            print('controller payload before:', self.recv.getPayload(), 'kg, cog', self.recv.getPayloadCog())
            self.ctrl.setPayload(self.payload_mass, list(self.payload_cog))
            time.sleep(0.2)  # let the payload change propagate before zeroing
        self.ctrl.zeroFtSensor()
        zero_pose = URPose(*self.recv.getActualTCPPose())
        self._ft_zero_rot = R.from_rotvec([zero_pose.rx, zero_pose.ry, zero_pose.rz])

        print('payload kg', self.recv.getPayload())
        print('payload cog', self.recv.getPayloadCog())

        print('Environment reset complete.')

    def close(self):
        self.stop_flag = True
        for thr in self.threads:
            thr.join()
        self.camera.close()
        if self.dataset_path is not None:
            self.save_data()

    def init_period(self):
        self.period_init = time.perf_counter()

    def wait_period(self):
        """
        Waits for a time corresponding to `input_frequency`. Expects `init_period()` to be called at the top of the loop.
        """
        delta = time.perf_counter() - self.period_init
        sleep_time = max(0, 1 / self.input_frequency - delta)
        time.sleep(sleep_time)

    def interpolate(self):
        t = time.perf_counter() - self.last_step_t
        perc = min(1, t * self.input_frequency)

        actual_pose = self.last_step_end
        des_pose = self.des_pose

        interp_position = (
            actual_pose.x + perc * (des_pose.x - actual_pose.x),
            actual_pose.y + perc * (des_pose.y - actual_pose.y),
            actual_pose.z + perc * (des_pose.z - actual_pose.z),
        )
        R1 = R.from_rotvec([actual_pose.rx, actual_pose.ry, actual_pose.rz])
        R2 = R.from_rotvec([des_pose.rx, des_pose.ry, des_pose.rz])
        delta_theta = (R1.inv() * R2).magnitude()
        if delta_theta < 1e-6:
            interp_orientation = R1.as_rotvec()
        else:
            interp_orientation = slerp(R1, R2, perc).as_rotvec()
        return URPose(*interp_position, *interp_orientation)

    _GHAT = np.array([0., 0., 1.])  # gravity direction in the base frame (unit)

    def compensate_gravity(self, force, actual_pose):
        """Remove the orientation-dependent residual from a getActualTCPForce() wrench.

        With the payload configured correctly the controller handles the bulk of
        gravity compensation; the fitted B matrix captures what leaks through
        (payload config error, sensor gain/crosstalk on the payload weight),
        zeroed at the zeroFtSensor() orientation:

            f_tool = B (R^T - R0^T) ghat,   tau_tool = c x f_tool
        """
        if self._ft_zero_rot is None:
            return np.asarray(force)
        rot = R.from_rotvec([actual_pose.rx, actual_pose.ry, actual_pose.rz])
        u = rot.inv().apply(self._GHAT) - self._ft_zero_rot.inv().apply(self._GHAT)
        f_tool = self.grav_residual_B @ u
        tau_tool = np.cross(self.grav_residual_cog, f_tool)
        artifact = np.r_[rot.apply(f_tool), rot.apply(tau_tool)]
        return np.asarray(force) - artifact

    def calibrate_gravity_residual(self, tilt_angles=(0.15, 0.3), settle_time=0.7, sample_time=1.0, save=True):
        """Fit the orientation-dependent residual left in getActualTCPForce().

        Tilts the TCP through a set of orientations in free space and fits the
        force drift as f_tool = B (R^T - R0^T) ghat (a pure payload mass error is
        the special case B = dm*g*I; a full matrix also captures sensor gain and
        crosstalk on the payload weight), plus a tool-frame lever arm `c` for the
        torque part. The zeroing orientation is revisited throughout the run to
        measure and remove the sensor's slow time drift, which would otherwise
        contaminate the fit.

        Run right after reset() (payload configured, FT sensor freshly zeroed),
        before start(), with no contact and nothing grasped. The result is saved
        to `grav_cal_path` and loaded automatically on the next Env construction.
        """
        assert self._ft_zero_rot is not None, 'call reset() first'
        assert not any(t.is_alive() for t in self.threads), 'run before start()'
        start_pose = URPose(*self.recv.getActualTCPPose())
        rot0 = self._ft_zero_rot

        def measure(rot):
            self.ctrl.moveL(URPose(start_pose.x, start_pose.y, start_pose.z, *rot.as_rotvec()), 0.1, 0.1)
            time.sleep(settle_time)
            samples = []
            t_end = time.perf_counter() + sample_time
            while time.perf_counter() < t_end:
                samples.append(self.recv.getActualTCPForce())
                time.sleep(0.008)
            return time.perf_counter(), np.mean(samples, axis=0)

        s2 = np.sqrt(0.5)
        axes = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0),
                (s2, s2, 0), (-s2, s2, 0), (s2, -s2, 0), (-s2, -s2, 0)]
        tilts = [rot0 * R.from_rotvec(np.array(ax) * ang) for ang in tilt_angles for ax in axes]

        # Interleave revisits of the zeroing orientation, where the reading should
        # stay zero; whatever accumulates there is time drift, not orientation.
        anchor_ts, anchor_ws = [], []
        tilt_ts, tilt_ws = [], []
        for i, rot in enumerate(tilts):
            if i % 4 == 0:
                t, w = measure(rot0)
                anchor_ts.append(t), anchor_ws.append(w)
            t, w = measure(rot)
            tilt_ts.append(t), tilt_ws.append(w)
        t, w = measure(rot0)
        anchor_ts.append(t), anchor_ws.append(w)
        self.ctrl.moveL(start_pose, 0.1, 0.1)

        anchor_ws = np.array(anchor_ws)
        drift = np.stack([np.interp(tilt_ts, anchor_ts, anchor_ws[:, k]) for k in range(6)], axis=1)
        meas = np.array(tilt_ws) - drift
        print(f'Sensor time drift over run: up to {np.abs(anchor_ws[:, :3]).max():.3f} N (removed)')

        # f_tool_i = B u_i, u_i = (R_i^T - R0^T) ghat  ->  least-squares for B
        u0 = rot0.inv().apply(self._GHAT)
        us = [rot.inv().apply(self._GHAT) - u0 for rot in tilts]
        f_tool = np.array([rot.inv().apply(w[:3]) for rot, w in zip(tilts, meas)])
        A = np.vstack([np.kron(np.eye(3), u) for u in us])
        x, *_ = np.linalg.lstsq(A, f_tool.ravel(), rcond=None)
        B = x.reshape(3, 3)

        # tau_tool_i = c x (B u_i) = -[B u_i]x c  ->  least-squares for c
        def skew(u):
            return np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        f_pred = np.array([B @ u for u in us])
        if np.abs(f_pred).max() < 0.05:
            c = np.zeros(3)  # residual too small to identify a lever arm; don't fit noise
        else:
            M = np.vstack([-skew(f) for f in f_pred])
            tau_tool = np.array([rot.inv().apply(w[3:]) for rot, w in zip(tilts, meas)])
            c, *_ = np.linalg.lstsq(M, tau_tool.ravel(), rcond=None)

        rms_before = np.sqrt(np.mean(f_tool ** 2))
        rms_after = np.sqrt(np.mean((f_tool - f_pred) ** 2))
        print(f'Equivalent payload config error ~{np.trace(B) / 3 / 9.81 * 1000:.0f} g')
        print(f'Force drift RMS across tilts: {rms_before:.3f} N -> {rms_after:.3f} N after fit')
        print('Fitted B [N]:\n', np.array_str(B, precision=3, suppress_small=True))

        self.grav_residual_B = B
        self.grav_residual_cog = c
        if save:
            np.savez(self.grav_cal_path, residual_B=B, residual_cog=c)
            print(f'Saved calibration to {self.grav_cal_path}')
        return B, c

    force_alpha = 0.03
    _force_filtered = np.zeros(6)

    def filter_force(self, force):
        self._force_filtered = self.force_alpha * np.array(force) + (1 - self.force_alpha) * self._force_filtered
        return self._force_filtered

    _prev_force_err = 0.

    def zforce_pid(self, actual_pose, filtered_force):
        kp = .0007
        kd = .00001
        fz = filtered_force.z
        force_err = fz - self.des_zforce
        d_force_err = (force_err - self._prev_force_err) / self.dt
        self._prev_force_err = force_err

        zdes = actual_pose.z + kp * force_err + kd * d_force_err
        return zdes

    def _control_loop(self):
        while not self.stop_flag:
            t_start = self.ctrl.initPeriod()
            actual_pose = URPose(*self.recv.getActualTCPPose())
            actual_force = URPose(*self.compensate_gravity(self.recv.getActualTCPForce(), actual_pose))
            filtered_force = URPose(*self.filter_force(actual_force))
            self.robot_obs.append(RobotObs(time=time.time() - self.t0,
                                  actual_pose=actual_pose, actual_force=actual_force,
                                  filtered_force=filtered_force))

            des_pose = self.des_pose
            des_gripper_state = self.des_gripper_state
            gripper_state = self.gripper_state

            # ----------------------------
            # gripper logic (non-blocking preferred)
            # ----------------------------
            if gripper_state != des_gripper_state:
                gs = self.gripper.gripstate().value
                if gripper_state == 0:
                    if gs != wsg.GripperState.IDLE.value:
                        self.gripper.stop().wait()
                    self.gripper.grip(force=self.g_force, width=self.g_width, speed=self.g_speed)
                    self.gripper_state = 1
                else:
                    if gs == wsg.GripperState.GRASPING.value:
                        self.gripper.stop().wait()
                        self.gripper.move(self.open_width, speed=self.g_speed)
                        self.gripper_state = 0
                    else:
                        cur_width = self.gripper_obs[-1].gripper_width
                        self.gripper.release(pullback=(self.open_width - cur_width) / 2, speed=self.g_speed)
                        self.gripper_state = 0

            # ----------------------------
            # blend + servo
            # ----------------------------
            if self.last_step_t > 0:
                # Received at least 1 input
                des_pose = self.interpolate()
            command = blend(
                actual_pose,
                des_pose,
                self.max_position_step,
                self.max_orientation_step,
            )

            # ----------------------------
            # adaptive z-force control
            # ----------------------------
            if self.adaptive_mode:
                command = command._replace(z=self.zforce_pid(actual_pose, filtered_force))
            else:
                self._prev_force_err = 0.

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
            image = self.camera.get_image().copy()
            self.camera_obs.append(CameraObs(time=time.time() - self.t0, image=image))

    def _gripper_loop(self):
        while not self.stop_flag:
            t0 = time.perf_counter()
            # print('============ SENDING QUERIES ================')
            force = self.gripper.force()
            pos = self.gripper.position()

            self.gripper_obs.append(GripperObs(time=time.time() - self.t0,
                                               gripper_width=pos.value,
                                               gripper_force=force.value))
            sleep_dur = max(0, 1.0 / self.gripper_query_frequency - (time.perf_counter() - t0))
            # print('============ QUERY RESOLVED ================')
            time.sleep(sleep_dur)

    def _logger_loop(self):
        image_path = self.epi_path / 'images'
        time_list = []
        pose_list = []
        force_list = []
        filt_force_list = []
        gpos_list = []
        gforce_list = []
        self.wait_for_obs()  # Ensure we have at least one obs before starting logging

        last_pose = None

        image_idx = 0
        tinit = time.time()
        while not self.stop_flag:
            t0 = time.perf_counter()
            obs = self.get_obs()

            # Do not log if stationary
            # cur_pose = np.r_[obs['state']['actual_pose'], obs['state']['gripper_width']]
            # delta = max(abs(cur_pose - last_pose)) if last_pose is not None else float('inf')
            # if delta < self.save_eps:
            #     sleep_time = max(0, self.save_interval - (time.perf_counter() - t0))
            #     time.sleep(sleep_time)
            #     continue
            # last_pose = cur_pose

            im_path = pathlib.Path(image_path) / f'{image_idx:06d}.png'
            cv2.imwrite(im_path, obs['image'])
            time_list.append(time.time() - tinit)
            pose_list.append(obs['state']['actual_pose'])
            force_list.append(obs['state']['actual_force'])
            filt_force_list.append(obs['state']['filtered_force'])
            gpos_list.append(obs['state']['gripper_width'])
            gforce_list.append(obs['state']['gripper_force'])
            image_idx += 1

            sleep_time = max(0, self.save_interval - (time.perf_counter() - t0))
            time.sleep(sleep_time)

        # IMPORTANT: logger must exit via stop_flag for data to be saved
        np.savez_compressed(
            self.epi_path / 'states.npz',
            time=np.array(time_list),
            pose=np.array(pose_list),
            force=np.array(filt_force_list),
            force_raw=np.array(force_list),
            gripper_width=np.array(gpos_list),
            gripper_force=np.array(gforce_list),
            metadata=self.metadata,
            allow_pickle=True
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
            f.create_dataset('camera_obs/image_bgr', data=[obs.image for obs in self.camera_obs])

            f.create_dataset('commands/time', data=[cmd.time for cmd in self.commands])
            f.create_dataset('commands/des_pose', data=[cmd.des_pose for cmd in self.commands])
            f.create_dataset('commands/des_gripper', data=[cmd.des_gripper for cmd in self.commands])
            f.create_dataset('commands/adaptive_mode', data=[cmd.adaptive_mode for cmd in self.commands])
            f.create_dataset('commands/des_zforce', data=[cmd.des_zforce for cmd in self.commands])

            control = [vars(cmd.controller_state) for cmd in self.commands]
            f.create_dataset('dualsense/time', data=[cmd.time for cmd in self.commands])
            for key in control[0].keys():
                f.create_dataset(f'dualsense/{key}', data=[item[key] for item in control])

            m = f.create_group('metadata')
            dict2hdf5(m, self.metadata)

        print(f'Data saved to {path}')
