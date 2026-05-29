import os, cv2, copy, time, threading
import imageio, shutil
from collections import namedtuple

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

import rtde_control
import rtde_receive

import wsg
from camera import Camera


class RotationVector(namedtuple("RotationVector", ("rx", "ry", "rz"))):
    pass


class URPose(namedtuple("URPose", ("x", "y", "z") + RotationVector._fields)):
    pass


def slerp(q1, q2, fraction):
    """
    Spherical linear interpolation between two scipy Rotation objects.
    """
    quats = R.from_quat(np.array([q1.as_quat(), q2.as_quat()]))
    q = Slerp([0, 1], quats)(fraction)
    return q

def blend(
    p_start: URPose,
    p_end: URPose,
    max_position_step =  [0.008, 0.008, 0.008],
    max_orientation_step = 0.02,
):
    """
    Smoothly blend current pose toward target pose while limiting
    translational and rotational step sizes.
    """

    blended_position = (
        p_start.x
        + np.clip(
            p_end.x - p_start.x,
            -max_position_step[0],
            max_position_step[0],
        ),
        p_start.y
        + np.clip(
            p_end.y - p_start.y,
            -max_position_step[1],
            max_position_step[1],
        ),
        p_start.z
        + np.clip(
            p_end.z - p_start.z,
            -max_position_step[2],
            max_position_step[2],
        ),
    )

    R1 = R.from_rotvec([p_start.rx, p_start.ry, p_start.rz])
    R2 = R.from_rotvec([p_end.rx, p_end.ry, p_end.rz])

    delta_theta = (R1.inv() * R2).magnitude()

    if delta_theta == 0:
        blended_orientation = R1
    else:
        frac = np.clip(
            max_orientation_step / delta_theta,
            0,
            1,
        )
        blended_orientation = slerp(R1, R2, frac)

    return URPose(
        *blended_position,
        *blended_orientation.as_rotvec(),
    )

# class Env:
#     """
#     Minimal robot environment wrapper for:
#         - Teleoperation
#         - Robot policy evaluation
#         - Data collection

#     Components:
#         - Single RGB camera
#         - UR robot arm
#         - WSG gripper
#     """

#     def __init__(
#         self,
#         robot_ip="192.168.0.100", # could be 101 or 100 depending on your setup
#         gripper_ip="192.168.0.20",
#         camera_crop_mode=1, # crop on the right half of the image to focus on the workspace
#         servo_frequency=500,
#         max_position_step=(0.008, 0.008, 0.008),
#         max_orientation_step=0.02,
#         lookahead_time=0.1,
#         servo_gain=500,
#     ):
#         # ============================================================
#         # Camera
#         # ============================================================
#         self.camera = Camera(crop_mode=camera_crop_mode)

#         # ============================================================
#         # Robot interfaces
#         # ============================================================
#         self.robot_ip = robot_ip
#         self.gripper_ip = gripper_ip

#         self.ctrl = rtde_control.RTDEControlInterface(robot_ip)
#         self.recv = rtde_receive.RTDEReceiveInterface(robot_ip)

#         self.gripper = wsg.WSG(ip=gripper_ip)
#         self.pos_query, self.force_query, self.g_pos, self.g_force = None, None, -1, -1
#         # ============================================================
#         # Servo parameters
#         # ============================================================
#         self.servo_frequency = servo_frequency
#         self.dt = 1.0 / servo_frequency

#         self.max_position_step = np.array(max_position_step)
#         self.max_orientation_step = max_orientation_step

#         self.lookahead_time = lookahead_time
#         self.servo_gain = servo_gain

#         # ============================================================
#         # Internal states
#         # ============================================================
#         self.open_width = 35
#         self.home_pose =  URPose(-0.125, 0.545, 0.305, 2.44, 2.44, 0.653) 
#         self.gripper_state = 0  # 0=open, 1=closed

#         print("Initializing environment...")
#         print(f"Robot IP:   {robot_ip}")
#         print(f"Gripper IP: {gripper_ip}")
#         print(f"Servo  URPose(-0.125, 0.545, 0.305, 2.44, 2.44, 0.653) frequency: {servo_frequency} Hz")

#     def reset(self, home_pose):
#         """
#         Reset environment:
#             1. Open/home the gripper
#             2. Move robot to home pose
#             3. Set gripper to default width

#         Args:
#             home_pose: URPose
#         """

#         print("Resetting environment...")

#         # ============================================================
#         # Home / open gripper
#         # ============================================================
#         g = self.gripper.home()
#         g.ack.wait()

#         # ============================================================
#         # Move robot home (blocking)
#         # ============================================================
#         self.ctrl.moveL(home_pose, 0.1, 0.1)

#         # Wait for gripper homing to finish
#         g.finished.wait()

#         # ============================================================
#         # Move gripper to default open width
#         # ============================================================
#         g = self.gripper.move(position=self.open_width, speed=50)
#         g.finished.wait()

#         self.gripper_state = 0
#         if self.pos_query is None:
#             self.pos_query = self.gripper.position()
#         if self.force_query is None:
#             self.force_query = self.gripper.force()
#         if self.pos_query.is_set():
#             self.g_pos = self.pos_query.value
#             self.pos_query = None
#         if self.force_query.is_set():
#             self.g_force = self.force_query.value
#             self.force_query = None

#         print("Environment reset complete.")

#         # return initial observation
#         obs = {
#             "rgb": self.camera.get_rgb(),
#             "state": {
#                 "actual_pose": self.home_pose,
#                 "gripper_width": self.g_pos,
#                 "gripper_force": self.g_force,
#             }
#         }

#         return obs

#     def flip_gripper(self):
#         """
#         Toggle gripper state.

#         Current convention:
#             0 = open
#             1 = closed
#         """

#         # Avoid interrupting an ongoing gripper command
#         if self.gripper._pending_action is not None:
#             return

#         # ============================================================
#         # Close gripper
#         # ============================================================
#         if self.gripper_state == 0:
#             g = self.gripper.grip(
#                 force=40,
#                 width=20,
#                 speed=50,
#             )
#             self.gripper_state = 1

#         # ============================================================
#         # Open gripper
#         # ============================================================
#         else:
#             g = self.gripper.release(
#                 pullback=10,
#                 speed=50,
#             )
#             self.gripper_state = 0

#         # Wait until command acknowledged
#         g.ack.wait()
       
    
#     def step(self, actual_pose, des_pose, des_gripper_state):
#         """
#         Execute one control step.

#         Args:
#             actual_pose: URPose
#                 Current end-effector pose.
#             des_pose: URPose
#                 Desired end-effector pose.

#             des_gripper_state: int
#                 0 = open
#                 1 = closed

#         Returns:
#             obs: dict
#                 {
#                     "rgb": np.ndarray,
#                     "state": {
#                         "actual_pose": URPose,
#                         "gripper_width": float,
#                         "gripper_force": float,
#                     }
#                 }
#         """

       
#         # ============================================================
#         # Update gripper state if needed
#         # ============================================================
#         if des_gripper_state != self.gripper_state:
#             self.flip_gripper()

#         # ============================================================
#         # Compute smooth servo command
#         # ============================================================
#         command = blend(
#             p_start=actual_pose,
#             p_end=des_pose,
#             max_position_step=self.max_position_step,
#             max_orientation_step=self.max_orientation_step,
#         )
#         # return  command
#         # print(f"Command: {command}")
#         # ============================================================
#         # Send servo command
#         # ============================================================
#         self.ctrl.servoL(
#             command,
#             0.0,
#             0.0,
#             self.dt,
#             self.lookahead_time,
#             self.servo_gain,
#         )

#         self.ctrl.waitPeriod(self.dt)

#         # ============================================================
#         # Read latest observation
#         # ============================================================
#         # latest_pose = URPose(*self.recv.getActualTCPPose())
#         # gripper_width, gripper_force = self.gripper.position(), self.gripper.force()
#         obs = {
#             # "rgb": self.camera.get_rgb(),
#             "state": {
#                 # "actual_pose": latest_pose,
#                 # "gripper_width": gripper_width,
#                 # "gripper_force": gripper_force,
#             },
#         }

#         return obs

#     def close(self):
#         """
#         Clean up resources.
#         """
#         self.camera.close()


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
        
        robot_ip="192.168.0.100", # could be 101 or 100 depending on your setup
        gripper_ip="192.168.0.20",
        camera_crop_mode=1, # crop on the right half of the image to focus on the workspace
        servo_frequency=500,
        max_position_step=(0.008, 0.008, 0.008),
        max_orientation_step=0.02,
        lookahead_time=0.1,
        servo_gain=500,
        dataset_path = None,
        save_interval = 0.1,
        save_eps = 1e-3,
    ):
        # ============================================================
        # Camera
        # ============================================================
        self.camera_crop_mode = camera_crop_mode
        
        # ============================================================
        # Robot interfaces
        # ============================================================
        self.robot_ip = robot_ip
        self.gripper_ip = gripper_ip

        self.ctrl = rtde_control.RTDEControlInterface(self.robot_ip)
        self.recv = rtde_receive.RTDEReceiveInterface(self.robot_ip)

        self.gripper = wsg.WSG(ip=self.gripper_ip)
        self.pos_query, self.force_query, self.g_pos, self.g_force = None, None, -1, -1
        # ============================================================
        # Servo parameters
        # ============================================================
        self.servo_frequency = servo_frequency
        self.dt = 1.0 / servo_frequency

        self.max_position_step = np.array(max_position_step)
        self.max_orientation_step = max_orientation_step

        self.lookahead_time = lookahead_time
        self.servo_gain = servo_gain

        # ============================================================
        # Internal states
        # ============================================================
        self.open_width = 35
        self.home_pose =  URPose(-0.125, 0.545, 0.305, 2.44, 2.44, 0.653) 
        self.gripper_state = 0  # 0=open, 1=closed
        self.des_pose, self.des_gripper_state = None, self.gripper_state

        print("Initializing environment...")
        print(f"Robot IP:   {robot_ip}")
        print(f"Gripper IP: {gripper_ip}")
        print(f"Servo  URPose(-0.125, 0.545, 0.305, 2.44, 2.44, 0.653) frequency: {servo_frequency} Hz")

        # ----------------------------
        # threading
        # ----------------------------
        self.stop_flag = False
        self.lock = threading.Lock()
        self.obs_lock = threading.Lock()
        self.control_thread = None
        self.obs_thread = None
        self.logger_thread = None
        # ----------------------------
        # observation buffer
        # ----------------------------
        self.latest_obs = None
        self.dataset_path = dataset_path
        self.save_interval = save_interval # save thread loop interval in seconds
        self.save_eps = save_eps
        self.image_idx = 0

    def get_gripper_state(self):
        if self.pos_query is None:
            self.pos_query = self.gripper.position()
        if self.force_query is None:
            self.force_query = self.gripper.force()
        if self.pos_query.is_set():
            self.g_pos = self.pos_query.value
            self.pos_query = None
        if self.force_query.is_set():
            self.g_force = self.force_query.value
            self.force_query = None

        return self.g_pos, self.g_force

    def reset(self, home_pose):
        """
        Reset environment:
            1. Open/home the gripper
            2. Move robot to home pose
            3. Set gripper to default width

        Args:
            home_pose: URPose
        """

        print("Resetting environment...")
        self.camera = Camera(crop_mode=self.camera_crop_mode)
        # ============================================================
        # Home / open gripper
        # ============================================================
        g = self.gripper.home()
        g.ack.wait()

        # ============================================================
        # Move robot home (blocking)
        # ============================================================
        self.ctrl.moveL(home_pose, 0.1, 0.1)

        # Wait for gripper homing to finish
        g.finished.wait()

        # ============================================================
        # Move gripper to default open width
        # ============================================================
        g = self.gripper.move(position=self.open_width, speed=50)
        g.finished.wait()

        self.gripper_state = 0
        if self.pos_query is None:
            self.pos_query = self.gripper.position()
        if self.force_query is None:
            self.force_query = self.gripper.force()
        if self.pos_query.is_set():
            self.g_pos = self.pos_query.value
            self.pos_query = None
        if self.force_query.is_set():
            self.g_force = self.force_query.value
            self.force_query = None

        print("Environment reset complete.")

        # return initial observation
        obs = {
            "rgb": self.camera.get_rgb(),
            "state": {
                "actual_pose": self.home_pose,
                "gripper_width": self.g_pos,
                "gripper_force": self.g_force,
            }
        }

        return obs

    def _control_loop(self):

        t_prev = time.time()

        while not self.stop_flag:

            t_start = self.ctrl.initPeriod()

            t_now = time.time()
            actual_pose = URPose(*self.recv.getActualTCPPose())

            # ----------------------------
            # read shared command
            # ----------------------------
            with self.lock:
                des_pose = self.des_pose
                des_gripper_state = self.des_gripper_state
                gripper_state = self.gripper_state

            if des_pose is None:
                des_pose = actual_pose

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
    
    def _obs_loop(self):

        while not self.stop_flag:
            self.get_gripper_state()
            obs = {
                "rgb": self.camera.get_rgb(),
                "state": {
                    "actual_pose": URPose(*self.recv.getActualTCPPose()),
                    "gripper_width": self.g_pos,
                    "gripper_force": self.g_force,
                },
            }
            with self.obs_lock:
                self.latest_obs = obs
    
    def _logger_loop(self):

        image_dir = os.path.join(self.dataset_path, "images")

        actual_pose_list = []
        gripper_width_list = []
        gripper_force_list = []
        last_pose = None
        while not self.stop_flag:

            t0 = time.time()

            with self.obs_lock:
                obs = copy.deepcopy(self.latest_obs)
            if obs is not None:

                # -----------------------------------
                # Save image
                # -----------------------------------
                rgb = obs["rgb"][:, :, ::-1]  # convert RGB to BGR for OpenCV
               
                image_path = os.path.join(
                    image_dir,
                    f"{self.image_idx}.png",
                )

                
                # -----------------------------------
                # Save states
                # -----------------------------------
                state = obs["state"]
                new_pose = np.concatenate( [ np.array(state["actual_pose"]), np.array([state["gripper_width"]]) ] )
                max_delta = max( abs(new_pose - last_pose) ) if last_pose is not None else float('inf')
                
                if max_delta < self.save_eps:
                    continue

                cv2.imwrite(
                    image_path,
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                )

                actual_pose_list.append(
                    np.array(state["actual_pose"])
                )
               
                last_pose = np.concatenate( [ np.array(state["actual_pose"]), np.array([state["gripper_width"]]) ] )

                gripper_width_list.append(
                    state["gripper_width"]
                )

                gripper_force_list.append(
                    state["gripper_force"]
                )

                self.image_idx += 1

            # -----------------------------------
            # maintain logging frequency
            # -----------------------------------
            elapsed = time.time() - t0

            sleep_time = max(
                0,
                self.save_interval - elapsed,
            )

            time.sleep(sleep_time)

        # ============================================================
        # save npz once thread exits
        # ============================================================

        np.savez_compressed(
            os.path.join(self.dataset_path, "states.npz"),

            actual_pose=np.array(actual_pose_list),
            gripper_width=np.array(gripper_width_list),
            gripper_force=np.array(gripper_force_list),
        )


    def step(self, des_pose, des_gripper_state):

        with self.lock:
            self.des_pose = des_pose
            self.des_gripper_state = des_gripper_state

        return self.latest_obs
    
    def start(self):

        if self.dataset_path is not None:
            # if dataset path exists, delete it to avoid confusion
            if os.path.exists(self.dataset_path):
                shutil.rmtree(self.dataset_path)

            os.makedirs(self.dataset_path, exist_ok=True)
            os.makedirs(
                os.path.join(self.dataset_path, "images"),
                exist_ok=True,
            )
           
        self.stop_flag = False
        if self.dataset_path is not None:
            self.logger_thread =threading.Thread(target=self._logger_loop, daemon=True,)
            self.logger_thread.start()
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True,)
        self.obs_thread = threading.Thread(target=self._obs_loop, daemon=True,)
        self.obs_thread.start()
        time.sleep(1.0)  # give some time for the threads to start and populate initial obs
        self.control_thread.start()
        
    def close(self):
        self.stop_flag = True
        self.control_thread.join()
        self.obs_thread.join()
        self.logger_thread.join()

        self.camera.close()