import rtde_control
import rtde_receive
import numpy as np
import threading
import time

from util import URPose, blend
from camera import Camera
import wsg


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
        max_position_step=(0.008, 0.008, 0.008),
        max_orientation_step=0.02,
        lookahead_time=0.1,
        servo_gain=500,
    ):
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
        self.home_pose = URPose(-0.125, 0.545, 0.305, 2.44, 2.44, 0.653)
        self.gripper_state = 0  # 0=open, 1=closed

        print("Initializing environment...")
        print(f"Robot IP:   {robot_ip}")
        print(f"Gripper IP: {gripper_ip}")
        print(f"Servo  URPose(-0.125, 0.545, 0.305, 2.44, 2.44, 0.653) frequency: {servo_frequency} Hz")

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

        print("Environment reset complete.")

        # return initial observation
        obs = {
            "rgb": self.camera.get_rgb(),
            "state": {
                # "actual_pose": self.home_pose,
                # "gripper_width": self.open_width,
                # "gripper_force": 0.0,
            }
        }

        return obs

    def flip_gripper(self):
        """
        Toggle gripper state.

        Current convention:
            0 = open
            1 = closed
        """

        # Avoid interrupting an ongoing gripper command
        if self.gripper._pending_action is not None:
            return

        # ============================================================
        # Close gripper
        # ============================================================
        if self.gripper_state == 0:
            g = self.gripper.grip(
                force=40,
                width=20,
                speed=50,
            )
            self.gripper_state = 1

        # ============================================================
        # Open gripper
        # ============================================================
        else:
            g = self.gripper.release(
                pullback=10,
                speed=50,
            )
            self.gripper_state = 0

        # Wait until command acknowledged
        g.ack.wait()

    def step(self, actual_pose, des_pose, des_gripper_state):
        """
        Execute one control step.

        Args:
            actual_pose: URPose
                Current end-effector pose.
            des_pose: URPose
                Desired end-effector pose.

            des_gripper_state: int
                0 = open
                1 = closed

        Returns:
            obs: dict
                {
                    "rgb": np.ndarray,
                    "state": {
                        "actual_pose": URPose,
                        "gripper_width": float,
                        "gripper_force": float,
                    }
                }
        """

        # ============================================================
        # Update gripper state if needed
        # ============================================================
        if des_gripper_state != self.gripper_state:
            self.flip_gripper()

        # ============================================================
        # Compute smooth servo command
        # ============================================================
        command = blend(
            p_start=actual_pose,
            p_end=des_pose,
            max_position_step=self.max_position_step,
            max_orientation_step=self.max_orientation_step,
        )
        # return  command
        # print(f"Command: {command}")
        # ============================================================
        # Send servo command
        # ============================================================
        self.ctrl.servoL(
            command,
            0.0,
            0.0,
            self.dt,
            self.lookahead_time,
            self.servo_gain,
        )

        self.ctrl.waitPeriod(self.dt)

        # ============================================================
        # Read latest observation
        # ============================================================
        # latest_pose = URPose(*self.recv.getActualTCPPose())
        # gripper_width, gripper_force = self.gripper.position(), self.gripper.force()
        obs = {
            # "rgb": self.camera.get_rgb(),
            "state": {
                # "actual_pose": latest_pose,
                # "gripper_width": gripper_width,
                # "gripper_force": gripper_force,
            },
        }

        return obs

    def close(self):
        """
        Clean up resources.
        """
        self.camera.close()

    # ================================================================
    # Utility functions
    # ================================================================
