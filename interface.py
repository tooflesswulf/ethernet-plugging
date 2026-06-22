from scipy.spatial.transform import Rotation as R
from dualsense import DualSense
from robosuite import make
import numpy as np


class DualSenseInterface:
    gripper_state = 0
    adaptive_mode = False

    def __init__(self, start_pose, xyzspeed=0.1, rpyspeed=1.0, forcespeed=1.0, enable_zadaptive=True):
        self.env = make("Lift", robots="Panda")
        self.dualsense = DualSense(self.env)
        self.dualsense.start_control()

        self.targ_pose = np.array(start_pose)
        self.targ_zforce = 0.
        self.speed = np.r_[xyzspeed, xyzspeed, xyzspeed, rpyspeed, rpyspeed, 2 * rpyspeed]
        self.zfspeed = forcespeed
        self.enable_zadaptive = enable_zadaptive

    @property
    def target_pose(self):
        return self.targ_pose

    @property
    def target_zforce(self):
        return self.targ_zforce

    def flip_actions(self, act):
        # Manual flips
        tr = np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ])
        delta = act['right_delta']
        delta[:3] = tr @ delta[:3]

        # Manual flip rotations
        drx, dry, drz = delta[3:]
        delta[3:] = dry, drx, -drz
        return delta

    def update(self, dt):
        act = self.dualsense.input2action()
        if act is None:
            print('Act is None, skipping update')
            return -1
        self.act = act
        if act['right_gripper']:
            self.gripper_state = 1 - self.gripper_state
        if self.enable_zadaptive and act['toggle_zforce']:
            if self.adaptive_mode:
                self.adaptive_mode = False
                self.deactivate_adaptive_mode()
            else:
                self.adaptive_mode = True
                self.activate_adaptive_mode()

        self.flip_actions(act)
        if self.adaptive_mode:
            self.update_force_mode(act, dt)
        else:
            self.update_pos_mode(act, dt)

    def update_pos_mode(self, act, dt):
        delta = act['right_delta']

        # Position: simple addition
        dpos = delta[:3] * self.speed[:3] * dt
        self.targ_pose[:3] += dpos

        # Orientation: compose delta Euler (ZYX) onto current rotation vector
        drx, dry, drz = delta[3:] * self.speed[3:] * dt
        R_cur = R.from_rotvec(self.targ_pose[3:])
        R_delta = R.from_euler('ZYX', [drz, dry, drx])
        self.targ_pose[3:] = (R_cur * R_delta).as_rotvec()

    def update_force_mode(self, act, dt):
        delta = act['right_delta']
        self.targ_zforce += delta[2] * dt * self.zfspeed
        delta[2] = 0

        # Position: simple addition
        dpos = delta[:3] * self.speed[:3] * dt
        self.targ_pose[:3] += dpos

        # Orientation: compose delta Euler (ZYX) onto current rotation vector
        drx, dry, drz = delta[3:] * self.speed[3:] * dt
        R_cur = R.from_rotvec(self.targ_pose[3:])
        R_delta = R.from_euler('ZYX', [drz, dry, drx])
        self.targ_pose[3:] = (R_cur * R_delta).as_rotvec()

    def activate_adaptive_mode(self):
        self.targ_zforce = self.latest_obs['state']['filtered_force'].z

    def deactivate_adaptive_mode(self):
        self.targ_pose = np.array(self.latest_obs['state']['actual_pose'])
        self.targ_zforce = 0.

    def store_obs(self, obs):
        self.latest_obs = obs

    def residual_action(self, des_pose, dt):
        delta = self.act['right_delta'] * 5

        # Position: simple addition
        dpos = delta[:3] * self.speed[:3] * dt
        new_des_pos = des_pose[:3] + dpos

        # Orientation: compose delta Euler (ZYX) onto current rotation vector
        drx, dry, drz = delta[3:] * self.speed[3:] * dt
        R_cur = R.from_rotvec(self.targ_pose[3:])
        R_delta = R.from_euler('ZYX', [drz, dry, drx])
        new_des_ori = (R_cur * R_delta).as_rotvec()
        return np.r_[new_des_pos, new_des_ori]
