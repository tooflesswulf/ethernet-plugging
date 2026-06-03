from scipy.spatial.transform import Rotation as R
from dualsense import DualSense
from robosuite import make
import numpy as np


class DualSenseInterface:
    gripper_state = 0
    adaptive_mode = False

    def __init__(self, start_pose, xyzspeed=0.1, rpyspeed=1.0):
        self.env = make("Lift", robots="Panda")
        self.dualsense = DualSense(self.env)
        self.dualsense.start_control()

        self.targ_pose = np.array(start_pose)
        self.targ_zforce = 0.
        self.speed = np.r_[xyzspeed, xyzspeed, xyzspeed, rpyspeed, rpyspeed, rpyspeed]

    @property
    def target_pose(self):
        return self.targ_pose
    
    @property
    def target_zforce(self):
        return self.targ_zforce

    def flip_actions(self, act):
        # Manual flips
        tr = np.array([
            [1, 0, 0],
            [0, -1, 0],
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
        if act['right_gripper']:
            self.gripper_state = 1 - self.gripper_state
        if act['toggle_zforce']:
            self.adaptive_mode = not self.adaptive_mode

        self.flip_actions(act)
        if self.adaptive_mode:
            pass # TODO: implement adaptive mode
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

        # Position: simple addition
        dpos = delta[:3] * self.speed[:3] * dt
        self.targ_pose[:3] += dpos

        # Orientation: compose delta Euler (ZYX) onto current rotation vector
        drx, dry, drz = delta[3:] * self.speed[3:] * dt
        R_cur = R.from_rotvec(self.targ_pose[3:])
        R_delta = R.from_euler('ZYX', [drz, dry, drx])
        self.targ_pose[3:] = (R_cur * R_delta).as_rotvec()

        self.targ_zforce += delta[2] * dt * zfspeed

    def activate_adaptive_mode(self):
        self.targ_zforce = self.latest_obs['state']['force'].z

    def deactivate_adaptive_mode(self):
        self.targ_pose = self.latest_obs['state']['pose']
        self.targ_zforce = 0.

    def store_obs(self, obs):
        self.latest_obs = obs
