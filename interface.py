from scipy.spatial.transform import Rotation as R
from dualsense import DualSense
import numpy as np

from env import GRIP_OPEN, GRIP_CLOSED


class DualSenseInterface:
    gripper_state = GRIP_OPEN
    adaptive_mode = False

    def __init__(self, start_pose, xyzspeed=0.1, rpyspeed=1.0, forcespeed=1.0, enable_zadaptive=True):
        self.dualsense = DualSense()
        self.dualsense.start_control()

        self.targ_pose = np.array(start_pose)
        # Vestigial: impedance control has no z-force setpoint (adaptive_mode just
        # lowers Kz). Kept at 0 so the Command/HDF5 schema and interrupt_sequence's
        # existing assignments keep working on old episodes.
        self.targ_zforce = 0.
        self.speed = np.r_[xyzspeed, xyzspeed, xyzspeed, rpyspeed, rpyspeed, 2 * rpyspeed]
        self.zfspeed = forcespeed   # unused; constructor arg kept for callers
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
            self.gripper_state = GRIP_OPEN if self.gripper_state == GRIP_CLOSED else GRIP_CLOSED
        if self.enable_zadaptive and act['toggle_zforce']:
            if self.adaptive_mode:
                self.adaptive_mode = False
                self.deactivate_adaptive_mode()
            else:
                self.adaptive_mode = True
                self.activate_adaptive_mode()

        self.flip_actions(act)
        # adaptive_mode is now purely a gain schedule in Env: it lowers Kz so the
        # operator commands force by driving the z target into the surface
        # (F_z = Kz * penetration). The joystick mapping is therefore identical in
        # both modes, and the old update_force_mode -- which zeroed delta[2] and
        # redirected the z stick into targ_zforce -- is gone.
        self.update_pos_mode(act, dt)

    def update_pos_mode(self, act, dt):
        delta = act['right_delta']

        # Position: simple addition
        dpos = delta[:3] * self.speed[:3] * dt
        self.targ_pose[:3] += dpos

        # Orientation: compose delta Euler (ZYX) onto current rotation vector
        drx, dry, drz = delta[3:] * self.speed[3:] * dt
        R_cur = R.from_rotvec(self.targ_pose[3:])
        R_delta = R.from_euler('ZYX', [0, dry, drx])
        # self.targ_pose[3:] = (R_cur * R_delta).as_rotvec() # Local rotation
        Rz = R.from_euler('ZYX', [drz, 0, 0])
        # R_delta = R.from_euler('ZYX', [-drz, -dry, drx])
        self.targ_pose[3:] = (Rz * R_cur * R_delta).as_rotvec() # Mixed rotation

    def activate_adaptive_mode(self):
        # Nothing to latch: entering the soft gain set with e = 0 is already
        # bumpless, and Env ramps the gain change over mode_blend_time.
        pass

    def deactivate_adaptive_mode(self):
        # LOAD-BEARING, not housekeeping. In adaptive mode the z target sits
        # centimetres below the surface -- that offset IS the force command.
        # Restoring stiff Kz against a 4 cm error would command ~60 N, so the
        # target must be re-synced to the measured pose FIRST; Env then ramps
        # the gains over mode_blend_time.
        self.targ_pose = np.array(self.latest_obs['state']['actual_pose'])

    def store_obs(self, obs):
        self.latest_obs = obs

    def residual_action(self, des_pose, dt):
        delta = self.act['right_delta'] * 5

        # Position: simple addition
        dpos = delta[:3] * self.speed[:3] * dt
        new_des_pos = des_pose[:3] + dpos

        # Orientation: compose delta Euler (ZYX) onto current rotation vector
        drx, dry, drz = delta[3:] * self.speed[3:] * dt
        R_cur = R.from_rotvec(des_pose[3:])
        R_delta = R.from_euler('ZYX', [-drz, -dry, drx])
        new_des_ori = (R_delta * R_cur).as_rotvec() # Global rotation
        return np.r_[new_des_pos, new_des_ori]
