import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R


class KeyboardInterface:
    pressed = set()
    gripper = 1

    def __init__(self, start_pose, xyzspeed=0.1, rpyspeed=1.0):
        self.targ_pose = np.array(start_pose)
        self.speed = np.r_[xyzspeed, xyzspeed, xyzspeed, rpyspeed, rpyspeed, rpyspeed]

        self.klistener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.klistener.start()

    @property
    def target_pose(self):
        return self.targ_pose

    def on_press(self, key):
        self.pressed.add(key)

    def on_release(self, key):
        self.pressed.discard(key)

    def update(self, dt):
        if keyboard.Key.up in self.pressed or keyboard.KeyCode.from_char('w') in self.pressed:
            self.targ_pose[0] += self.speed[0] * dt
        if keyboard.Key.down in self.pressed or keyboard.KeyCode.from_char('s') in self.pressed:
            self.targ_pose[0] -= self.speed[0] * dt
        if keyboard.Key.left in self.pressed or keyboard.KeyCode.from_char('a') in self.pressed:
            self.targ_pose[1] += self.speed[1] * dt
        if keyboard.Key.right in self.pressed or keyboard.KeyCode.from_char('d') in self.pressed:
            self.targ_pose[1] -= self.speed[1] * dt
        if keyboard.KeyCode.from_char('q') in self.pressed:
            self.targ_pose[2] += self.speed[2] * dt
        if keyboard.KeyCode.from_char('e') in self.pressed:
            self.targ_pose[2] -= self.speed[2] * dt

        if keyboard.Key.space in self.pressed:
            grip_signal = 1
        else:
            grip_signal = 0


from dualsense import DualSense
from robosuite import make


class DualSenseInterface:
    last_grip_signal = 0
    gripper_state = 0

    def __init__(self, start_pose, xyzspeed=0.1, rpyspeed=1.0):
        self.env = make("Lift", robots="Panda")
        self.dualsense = DualSense(self.env)
        self.dualsense.start_control()

        self.targ_pose = np.array(start_pose)
        self.speed = np.r_[xyzspeed, xyzspeed, xyzspeed, rpyspeed, rpyspeed, rpyspeed]

    @property
    def target_pose(self):
        return self.targ_pose

    def update(self, dt):
        act = self.dualsense.input2action()
        delta = act['right_delta']
        grip_signal = (1 if act['right_gripper'] == 1 else 0)

        if grip_signal == 1 and self.last_grip_signal == 0:
            # Toggle gripper state on rising edge of grip signal
            self.gripper_state = 1 - self.gripper_state

        # Manual flips
        flips = np.array([1, -1, 1])

        # Position: simple addition
        dpos = delta[:3] * flips * self.speed[:3] * dt
        self.targ_pose[:3] += dpos

        # Orientation: compose delta Euler (ZYX) onto current rotation vector
        drx, dry, drz = delta[3:] * self.speed[3:] * dt

        # Manually flip rotations
        drx, dry = -dry, -drx

        R_cur = R.from_rotvec(self.targ_pose[3:])
        R_delta = R.from_euler('ZYX', [drz, dry, drx])
        self.targ_pose[3:] = (R_cur * R_delta).as_rotvec()

        self.last_grip_signal = grip_signal
