import numpy as np
from pynput import keyboard


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
        if keyboard.KeyCode.from_char('[') in self.pressed:
            self.gripper = 0
        if keyboard.KeyCode.from_char(']') in self.pressed:
            self.gripper = 1


from dualsense import DualSense
from robosuite import make


class DualSenseInterface:
    grip_lock = 0
    gripper = 0

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
        if act['right_gripper'] == -1:
            self.gripper = 0
            self.grip_lock = 0
        elif self.grip_lock == 1:
            self.gripper = 0
        else:
            self.gripper = 1
            self.grip_lock = 1

        self.targ_pose = self.targ_pose + delta * self.speed * dt
