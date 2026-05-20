from dualsense import DualSense
from robosuite import make
import time

env = make("Lift", robots="Panda")
dualsense = DualSense(env)

dualsense.start_control()
for i in range(100):
    # print(dualsense.control, dualsense.control_gripper)
    print(dualsense.state)
    print(dualsense.input2action())
    time.sleep(0.5)
