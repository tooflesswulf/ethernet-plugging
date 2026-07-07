"""Fit the residual gravity artifact in getActualTCPForce().

The UR controller compensates the configured payload internally, but a
slightly-off payload configuration leaks a force/torque residual that drifts
as the end effector rotates. This tilts the tool through a few orientations
in free space, fits the residual, and saves it next to env.py so Env picks
it up automatically.

Run with the robot clear of contact and nothing grasped.

NOTE: the default sweep includes tool-axis rotations up to +-120 deg (needed to
separate sensor-bias artifacts from gravity-shaped ones). Make sure the gripper
can spin that far without cable snag or collision, or pass a custom orientation
list to calibrate_gravity_residual().
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from env import Env  # noqa: E402

if __name__ == '__main__':
    env = Env(dataset_path=None)
    env.reset(None)
    env.calibrate_gravity_residual()
