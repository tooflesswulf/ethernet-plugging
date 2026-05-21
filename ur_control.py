from collections import namedtuple
from scipy.spatial.transform import Rotation as R, Slerp
import rtde_control
import rtde_receive
import numpy as np

import interface
import wsg


class RotationVector(namedtuple('RotationVector', ('rx', 'ry', 'rz'))):
    pass


class URPose(namedtuple('URPose', ('x', 'y', 'z') + RotationVector._fields)):
    pass


def slerp(q1, q2, fraction):
    quats = R.from_quat(np.array([q1.as_quat(), q2.as_quat()]))
    q = Slerp([0, 1], quats)(fraction)
    return q


if __name__ == '__main__':
    robot_ip = '192.168.0.101'
    ctrl = rtde_control.RTDEControlInterface(robot_ip)
    recv = rtde_receive.RTDEReceiveInterface(robot_ip)
    gripper = wsg.WSG(ip='192.168.0.20')
    home_pose = URPose(0.14, 0.5, 0.5, 2.22, 2.22, 0)  # TODO: actual home pose

    # Move to home, blocking
    _g = gripper.home()
    _g.ack.wait()
    ctrl.moveL(home_pose, 0.1, 0.1)
    _g.finished.wait()
    _g = gripper.move(position=35, speed=50)
    _g.finished.wait()
    gripper_state = 0 # 0=open, 1=closed
    print('Homed, starting control loop...')

    servo_frequency = 500  # hz
    dt = 1 / servo_frequency
    max_position_step = [0.008, 0.008, 0.008]  # m
    max_orientation_step = 0.02  # rad
    lookahead_time = 0.1
    servo_gain = 500
    # iface = interface.KeyboardInterface(home_pose, xyzspeed=0.003, rpyspeed=0.1)
    iface = interface.DualSenseInterface(home_pose, xyzspeed=0.01, rpyspeed=0.1)

    def blend(p_start: URPose, p_end: URPose):
        blended_position = (
            p_start.x + np.clip(p_end.x - p_start.x, -max_position_step[0], max_position_step[0]),
            p_start.y + np.clip(p_end.y - p_start.y, -max_position_step[1], max_position_step[1]),
            p_start.z + np.clip(p_end.z - p_start.z, -max_position_step[2], max_position_step[2])
        )

        R1 = R.from_rotvec([p_start.rx, p_start.ry, p_start.rz])
        R2 = R.from_rotvec([p_end.rx, p_end.ry, p_end.rz])
        delta_theta = (R1.inv() * R2).magnitude()
        if delta_theta == 0:
            blended_orientation = R1
        else:
            frac = np.clip(max_orientation_step / delta_theta, 0, 1)
            blended_orientation = slerp(R1, R2, frac)

        out = URPose(*blended_position, *blended_orientation.as_rotvec())
        return out

    while True:
        actual_pose = URPose(*recv.getActualTCPPose())

        # Get desired pose
        iface.update(dt)
        des_pose = URPose(*iface.target_pose)

        if gripper._pending_action is None and gripper_state != iface.gripper_state:
            if gripper_state == 0:
                _g = gripper.grip(force=40, width=20, speed=50)
                gripper_state = 1
            else:
                _g = gripper.release(pullback=10, speed=50)
                gripper_state = 0
            _g.ack.wait()

        # Compute next servo command
        command = blend(actual_pose, des_pose)
        ctrl.servoL(
            command,
            0.0, 0.0, dt, lookahead_time, servo_gain)
        ctrl.waitPeriod(dt)
