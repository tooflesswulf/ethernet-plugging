from scipy.spatial.transform import Rotation as R, Slerp
from collections import namedtuple
import numpy as np


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
    max_position_step=[0.008, 0.008, 0.008],
    max_orientation_step=0.02,
):
    """
    Smoothly blend current pose toward target pose while limiting
    translational and rotational step sizes.
    """

    blended_position = (
        p_start.x + np.clip(p_end.x - p_start.x,
                            -max_position_step[0], max_position_step[0]),
        p_start.y + np.clip(p_end.y - p_start.y,
                            -max_position_step[1], max_position_step[1]),
        p_start.z + np.clip(p_end.z - p_start.z,
                            -max_position_step[2], max_position_step[2]),
    )

    R1 = R.from_rotvec([p_start.rx, p_start.ry, p_start.rz])
    R2 = R.from_rotvec([p_end.rx, p_end.ry, p_end.rz])

    delta_theta = (R1.inv() * R2).magnitude()
    if delta_theta < 1e-6:
        blended_orientation = R1
    else:
        frac = np.clip(max_orientation_step / delta_theta, 0, 1,)
        blended_orientation = slerp(R1, R2, frac)

    return URPose(
        *blended_position,
        *blended_orientation.as_rotvec(),
    )
