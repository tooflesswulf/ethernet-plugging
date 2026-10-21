"""
Twist / wrench actions (agent/utils/se3.py): dataset label -> anchor action -> what eval
hands Env.step is the identity, and the anchor chart is the pose actions' one.

    python -m pytest tests/test_fv_actions.py
"""
import numpy as np
from scipy.spatial.transform import Rotation as R, RigidTransform as Tf

from agent.utils import se3
from agent.dataset.sequence import StitchedSequenceDataset
from robot import fdcc
from scripts.rawdata_to_dataset import body_twist_wrench

rng = np.random.default_rng(0)


def random_poses(n, spread=0.05):
    p0 = np.r_[-0.12, 0.5, 0.1, R.random(random_state=1).as_rotvec()]
    p = p0[None].repeat(n, 0)
    p[:, :3] += spread * rng.standard_normal((n, 3))
    p[:, 3:] = (R.from_rotvec(spread * 5 * rng.standard_normal((n, 3))) * R.from_rotvec(p0[3:])).as_rotvec()
    return p0, p


def anchor_actions(p0, poses):
    """local_delta pose actions [w; v] and T_0i, as StitchedSequenceDataset computes them."""
    T_0i = StitchedSequenceDataset.relative_transforms(poses, p0)
    t0 = Tf.from_components(p0[:3], R.from_rotvec(p0[3:]))
    xi = (t0.inv() * Tf.from_components(poses[:, :3], R.from_rotvec(poses[:, 3:]))).as_exp_coords()
    return xi, T_0i


def test_round_trip():
    p0, poses = random_poses(16)
    xi, T_0i = anchor_actions(p0, poses)
    # eval rebuilds T_0i from the predicted pose actions alone (DiffusionPolicy.decode_feedforward)
    np.testing.assert_allclose(Tf.from_exp_coords(xi).as_matrix(), T_0i, atol=1e-12)
    V, F = rng.standard_normal((16, 6)), rng.standard_normal((16, 6))
    a_V, a_F = se3.twist_to_anchor(T_0i, V), se3.wrench_to_anchor(T_0i, F)
    np.testing.assert_allclose(se3.twist_from_anchor(T_0i, a_V), V, atol=1e-12)
    np.testing.assert_allclose(se3.wrench_from_anchor(T_0i, a_F), F, atol=1e-12)
    # power is frame-invariant
    np.testing.assert_allclose(np.sum(a_V * a_F, axis=1), np.sum(V * F, axis=1), atol=1e-12)


def test_twist_is_pose_action_derivative():
    """Along a motion, d xi / dt ~ the anchor twist (exactly at xi = 0, first order nearby)."""
    p0, _ = random_poses(1)
    V_b = np.r_[0.02, -0.01, 0.03, 0.2, -0.1, 0.3]          # constant body twist [v; w]
    dt = 1e-4
    T0 = fdcc.pose_to_T(p0)
    traj = [T0 @ fdcc.se3_exp(V_b * k * dt) for k in range(3)]
    poses = np.array([np.r_[T[:3, 3], R.from_matrix(T[:3, :3]).as_rotvec()] for T in traj])
    xi, T_0i = anchor_actions(p0, poses)
    a_V = se3.twist_to_anchor(T_0i, np.repeat(V_b[None], 3, 0))
    np.testing.assert_allclose((xi[2] - xi[0]) / (2 * dt), a_V[1], atol=1e-6)


def test_dataset_labels_match_fdcc():
    """rawdata_to_dataset's vectorized labels == fdcc.body_twist / fdcc.body_wrench."""
    n = 50
    _, poses = random_poses(n, spread=0.01)
    t = np.arange(n) * 0.002
    W = rng.standard_normal((n, 6)) * [10, 10, 10, 1, 1, 1]
    tcp = np.array([0, 0, 0.1537, 0, 0, 0.])
    tw, wr = body_twist_wrench(t, poses, W, tcp, lpf_hz=0)
    for i in range(1, n - 1):
        np.testing.assert_allclose(tw[i], fdcc.body_twist(poses[i - 1], poses[i + 1], t[i + 1] - t[i - 1]), atol=1e-9)
        np.testing.assert_allclose(wr[i], fdcc.body_wrench(poses[i], W[i], tcp), atol=1e-9)


def test_wrench_target_cancels_in_controller():
    """Impedance.wrench(): the measured wrench as the expected one reads (about) zero."""
    imp = fdcc.Impedance(fdcc.ImpedanceParams(lowpass_hz=0.0))
    pose = np.r_[-0.12, 0.5, 0.1, 2.4, 2.4, 0.6]
    W = np.r_[3.0, -12.0, 8.0, 0.3, 0.2, -0.1]
    F_b = fdcc.body_wrench(pose, W, imp.p.tcp_offset, imp.p.ft_ref, imp.p.ft_sign)
    assert np.linalg.norm(imp.wrench(pose, W)) > 5
    np.testing.assert_allclose(imp.wrench(pose, W, F_b), 0, atol=1e-9)
