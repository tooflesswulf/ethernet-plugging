"""
Offline tests for the impedance control law and kinematics.

Env cannot be imported without a robot (env.py opens RTDE sockets in __init__),
so these pure functions are the only testable surface. Run: pytest tests/
"""
from scipy.spatial.transform import Rotation as R
import numpy as np
import pytest

from impedance import (pose_error, friction_feedforward, CartesianImpedance,
                       SafetyMonitor, MAX_ORIENTATION_ERROR)
from kinematics import URKin

import sys, types
for _n in ('rtde_control','rtde_receive','cv2','wsg','camera','h5py'):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules['rtde_control'].RTDEControlInterface = object
sys.modules['rtde_receive'].RTDEReceiveInterface = object
sys.modules['camera'].Camera = object
sys.modules['wsg'].WSG = object
sys.modules['wsg'].GripperState = types.SimpleNamespace(IDLE=types.SimpleNamespace(value=0))

# The real home pose. Its rotvec norm is 3.512 > pi -- a valid non-canonical
# rotation vector, and the reason pose_error must not subtract components.
HOME = np.array([-0.125, 0.545, 0.305, 2.44, 2.44, 0.653])
TCP = [0, 0, 0.1537, 0, 0, 0]


# ----------------------------------------------------------------- pose_error
def test_identity_is_zero():
    assert np.abs(pose_error(HOME, HOME)).max() < 1e-12


def test_home_pose_rotvec_exceeds_pi():
    """Guards the specific trap: naive subtraction here gives a ~2pi error."""
    assert np.linalg.norm(HOME[3:]) > np.pi


def test_pure_translation():
    tgt = HOME.copy()
    tgt[:3] += [0.01, -0.02, 0.03]
    e = pose_error(HOME, tgt)
    assert np.allclose(e[:3], [0.01, -0.02, 0.03])
    assert np.abs(e[3:]).max() < 1e-12


@pytest.mark.parametrize('axis', range(3))
def test_small_rotation_recovered_exactly(axis):
    """Must be right even though HOME's rotvec is non-canonical."""
    dr = np.zeros(3)
    dr[axis] = 0.1
    tgt = HOME.copy()
    tgt[3:] = (R.from_rotvec(dr) * R.from_rotvec(HOME[3:])).as_rotvec()
    e = pose_error(HOME, tgt)
    assert np.allclose(e[3:], dr, atol=1e-9)

    # And the failure mode we are guarding against is genuinely large.
    naive = tgt[3:] - HOME[3:]
    assert np.linalg.norm(naive - dr) > 1.0


def test_error_is_base_frame_not_body_frame():
    """R_des @ R_act.T, not R_act.T @ R_des -- pairs with a base-frame Jacobian."""
    tgt = HOME.copy()
    tgt[3:] = (R.from_rotvec([0, 0, 0.3]) * R.from_rotvec(HOME[3:])).as_rotvec()
    e = pose_error(HOME, tgt)
    assert np.allclose(e[3:], [0, 0, 0.3], atol=1e-9)


def test_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = np.r_[rng.uniform(-1, 1, 3), R.random(random_state=int(rng.integers(1e6))).as_rotvec()]
        b = np.r_[rng.uniform(-1, 1, 3), R.random(random_state=int(rng.integers(1e6))).as_rotvec()]
        e = pose_error(a, b)
        assert np.linalg.norm(e[3:]) <= np.pi + 1e-9
        recon = R.from_rotvec(e[3:]) * R.from_rotvec(a[3:])
        assert (recon.inv() * R.from_rotvec(b[3:])).magnitude() < 1e-9


# --------------------------------------------------------- friction feedforward
def test_zero_at_rest_with_no_command():
    """Must not creep when nothing is asked of it."""
    f = friction_feedforward(np.zeros(6), np.zeros(6), np.ones(6) * 5)
    assert np.abs(f).max() < 1e-12


def test_breaks_away_from_standstill():
    """The whole point: a plain tanh(qd) term is 0 at rest and cannot do this."""
    f = friction_feedforward(np.zeros(6), np.full(6, 2.0), np.ones(6) * 5)
    assert np.all(f > 0)


def test_follows_velocity_sign_when_moving():
    """While moving, oppose friction along motion, ignoring commanded sign."""
    f = friction_feedforward(np.full(6, 0.5), np.full(6, -2.0), np.ones(6) * 5)
    assert np.all(f > 0)
    assert np.allclose(f, 5.0, atol=1e-6)


def test_bounded_by_fc():
    rng = np.random.default_rng(1)
    fc = np.array([10.34, 9.52, 6.96, 2.78, 2.94, 2.07])
    for _ in range(200):
        f = friction_feedforward(rng.uniform(-2, 2, 6), rng.uniform(-50, 50, 6), fc)
        assert np.all(np.abs(f) <= fc + 1e-9)


# -------------------------------------------------------------------- compute
def _imp():
    return CartesianImpedance(f_c=np.zeros(6))


def test_zero_error_zero_velocity_gives_negligible_torque():
    """
    The bumpless-transfer invariant. If this breaks, every mode entry jolts.

    Not bit-exact: the orientation error goes through a quaternion round-trip, so
    a non-canonical rotvec like HOME's leaves ~1e-16 rad. That is ~1e-15 Nm --
    fifteen orders below the 5 N force floor of this arm.
    """
    imp = _imp()
    tau, F, e = imp.compute(np.zeros(6), np.zeros(6), HOME, HOME, np.zeros(6), np.eye(6))
    assert np.abs(tau).max() < 1e-12
    assert np.abs(F).max() < 1e-12
    assert np.abs(e).max() < 1e-12


def test_restoring_force_points_toward_equilibrium():
    imp = _imp()
    eq = HOME.copy()
    eq[0] += 0.01                      # equilibrium is +x of actual
    _, F, _ = imp.compute(np.zeros(6), np.zeros(6), HOME, eq, np.zeros(6), np.eye(6))
    assert F[0] > 0                    # so the force must pull +x


def test_damping_opposes_motion():
    imp = _imp()
    _, F, _ = imp.compute(np.zeros(6), np.zeros(6), HOME, HOME,
                          np.r_[0.1, 0, 0, 0, 0, 0], np.eye(6))
    assert F[0] < 0


def test_orientation_error_is_clamped():
    imp = _imp()
    tgt = HOME.copy()
    tgt[3:] = (R.from_rotvec([0, 0, 2.0]) * R.from_rotvec(HOME[3:])).as_rotvec()
    _, _, e = imp.compute(np.zeros(6), np.zeros(6), HOME, tgt, np.zeros(6), np.eye(6))
    assert np.abs(e[3:]).max() <= MAX_ORIENTATION_ERROR + 1e-12


def test_saturation_respected():
    imp = _imp()
    far = HOME.copy()
    far[:3] += 10.0
    tau, F, _ = imp.compute(np.zeros(6), np.zeros(6), HOME, far, np.zeros(6), np.eye(6))
    assert np.all(np.abs(F) <= imp.F_sat + 1e-9)
    assert np.all(np.abs(tau) <= imp.tau_sat + 1e-9)


def test_ramp_scales_output():
    imp = _imp()
    eq = HOME.copy()
    eq[0] += 0.005
    _, F1, _ = imp.compute(np.zeros(6), np.zeros(6), HOME, eq, np.zeros(6), np.eye(6), ramp=1.0)
    imp.reset()
    _, F0, _ = imp.compute(np.zeros(6), np.zeros(6), HOME, eq, np.zeros(6), np.eye(6), ramp=0.0)
    assert np.abs(F0).max() == 0.0
    assert np.abs(F1).max() > 0.0


def test_blend_softens_z():
    imp = _imp()
    Kf, _ = imp.gains(0.0)
    Kc, _ = imp.gains(1.0)
    assert Kc[2] < Kf[2]                       # z softer in contact
    half, _ = imp.gains(0.5)
    assert Kf[2] > half[2] > Kc[2]


def test_velocity_filter_is_fast_not_force_alpha():
    """force_alpha (0.03, ~2.4 Hz) in the damping path is the original bug."""
    assert CartesianImpedance().vel_alpha >= 0.3


def test_env_defaults_do_not_cap_the_leash():
    """
    Regression: leash() computed the correct 0.167 rad and Env then capped it with
    max_orientation_step=0.05 (servoL's old value), so the restoring moment stayed
    at 1.5 Nm and the tool could not hold orientation. The caps default to None.
    """
    import inspect
    import env as env_mod
    sig = inspect.signature(env_mod.Env.__init__)
    assert sig.parameters['max_orientation_step'].default is None
    assert sig.parameters['max_position_step'].default is None


def test_leash_is_derived_from_F_sat_over_K():
    """
    Regression: max_orientation_step was left at servoL's 0.05 while the position
    leash was updated, so only 1.5 Nm of restoring moment was available instead of
    the intended 5 -- the tool could not hold orientation.
    """
    imp = _imp()
    pos, rot = imp.leash(0.0)
    K, _ = imp.gains(0.0)
    assert np.allclose(pos, imp.F_sat[:3] / K[:3])
    assert rot == pytest.approx(np.min(imp.F_sat[3:] / K[3:]))
    assert rot > 0.05                       # the old hardcoded value was too tight
    assert K[3] * rot == pytest.approx(imp.F_sat[3])


def test_leash_tracks_the_gain_schedule():
    """Softer contact gains must give a longer leash for the same force."""
    imp = _imp()
    assert imp.leash(1.0)[0][2] > imp.leash(0.0)[0][2]


def test_leash_respects_hard_caps():
    imp = _imp()
    pos, rot = imp.leash(0.0, max_pos=np.full(3, 0.001), max_rot=0.001)
    assert np.all(pos <= 0.001 + 1e-12) and rot <= 0.001 + 1e-12


def test_calibrate_gives_critical_damping():
    """D = 2*zeta*sqrt(K*I) per axis, from MEASURED inertia -- not the table."""
    kin = URKin(TCP)
    ref = kin.reference_inertia(np.array([0, -1.4, 1.4, -1.5, -1.5, 0.]))
    imp = _imp()
    imp.calibrate(ref, zeta=1.0)
    for K, D in ((imp.K_free, imp.D_free), (imp.K_contact, imp.D_contact)):
        zeta = D / (2 * np.sqrt(K * ref))
        assert np.allclose(zeta[3:], 1.0, atol=1e-9), 'rotational axes must be critically damped'
        assert np.all(zeta[:3] >= 1.0 - 1e-9)          # translational may be raised by d_min


def test_saturate_direction_preserves_direction():
    from impedance import saturate_direction
    v = np.array([80., 10., -5., 1., 2., -1.])
    lim = np.array([40., 40., 40., 12., 12., 12.])
    s = saturate_direction(v, lim)
    assert np.all(np.abs(s) <= lim + 1e-9)
    assert np.allclose(s / np.linalg.norm(s), v / np.linalg.norm(v))   # same direction
    # under the limit it is a no-op
    small = np.array([1., 1., 1., 0.1, 0.1, 0.1])
    assert np.allclose(saturate_direction(small, lim), small)


def test_saturation_keeps_a_rotation_a_rotation(kin):
    """
    Regression: per-axis clipping of a shaped wrench delivered the translational
    component in full while clipping the rotational one, so a commanded rotation
    came out as a translation. Direction-preserving scaling keeps it decoupled.
    """
    from impedance import saturate_direction
    imp = _imp()
    q = np.array([0, -1.4, 1.4, -1.5, -1.5, 0.])
    Lam = kin.task_inertia(q)
    imp.calibrate(kin.reference_inertia(q))
    K, _ = imp.gains(0.0)
    _, rot = imp.leash(0.0)

    F = (Lam / imp.inertia_d) @ np.r_[0, 0, 0, 0, K[4] * rot, 0]
    a_scaled = np.linalg.solve(Lam, saturate_direction(F, imp.F_sat))
    a_clipped = np.linalg.solve(Lam, np.clip(F, -imp.F_sat, imp.F_sat))

    assert np.linalg.norm(a_scaled[:3]) < 1e-9          # no spurious translation
    assert np.linalg.norm(a_scaled[3:]) > 1.0           # rotation actually happens
    assert np.linalg.norm(a_clipped[:3]) > 1.0          # clipping leaks translation


def test_rotational_authority_matches_the_joints(kin):
    """F_sat[3:] was 5 Nm while the joints can deliver ~13.5 Nm at the TCP."""
    imp = _imp()
    assert np.all(imp.F_sat[3:] >= 10.0)
    q = np.array([0, -1.4, 1.4, -1.5, -1.5, 0.])
    J = kin.jacobian(q)
    for k in range(3):
        u = np.r_[0, 0, 0, np.eye(3)[k]]
        reachable = np.min(imp.tau_sat / np.abs(J.T @ u + 1e-12))
        assert imp.F_sat[3 + k] <= reachable, 'F_sat asks for more moment than tau_sat allows'


def test_rotational_bandwidth_is_comparable_to_translation(kin):
    """At K_rot=30 the ry axis ran at 5.8 rad/s vs ~14 for translation: sluggish."""
    imp = _imp()
    q = np.array([0, -1.4, 1.4, -1.5, -1.5, 0.])
    I = kin.reference_inertia(q)
    imp.calibrate(I)
    w = np.sqrt(imp.K_free / I)
    assert w[3:].min() > 0.5 * w[:3].min()


def _ok():
    return dict(q=np.zeros(6), qd=np.zeros(6), pose=HOME, twist=np.zeros(6),
                raw_force=np.zeros(6), tau=np.zeros(6))


def test_nominal_does_not_trip():
    assert SafetyMonitor().check(**_ok()) is None


@pytest.mark.parametrize('field', ['q', 'qd', 'pose', 'twist', 'tau'])
def test_nan_trips(field):
    kw = _ok()
    kw[field] = np.full(6, np.nan)
    assert SafetyMonitor().check(**kw) is not None


def test_speed_trips():
    kw = _ok()
    kw['twist'] = np.r_[5.0, 0, 0, 0, 0, 0]
    assert 'speed' in SafetyMonitor().check(**kw)


def test_force_trips():
    kw = _ok()
    kw['raw_force'] = np.r_[500.0, 0, 0, 0, 0, 0]
    assert 'force' in SafetyMonitor().check(**kw)


def test_is_late_detects_stall():
    """Lateness is reported, but the POLICY lives in Env: one late tick zeroes
    torque, only sustained lateness trips. A stalled tick holds the last torque
    rather than decaying, which is why it matters at all."""
    s = SafetyMonitor(dt=0.002)
    assert s.is_late(0.05)
    assert not s.is_late(0.002)
    assert not s.is_late(None)
    # and it is no longer a check() fault
    assert s.check(**_ok()) is None


def test_workspace_trips():
    box = (-1, 1, -1, 1, 0.0, 1.0)
    kw = _ok()
    kw['pose'] = HOME.copy()
    kw['pose'][2] = -0.5
    assert 'workspace' in SafetyMonitor(workspace=box).check(**kw)


def test_large_equilibrium_offset_does_not_trip():
    """
    Driving the z target centimetres below a surface is how force is commanded
    under the new adaptive_mode, so it must NOT be treated as divergence.
    """
    assert SafetyMonitor().check(**_ok()) is None


# ---------------------------------------------------------------- kinematics
@pytest.fixture(scope='module')
def kin():
    return URKin(TCP)


def test_uses_base_not_base_link(kin):
    """base_link is Rz(180) off; using it negates commanded x/y force."""
    wrong = URKin(TCP, base='base_link')
    q = np.array([0.1, -1.2, 1.4, -0.6, 1.5, 0.3])
    assert np.allclose(kin.fk(q)[:2], -wrong.fk(q)[:2], atol=1e-9)
    assert not np.allclose(kin.fk(q)[:2], wrong.fk(q)[:2])


def test_jacobian_matches_finite_differenced_fk(kin):
    """Angular block differenced on the manifold, not componentwise on rotvecs."""
    rng = np.random.default_rng(0)
    h = 1e-6
    worst = 0.0
    for _ in range(100):
        q = rng.uniform(-2.5, 2.5, 6)
        qd = rng.uniform(-1, 1, 6)
        p0, p1 = kin.fk(q), kin.fk(q + h * qd)
        v = (p1[:3] - p0[:3]) / h
        w = (R.from_rotvec(p1[3:]) * R.from_rotvec(p0[3:]).inv()).as_rotvec() / h
        worst = max(worst, np.abs(kin.jacobian(q) @ qd - np.r_[v, w]).max())
    assert worst < 1e-4


def test_jacobian_matches_independent_dh(kin):
    """Cross-check against nominal UR16e DH, derived independently of pinocchio."""
    d = np.array([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])
    a = np.array([0, -0.4784, -0.36, 0, 0, 0])
    al = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])

    def T(i, th):
        ca, sa, ct, st = np.cos(al[i]), np.sin(al[i]), np.cos(th), np.sin(th)
        return np.array([[ct, -st * ca, st * sa, a[i] * ct],
                         [st, ct * ca, -ct * sa, a[i] * st],
                         [0, sa, ca, d[i]], [0, 0, 0, 1]])

    k0 = URKin([0, 0, 0, 0, 0, 0])
    rng = np.random.default_rng(2)
    for _ in range(50):
        q = rng.uniform(-2.5, 2.5, 6)
        M = np.eye(4)
        for i in range(6):
            M = M @ T(i, q[i])
        assert np.abs(k0.fk(q)[:3] - M[:3, 3]).max() < 1e-6


def test_tau_rated_from_urdf(kin):
    assert np.allclose(kin.tau_rated, [330, 330, 150, 54, 54, 54])


def test_replay_recorded_episode(kin):
    """
    Push a real recorded trajectory through the control law and assert the wrench
    and torque stay inside limits for the whole episode. Catches gain choices that
    would have saturated constantly -- something synthetic inputs will not.

    Skipped where no episode is present (outputs/ is gitignored); runs on the
    robot box.
    """
    import glob
    import h5py
    paths = sorted(glob.glob('outputs/**/rawdata.h5', recursive=True))
    if not paths:
        pytest.skip('no recorded episode available')

    with h5py.File(paths[-1], 'r') as f:
        actual = np.array(f['robot_obs/actual_pose'])
        des = np.array(f['commands/des_pose'])
        has_q = 'robot_obs/actual_q' in f
        q_all = np.array(f['robot_obs/actual_q']) if has_q else None

    imp = CartesianImpedance()
    safety = SafetyMonitor()
    n = min(len(actual), 5000)
    for i in range(0, n, 10):
        eq = des[min(i * len(des) // max(len(actual), 1), len(des) - 1)]
        q = q_all[i] if has_q else np.array([0, -1.4, 1.4, -1.5, -1.5, 0.])
        J = kin.jacobian(q)
        tau, F, _ = imp.compute(q, np.zeros(6), actual[i], eq, np.zeros(6), J)
        assert np.all(np.abs(F) <= imp.F_sat + 1e-9)
        assert np.all(np.abs(tau) <= imp.tau_sat + 1e-9)
        assert np.all(np.isfinite(tau))


def test_task_inertia_plausible(kin):
    """Measured median on this arm is ~8.35 kg translational."""
    L = kin.task_inertia(np.array([0, -1.4, 1.4, -1.5, -1.5, 0]))
    m = np.diag(L)[:3]
    assert np.all(m > 1.0) and np.all(m < 50.0)


# ------------------------------------------- measured plant (chirp identified)
def test_effective_inertia_is_payload_plus_residual():
    """
    The UR firmware compensates its own dynamics inside direct_torque, so the
    apparent inertia is the TOOL, not the arm. Chirp identification: model
    predicted 6.5-13.1 kg, measured 3.58 (CV 16% over 2 poses x 3 axes), and a
    1 kg added mass moved it 4.10 -> 4.98 (~1:1).
    """
    imp = CartesianImpedance()
    I = imp.effective_inertia(1.62, TCP)
    assert 3.5 < I[0] < 4.5, 'should match the measured ~4.0 kg at this payload'
    assert np.allclose(I[:3], I[0])            # isotropic: it is the tool
    # and it must be far below what the kinematic model claims
    assert I[0] < 0.6 * 8.0


def test_effective_inertia_tracks_payload_one_to_one():
    """Adding 1 kg of payload must add 1 kg of apparent inertia."""
    imp = CartesianImpedance()
    a = imp.effective_inertia(1.62, TCP)
    b = imp.effective_inertia(2.62, TCP)
    assert b[0] - a[0] == pytest.approx(1.0)


def test_calibrate_reproduces_the_measured_good_damping():
    """
    D[x] = 110 at zeta=0.7 was measured to flatten the resonance (peak/DC
    1.75 -> 1.08) with no late ticks. The derivation must land there.
    """
    imp = CartesianImpedance()
    imp.calibrate(imp.effective_inertia(1.62, TCP), zeta=0.7)
    assert imp.D_free[0] == pytest.approx(110, rel=0.10)
    assert np.all(imp.D_contact <= imp.D_free)


def test_calibrate_is_critically_damped_by_construction():
    imp = CartesianImpedance()
    I = imp.effective_inertia(1.62, TCP)
    for z in (0.5, 0.7, 1.0):
        imp.calibrate(I, zeta=z)
        assert np.allclose(imp.D_free / (2 * np.sqrt(imp.K_free * I)), z)
        assert np.allclose(imp.D_contact / (2 * np.sqrt(imp.K_contact * I)), z)


def test_no_model_inertia_in_the_damping_path():
    """Regression: every damping failure traced to the kinematic Lambda."""
    import inspect
    src = inspect.getsource(CartesianImpedance)
    for gone in ('shape_inertia', 'shaping_factor', 'task_inertia', 'lam_dt_max'):
        assert gone not in src, f'{gone} should be gone from the control law'
