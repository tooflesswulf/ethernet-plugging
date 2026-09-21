"""
6-DOF Cartesian impedance control law and its safety monitor.

Pure computation: nothing here touches RTDE. `Env` cannot be imported without a
robot (env.py:108 opens sockets in __init__), so keeping the math here is what
makes any of it testable. See tests/test_impedance.py.
"""
from pathlib import Path
import tomllib
import warnings

from scipy.spatial.transform import Rotation as R
import numpy as np

# Control modes
IDLE = 'idle'
IMPEDANCE = 'impedance'
FAULT = 'fault'

MAX_ORIENTATION_ERROR = 0.2   # rad; the rotvec axis is ill-conditioned near pi

# Friction constants live in friction.toml, NOT here -- edit that file. A
# missing or malformed config falls back to NO compensation, which is the safe
# direction and is also what env.py has been running: a wrong f_c that
# over-compensates drives the joint rather than freeing it, so silence is
# preferable to a guess.
FRICTION_CONFIG = Path(__file__).with_name('friction.toml')

_FALLBACK = {'f_c_pos': [0.0] * 6, 'f_c_neg': [0.0] * 6,
             'f_k_pos': [0.0] * 6, 'f_k_neg': [0.0] * 6,
             'fc_assist': 0.5, 'fc_veps': 0.05, 'fc_teps': 4.0}


def load_gravity_residual(path=FRICTION_CONFIG):
    """
    Read [gravity_residual].theta -> (4,) mass-moment parameters.

    Zeros on any problem, which makes the correction a no-op rather than a
    guess. See kinematics.URKin.gravity_bias for what theta means.
    """
    try:
        with open(path, 'rb') as fh:
            th = np.asarray(tomllib.load(fh)['gravity_residual']['theta'], float)
        if th.shape != (4,) or not np.all(np.isfinite(th)):
            raise ValueError('theta must be 4 finite numbers')
        return th
    except Exception as e:
        warnings.warn(f'{path}: {e!r} -- gravity residual correction disabled',
                      RuntimeWarning)
        return np.zeros(4)


def load_scales(path=FRICTION_CONFIG):
    """
    Read the [scales] table from friction.toml -> (viscous, coulomb).

    These are the FIRMWARE compensation scales, not our feedforward. Kept here
    so env.py, brr.py and the measurement scripts share one copy: the repo
    previously carried two disagreeing coulomb tables and nothing said which was
    authoritative.

    Falls back to all zeros -- no firmware compensation -- on any problem, for
    the same reason load_friction does: a scale that is too high drives the
    joint rather than freeing it.
    """
    try:
        with open(path, 'rb') as fh:
            sc = tomllib.load(fh)['scales']
        out = []
        for k in ('viscous', 'coulomb'):
            v = np.asarray(sc[k], float)
            if v.shape != (6,) or not np.all(np.isfinite(v)) or np.any(v < 0):
                raise ValueError(f'{k} must be 6 finite non-negative numbers')
            out.append(v)
        return out[0], out[1]
    except Exception as e:
        warnings.warn(f'{path}: {e!r} -- falling back to ZERO firmware '
                      f'compensation scales', RuntimeWarning)
        return np.zeros(6), np.zeros(6)


def load_friction(path=FRICTION_CONFIG):
    """
    Read friction.toml. Returns a dict with the keys in _FALLBACK.

    Never raises: the 500 Hz loop must be able to start. A bad config degrades
    to zero compensation and says so, rather than taking the arm down or, worse,
    running with half a table.
    """
    try:
        with open(path, 'rb') as fh:
            cfg = tomllib.load(fh)['friction']
        out = dict(_FALLBACK)
        for k in out:
            if k in cfg:
                out[k] = cfg[k]
        for k in ('f_c_pos', 'f_c_neg', 'f_k_pos', 'f_k_neg'):
            if len(out[k]) != 6:
                raise ValueError(f'{k} must have 6 entries, got {len(out[k])}')
            out[k] = np.asarray(out[k], float)
            if np.any(out[k] < 0) or not np.all(np.isfinite(out[k])):
                raise ValueError(f'{k} must be finite and non-negative')
        if not 0.0 <= float(out['fc_assist']) <= 1.0:
            raise ValueError('fc_assist must be in [0, 1]')
        return out
    except Exception as e:
        warnings.warn(f'{path}: {e!r} -- falling back to NO friction '
                      f'compensation', RuntimeWarning)
        out = dict(_FALLBACK)
        for k in ('f_c_pos', 'f_c_neg', 'f_k_pos', 'f_k_neg'):
            out[k] = np.zeros(6)
        return out


def pose_error(actual, desired):
    """
    6-vector [dp; drotvec] in the base frame, pointing from actual to desired.

    Orientation MUST be R_des @ R_act.T (left-multiplied) so the error rotvec is
    in the base frame and pairs with a base-frame Jacobian.

    Never subtract rotvec components. home_pose's (2.44, 2.44, 0.653) has norm
    3.512 > pi -- a valid non-canonical rotvec -- so componentwise subtraction
    against a canonicalised pose gives a ~2pi error and full commanded torque.
    """
    actual = np.asarray(actual, float)
    desired = np.asarray(desired, float)
    R_act = R.from_rotvec(actual[3:])
    R_des = R.from_rotvec(desired[3:])
    return np.r_[desired[:3] - actual[:3], (R_des * R_act.inv()).as_rotvec()]


def saturate_direction(v, limit):
    """
    Scale v down uniformly until it fits inside `limit`, preserving direction.

    NOT per-axis clipping. Clipping one component of a shaped wrench breaks the
    Lambda*Lambda_d^-1 combination that makes it decoupled: a pure rotation
    command clipped only in its rotational component still delivers its full
    translational component, so the arm translates instead of rotating.
    """
    v = np.asarray(v, float)
    limit = np.asarray(limit, float)
    over = np.abs(v) / np.maximum(limit, 1e-12)
    m = float(np.max(over))
    return v / m if m > 1.0 else v


def friction_feedforward(qd, tau_cmd, f_c, v_eps=0.05, t_eps=4.0, assist=0.5,
                         f_c_neg=None, f_k=None, f_k_neg=None):
    """
    Per-joint friction compensation: a STATIC term and a KINETIC term, each with
    its own magnitude and its own direction.

    Not optional on this arm: it is the difference between a 20 N and a 5 N
    contact-force floor.

        static   f_s * assist * (1 - |s_v|) * s_t      direction from s_t
        kinetic  f_k * s_v                             direction from s_v

    A pure tanh(qd) Coulomb term is identically zero at rest, so it cannot break
    away from stiction -- the case that matters. The static term is keyed to
    COMMANDED TORQUE instead, which is the one signal that is non-zero when a
    joint is stuck. `(1 - |s_v|)` hands over to the kinetic term as the joint
    starts moving.

    TWO MAGNITUDES, because they are two quantities. Static breakaway is what
    test-breakaway.py measures; kinetic friction is what remains while moving,
    and it is LOWER. On joint 1 the coast data in test-scale-sweep.py puts
    kinetic near 5.3 Nm (3.6-7.0 across directions, +/-50% -- the travel behind
    it is one significant figure) against a static minimum of 6.83 positive and
    11.03 negative. Driving the moving term with the static number therefore
    over-compensates by roughly 1.3x one way and 2.1x the other, and
    over-compensation while moving is a RUNAWAY, not a limit cycle: the term
    pushes along the velocity the joint already has, so an over-estimate drives
    it (test-scale-sweep.py measured exactly that at coulomb 0.8).

    `f_k` defaults to `f_c` when omitted, which reproduces the old single-
    magnitude behaviour exactly. Set it to ZERO to leave kinetic friction to the
    firmware, whose own coulomb compensation IS velocity-based and delivers
    about 6.0 Nm per unit scale on joint 1 -- the right tool for that half.

    DIRECTIONS ARE TAKEN PER TERM. `f_c`/`f_k` are the positive-direction
    values, `f_c_neg`/`f_k_neg` the negative ones. The static term's direction
    comes from the commanded torque and the kinetic term's from the velocity;
    these DISAGREE during a reversal, which is precisely when the output is
    largest, so they are selected separately rather than from one lumped sign.

    `f_c` must be MEASURED per joint (test-breakaway.py), and `f_k` likewise
    from the coast deceleration (test-scale-sweep.py). `assist` (0..1) weights
    ONLY the static term, so the delivered push at rest is assist * f_c.

    v_eps sets how sharply the kinetic term flips with velocity sign. Too small
    and it is effectively f_k*sign(qd), so every reversal swings the torque by
    2*f_k -- which chatters.
    """
    s_v = np.tanh(np.asarray(qd, float) / v_eps)
    s_t = np.tanh(np.asarray(tau_cmd, float) / t_eps)

    f_s_pos = np.asarray(f_c, float)
    f_s_neg = f_s_pos if f_c_neg is None else np.asarray(f_c_neg, float)
    f_k_pos = f_s_pos if f_k is None else np.asarray(f_k, float)
    if f_k_neg is not None:
        f_k_neg_ = np.asarray(f_k_neg, float)
    elif f_k is None:
        f_k_neg_ = f_s_neg          # old behaviour: kinetic inherits static
    else:
        f_k_neg_ = f_k_pos          # explicit symmetric kinetic

    static = np.where(s_t >= 0.0, f_s_pos, f_s_neg) \
        * assist * (1.0 - np.abs(s_v)) * s_t
    kinetic = np.where(s_v >= 0.0, f_k_pos, f_k_neg_) * s_v
    return static + kinetic


class CartesianImpedance:
    """
    F = K*e - D*xd ; tau = J^T F, plus joint damping and friction feedforward.

    Gains are plain mutable attributes so a live tuner can setattr them at runtime.
    ALWAYS assign a whole new array (self.K_free = np.array([...])); never mutate
    in place, or the 500 Hz loop can read a half-written gain vector.
    """

    def __init__(self, f_c=None, tau_rated=None, f_c_neg=None):
        # K and D are both EXPLICIT. D used to be derived in calibrate() from an
        # apparent-inertia model (payload + a measured residual); that model was
        # identified while joint friction was uncompensated, so it described a
        # plant that no longer exists now that the firmware does the friction.
        # These D are the values that model produced at zeta = 0.7 and that the
        # chirp campaign actually validated -- kept as measurements, without the
        # model that used to regenerate them. Re-measure by chirp before trusting
        # them under the new plant (RECALIBRATION-PLAN.md step 3).
        #
        # K_rot = 200 is measured, not guessed. The TCP sits 12.1 cm from the
        # payload's centre of mass, so ANY force at the TCP torques the tool --
        # a 15 N x command makes 1.85 Nm about y. K_rot only decides how far it
        # tilts before the spring balances. Chirp at K_rot 50 -> 200 (zeta held
        # at 0.70) cut off-axis ry from 15.07 to 3.77 mrad, a 4x improvement
        # matching the predicted M/K scaling, and nearly halved peak force
        # (13.5 -> 7.3 N). rx and z improved too; x was unchanged.
        self.K_free = np.array([1500., 1500., 1500., 200., 200., 200.])
        self.D_free = np.array([109.39, 109.39, 109.39, 10.60, 10.60, 10.60])
        self.K_contact = np.array([800., 800., 400., 160., 160., 200.])
        self.D_contact = np.array([79.89, 79.89, 56.49, 9.48, 9.48, 10.60])

        # Rotational limit sized to what the joints can actually deliver at the
        # TCP (13.5 Nm here). The old 5 Nm clipped every shaped rotation.
        self.F_sat = np.array([40., 40., 40., 12., 12., 12.])
        self.tau_sat = 0.25 * (np.asarray(tau_rated, float)
                               if tau_rated is not None
                               else np.array([330., 330., 150., 54., 54., 54.]))

        # Friction constants come from friction.toml -- see load_friction().
        # f_c is the POSITIVE-direction breakaway, f_c_neg the negative one, at
        # 100% of the minimum measured across five poses. The 80% the doctrine
        # asks for is delivered by fc_assist, not baked into f_c.
        #
        # Passing f_c alone still gives symmetric behaviour: f_c_neg only comes
        # from the config alongside f_c, so an explicit f_c with no f_c_neg
        # keeps the single-magnitude form a caller asked for.
        cfg = load_friction()
        if f_c is None:
            self.f_c = np.asarray(cfg['f_c_pos'], float).copy()
            self.f_c_neg = (np.asarray(cfg['f_c_neg'], float).copy()
                            if f_c_neg is None else np.asarray(f_c_neg, float))
        else:
            self.f_c = np.asarray(f_c, float)
            self.f_c_neg = None if f_c_neg is None else np.asarray(f_c_neg, float)
        # KINETIC magnitudes, separate from the static ones above. Zero by
        # default: the firmware's velocity-based coulomb compensation is the
        # right tool for this half, and an f_k that is too large drives the
        # joint. Note these are passed EXPLICITLY below, so zero really means
        # no kinetic term -- unlike friction_feedforward's own f_k=None
        # default, which inherits f_c for direct callers' backward compatibility.
        self.f_k = np.asarray(cfg['f_k_pos'], float).copy()
        self.f_k_neg = np.asarray(cfg['f_k_neg'], float).copy()
        self.fc_veps = float(cfg['fc_veps'])
        self.fc_teps = float(cfg['fc_teps'])
        # Weights the STANDSTILL term only, so the delivered push at rest is
        # fc_assist * f_c -- 0.8 * the weakest measured breakaway by default.
        #
        # The MOVING term is not weighted by it. At speed the output is the full
        # f_c, i.e. 100% of the STATIC breakaway applied against KINETIC
        # friction, which is lower -- joint 1's was inferred at ~2.8 Nm from the
        # coast data in test-scale-sweep.py against a 6.83-11.03 Nm static
        # minimum. If the arm runs on after a move instead of stopping, that is
        # this term, and the fix is a separate kinetic scale, not a lower
        # assist.
        self.fc_assist = float(cfg['fc_assist'])

        self.d_q = 0.5          # joint damping floor, unconditionally passive
        self.vel_alpha = 0.4    # ~40 Hz. NOT force_alpha (0.03, ~2.4 Hz):
                                # phase lag in the damping term is the original bug.
        self._xd_f = np.zeros(6)

        # Inertia shaping was removed. It tried to decouple Lambda by commanding
        # F = Lambda*Lambda_d^-1*u, which needs near-exact cancellation (266 m/s^2
        # of individual contributions summing to 0 for a pure rotation) against a
        # 5 N friction floor -- impossible here, and it turned rotation commands
        # into large translations. It was also modelling a plant the UR firmware
        # already compensates. See git history if it ever needs revisiting.

    def calibrate(self, inertia, zeta=1.0):
        """
        Derive D = 2 zeta sqrt(K I) per axis from an EXPLICIT inertia.

        No longer called in the control path -- D_free/D_contact are set directly
        above. This stays as a tool for chirp experiments that want to sweep a
        damping hypothesis (test-impedance2.py --inertia). Whatever you pass must
        be a number you measured; there is no apparent-inertia model here any
        more to hand you one.

        There is no discrete-stability bound here either: the one that used to
        live here scored the measured-stable configuration at lambda*dt = 13.6
        and the measured-diverging one at 24.9 -- overlapping, so it never
        discriminated, and it was computed from the wrong plant besides. Verify
        damping with a chirp sweep (test-impedance2.py), not a model.
        """
        I = np.asarray(inertia, float)
        self.inertia_d = I.copy()
        self.D_free = 2.0 * zeta * np.sqrt(self.K_free * I)
        self.D_contact = 2.0 * zeta * np.sqrt(self.K_contact * I)
        return I

    def reset(self):
        self._xd_f = np.zeros(6)

    def gains(self, blend):
        """Interpolate free <-> contact. blend in [0, 1]."""
        b = float(np.clip(blend, 0.0, 1.0))
        return (self.K_free + b * (self.K_contact - self.K_free),
                self.D_free + b * (self.D_contact - self.D_free))

    def filter_velocity(self, xd):
        self._xd_f = self.vel_alpha * np.asarray(xd, float) \
            + (1 - self.vel_alpha) * self._xd_f
        return self._xd_f

    def leash(self, blend=0.0, max_pos=None, max_rot=None):
        """
        Equilibrium-error leash, in the units clamp() wants.

        Derived from F_sat / K so it stays consistent when gains change --
        hardcoding it is how the orientation leash ended up 3.3x too tight while
        the position one was correct, leaving only 1.5 Nm of restoring moment.
        `max_pos`/`max_rot` are optional hard caps.
        """
        K, _ = self.gains(blend)
        pos = self.F_sat[:3] / np.maximum(K[:3], 1e-9)
        rot = float(np.min(self.F_sat[3:] / np.maximum(K[3:], 1e-9)))
        if max_pos is not None:
            pos = np.minimum(pos, max_pos)
        if max_rot is not None:
            rot = min(rot, max_rot)
        return pos, rot

    def compute(self, q, qd, actual_pose, eq_pose, xd, J, blend=0.0, ramp=1.0,
                tau_bias=None):
        """
        Returns (tau, F, e). Takes everything as arguments -- no RTDE access --
        which is what makes this testable without hardware.

        `ramp` scales K and D for bumpless entry. With e = 0 and xd = 0 the
        returned torque is zero to ~1e-15 Nm (the orientation error round-trips
        through a quaternion, so it is not bit-exact); that invariant is a unit
        test, because if it breaks every mode entry jolts.

        `tau_bias` is a per-joint standing torque to cancel -- the gravity-model
        residual, NOT friction. It is direction-independent, so it cannot be
        folded into f_c: doing that is what made joint 1 look like it had a 2.85x
        friction asymmetry. Measured over five poses, joint 1's standing torque
        ranges 1.05-7.13 Nm and tracks the gravity-error model (R^2 0.76) while
        joints 2 and 3 are flat, so it is pose dependent and has to be computed
        per tick by the caller -- impedance.py has no kinematics and must not
        grow any. It is scaled by `ramp` with everything else so mode entry
        stays bumpless, and it goes in BEFORE the friction term so that term
        sees the torque actually being asked of the joint.

        Nothing supplies tau_bias yet. The 4-parameter fit that would (dm =
        -1.031 kg at [0.202, 0.168, 0.111] kg m) sits at R^2 0.68 over all
        fifteen measurements, which is not good enough to put in the control
        path. The hook exists so the correction has somewhere to go once the
        residual is understood; see RECALIBRATION-PLAN.md.
        """
        qd = np.asarray(qd, float)
        K, D = self.gains(blend)

        e = pose_error(actual_pose, eq_pose)
        e[3:] = np.clip(e[3:], -MAX_ORIENTATION_ERROR, MAX_ORIENTATION_ERROR)

        xd_f = self.filter_velocity(xd)
        F = ramp * (K * e - D * xd_f)

        F = saturate_direction(F, self.F_sat)

        tau = J.T @ F - self.d_q * qd
        if tau_bias is not None:
            tau = tau + ramp * np.asarray(tau_bias, float)
        if np.any(self.f_c > 0):
            tau = tau + ramp * friction_feedforward(
                qd, tau, self.f_c, self.fc_veps, self.fc_teps, self.fc_assist,
                self.f_c_neg, self.f_k, self.f_k_neg)
        tau = saturate_direction(tau, self.tau_sat)
        return tau, F, e


class SafetyMonitor:
    """
    Returns a reason string on trip, else None.

    Note what is NOT here: a relative pose-error trip. Driving the z equilibrium
    centimetres below a surface is how force is commanded under the new
    adaptive_mode, so ||p_eq - p_act|| is no longer an anomaly signal. The error
    leash (util.clamp, sized F_sat/K) bounds the wrench instead, and the absolute
    workspace box below is the real runaway guard.
    """

    def __init__(self, workspace=None, dt=0.002):
        # (xmin, xmax, ymin, ymax, zmin, zmax) in base frame.
        self.workspace = workspace
        self.v_max = 0.5            # m/s
        self.w_max = 2.0            # rad/s
        self.qd_max = 2.0           # rad/s
        self.force_max = 60.0       # N, RAW force -- the 2.4 Hz filtered channel
                                    # is far too slow to be a safety signal
        self.late_factor = 3.0
        self.dt = dt

    def is_late(self, dt_actual):
        """
        A late tick is not fatal by itself -- the caller zeroes torque for that
        tick, which is safe (gravity-compensated float). It is only fatal if it
        keeps happening, because a tick that sends nothing leaves the controller
        re-applying the previous torque. Policy lives in the caller.
        """
        return dt_actual is not None and dt_actual > self.late_factor * self.dt

    def check(self, q, qd, pose, twist, raw_force, tau, dt_actual=None):
        for name, v in (('q', q), ('qd', qd), ('pose', pose),
                        ('twist', twist), ('tau', tau)):
            if not np.all(np.isfinite(v)):
                return f'non-finite {name}'

        if np.max(np.abs(qd)) > self.qd_max:
            return f'joint speed {np.max(np.abs(qd)):.2f} > {self.qd_max} rad/s'
        if np.linalg.norm(twist[:3]) > self.v_max:
            return f'TCP speed {np.linalg.norm(twist[:3]):.2f} > {self.v_max} m/s'
        if np.linalg.norm(twist[3:]) > self.w_max:
            return f'TCP angular speed {np.linalg.norm(twist[3:]):.2f} > {self.w_max} rad/s'
        if np.max(np.abs(raw_force[:3])) > self.force_max:
            return f'external force {np.max(np.abs(raw_force[:3])):.1f} > {self.force_max} N'

        if self.workspace is not None:
            lo = np.array(self.workspace[0::2], float)
            hi = np.array(self.workspace[1::2], float)
            if np.any(pose[:3] < lo) or np.any(pose[:3] > hi):
                return f'TCP {np.round(pose[:3], 3)} outside workspace'

        return None
