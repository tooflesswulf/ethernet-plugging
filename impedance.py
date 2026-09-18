"""
6-DOF Cartesian impedance control law and its safety monitor.

Pure computation: nothing here touches RTDE. `Env` cannot be imported without a
robot (env.py:108 opens sockets in __init__), so keeping the math here is what
makes any of it testable. See tests/test_impedance.py.
"""
from scipy.spatial.transform import Rotation as R
import numpy as np

# Control modes
IDLE = 'idle'
IMPEDANCE = 'impedance'
FAULT = 'fault'

MAX_ORIENTATION_ERROR = 0.2   # rad; the rotvec axis is ill-conditioned near pi


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


def friction_feedforward(qd, tau_cmd, f_c, v_eps=0.05, t_eps=4.0, assist=0.5):
    """
    Per-joint Coulomb friction compensation. Not optional on this arm: it is the
    difference between a 20 N and a 5 N contact-force floor.

    A pure tanh(qd) Coulomb term is identically zero at rest, so it cannot break
    away from stiction -- the case that matters. Blend instead: velocity direction
    while moving, commanded-torque direction while stationary.

    f_c must be PER JOINT and MEASURED (see test-impedance.py `identify`). A single
    scalar scaled by rated torque over-compensates some joints into a limit cycle
    while leaving others in deadband. Use ~80% of the WEAKER direction, never the
    mean: on this arm joint 1 measured 11.90/21.87 Nm, and the mean would
    over-compensate the weak direction.

    v_eps sets how sharply the term flips with velocity sign. Too small and it is
    effectively f_c*sign(qd), so every direction reversal swings the torque by
    2*f_c (up to 20 Nm here) -- which chatters. `assist` (0..1) weights only the
    standstill term; lowering it trades breakaway crispness for calm.
    """
    s_v = np.tanh(np.asarray(qd, float) / v_eps)
    s_t = np.tanh(np.asarray(tau_cmd, float) / t_eps)
    return np.asarray(f_c, float) * (s_v + assist * (1.0 - np.abs(s_v)) * s_t)


class CartesianImpedance:
    """
    F = K*e - D*xd ; tau = J^T F, plus joint damping and friction feedforward.

    Gains are plain mutable attributes so a live tuner can setattr them at runtime.
    ALWAYS assign a whole new array (self.K_free = np.array([...])); never mutate
    in place, or the 500 Hz loop can read a half-written gain vector.
    """

    def __init__(self, f_c=None, tau_rated=None):
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

        # Measured on this arm at 80% of the weaker direction.
        self.f_c = (np.asarray(f_c, float) if f_c is not None
                    else np.array([10.34, 9.52, 6.96, 2.78, 2.94, 2.07]))
        self.fc_veps = 0.05
        self.fc_teps = 4.0
        self.fc_assist = 0.5

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

    def compute(self, q, qd, actual_pose, eq_pose, xd, J, blend=0.0, ramp=1.0):
        """
        Returns (tau, F, e). Takes everything as arguments -- no RTDE access --
        which is what makes this testable without hardware.

        `ramp` scales K and D for bumpless entry. With e = 0 and xd = 0 the
        returned torque is zero to ~1e-15 Nm (the orientation error round-trips
        through a quaternion, so it is not bit-exact); that invariant is a unit
        test, because if it breaks every mode entry jolts.
        """
        qd = np.asarray(qd, float)
        K, D = self.gains(blend)

        e = pose_error(actual_pose, eq_pose)
        e[3:] = np.clip(e[3:], -MAX_ORIENTATION_ERROR, MAX_ORIENTATION_ERROR)

        xd_f = self.filter_velocity(xd)
        F = ramp * (K * e - D * xd_f)

        F = saturate_direction(F, self.F_sat)

        tau = J.T @ F - self.d_q * qd
        if np.any(self.f_c > 0):
            tau = tau + ramp * friction_feedforward(
                qd, tau, self.f_c, self.fc_veps, self.fc_teps, self.fc_assist)
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
