"""
FDCC-style Cartesian admittance through speedL, with the compliance centre
wherever we put it.

    M a + D (v - v_t) + K (x - x_t) = F_ext          at the compliance point p

is integrated to the twist of p, moved to the TCP and sent with speedL at 500 Hz.
UR's own joint velocity loops sit underneath, so joint stiction (12-22 Nm
breakaway on joint 1; a 15-40 N deadband at the TCP under direct torque) is
theirs to absorb, not ours.

Why not forceMode: on this arm it is an admittance of ~0.93 mm/s per N with
~90 ms lag, but centred on the FLANGE, and task_frame does not move the centre
(test-forcemode-center.py). The ftRtdeInputEnable path that would have moved it
protective-stops the robot on the enable call. Here all the geometry is ours: the
wrench is re-referenced to p, the dynamics run at p, and the twist is moved back
to the TCP for speedL.

Modes (--mode)
    teleop    DualSense drives the target; compliance is rendered around it (default)
    center    compliance-centre check, spin-ratio method from test-forcemode-center.py
    free      free-space response: mm/s per N and lag, vs forceMode 0.93 mm/s/N, 90 ms
    tap       approach a hard surface at several speeds: peak force and ringing
    stall     what the arm does when the loop stops sending (see Safety)
    selftest  offline, no robot: the same code against a kinematic fake. Run it first.
  --replay LOG.npz   re-run a saved log's analysis without the robot

Conventions
    Vectors are base-frame unless named otherwise. Angles in radians.
    pose / target are TCP poses [x y z rx ry rz]; p = TCP + R_tcp @ --point.
    x - x_t = [p - p_t, rotvec(R R_t^T)]. K, D, M and --sel act on the axes of the
    compliance frame: --frame tool = the TCP axes (moving with it), base = fixed.
    v_t is the target's own velocity, so a moving target is tracked without the
    D/K lag a plain spring would have.

Wrench reference point (--ft-ref)
    getActualTCPForce is a base-frame wrench (F, tau_q) with its moment about SOME
    point q. UR's docs say the TCP; `--mode center` on this arm says the FLANGE
    (2026-09-22), so that is the default. Re-referencing to p:

        F_p = F        tau_p = tau_q + (q - p) x F

    If the assumed q is off by delta, a force pushed through p leaves
    tau_p = delta x F, and the arm rotates about a centre shifted by delta --
    154 mm, the flange-to-TCP distance. `--mode center` measures it: the centre
    lands on --point when --ft-ref is right, and the report prints the implied q
    when it is not.

    ALSO: anything that blocks while the watchdog is armed trips it (C207A0). Slow
    setup -- the DualSense / interface.py import -- happens before it is armed.

Safety
    Speed and acceleration clamps (--vmax, --amax), force clamp (--fmax) and abort
    (--f-abort), a leash on target vs actual (--leash), a drift abort
    (--max-drift), abort on protective / e-stop, and speedStop() then stopScript()
    in a finally.

    speedL's time argument does NOT stop a stalled loop with ur_rtde 1.6.5. The
    control script's speed_thread re-issues speedl(target, a, t) in a loop with
    the last target until speedStop, so a stalled host keeps the last velocity.
    The argument must be ONE control cycle: speed_thread only picks up a new
    target when speedl() returns, so t = 8 ms fed the arm an 8 ms staircase and
    it buzzed at 125 Hz whenever the speed changed (fdcc-ramp-20260922-173332).
    The stall protection is the RTDE watchdog
    (--watchdog-hz): the controller stops the program if no input arrives for
    1/hz. Every speedL feeds it and every idle wait and prompt kicks it. Nothing
    here blocks without kicking, which is why the return between pushes is our own
    speedL position loop and not moveL. `--mode stall` checks both on the arm.

Controls (teleop)
    sticks / L2 R2 (+L1 R1 flips)   move target (same as teleoperation.py)
    Dpad Up / Down                  stiffness x1.5 / /1.5 (damping by sqrt: same damping ratio)
    Square                          toggle rotation lock (holds the current orientation)
    Cross                           re-anchor target to the actual pose
    Dpad Left                       stop and zero the F/T -- ONLY out of contact
    Ctrl-C                          stop (speedStop, then stopScript)
"""
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import numpy as np
import argparse
import contextlib
from collections import deque
import io as _io
import json
import os
import select
import subprocess
import sys
import time

EXPECT_TCP = [0.0, 0.0, 0.1537, 0.0, 0.0, 0.0]
EXPECT_PAYLOAD = 1.551

# forceMode, as measured on this arm: the baseline to beat.
FM_GAIN = 0.93e-3        # m/s per N, all axes
FM_LAG = 0.090           # s
FM_TAP_SLOPE = 2.0       # N of peak force above the commanded force, per mm/s of approach


# --------------------------------------------------------------------------- args

def csv6(s):
    v = [float(x) for x in str(s).split(',')]
    if len(v) == 1:
        v = v * 6
    if len(v) == 2:             # translational, rotational
        v = [v[0]] * 3 + [v[1]] * 3
    assert len(v) == 6, s
    return np.array(v)


def csv3(s):
    v = [float(x) for x in str(s).split(',')]
    assert len(v) == 3, s
    return np.array(v)


def csvn(s):
    return np.array([float(x) for x in str(s).split(',')])


def offsets_arg(s):
    """"tcp=0,flange=-0.154" -> [("tcp", 0.0), ("flange", -0.154)]"""
    out = []
    for part in s.split(','):
        name, _, val = part.partition('=')
        out.append((name.strip(), float(val)))
    if len(out) < 2:
        raise argparse.ArgumentTypeError('need at least two push points')
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='teleop',
                    choices=('teleop', 'center', 'free', 'tap', 'stall', 'ramp', 'selftest'))
    ap.add_argument('--replay', default=None, help='re-run the analysis of a saved log')
    ap.add_argument('--ip', default='192.168.0.100')
    ap.add_argument('--hz', type=float, default=500.0)

    g = ap.add_argument_group('admittance (axes of --frame; one value, "trans,rot", or 6)')
    g.add_argument('--k', type=csv6, default=csv6('300,10'), help='stiffness N/m, Nm/rad')
    g.add_argument('--d', type=csv6, default=csv6('1000,20'),
                   help='damping N s/m, Nm s/rad. 1/D is the free-space admittance: '
                        '1000 N s/m = 1 mm/s per N, forceMode was 0.93')
    g.add_argument('--m', type=csv6, default=csv6('15,0.6'),
                   help='virtual mass kg, kg m^2; M/D is the response time constant (15 ms). A trade: '
                        'too light and the 25 Hz loop through getActualTCPForce (it reads the arm\'s own '
                        'motion as ~20 kg, 34 ms late) limit-cycles -- 10 kg did, 9 N rms, '
                        'fdcc-tap-20260922-165218; too heavy and hard contact chatters (fakes: >= 17 kg '
                        'at 27 mm/s); 15 was clean in fdcc-tap-20260922-171421. Rotation: the 154 mm '
                        'flange-TCP lever turns the lateral 25 Hz force into torque; 0.3 kg m^2 rang in '
                        'teleop, 0.6 did not')
    g.add_argument('--sel', type=csv6, default=csv6('1'),
                   help='1 = compliant, 0 = stiff (position tracking at --stiff-gain). '
                        '"1,0" locks rotation only')
    g.add_argument('--frame', choices=('tool', 'base'), default='tool',
                   help='compliance frame: tool = TCP axes (moves with the tool), base = fixed')
    g.add_argument('--point', type=csv3, default=csv3('0,0,0'),
                   help='compliance point in the TCP frame [m]; default the TCP itself')
    g.add_argument('--ft-ref', choices=('tcp', 'flange'), default='flange',
                   help='point getActualTCPForce\'s moment is taken about: the flange, measured with '
                        '--mode center on 2026-09-22 (UR\'s docs say the TCP)')
    g.add_argument('--ft-sign', type=float, choices=(1.0, -1.0), default=1.0,
                   help='+1: getActualTCPForce is the external force ON the tool (push +x reads +x)')
    g.add_argument('--deadband', type=csvn, default=csvn('1.5,0.15'),
                   help='soft deadband on |F| [N], |tau| [Nm]; the F/T error is ~1.8 N median')
    g.add_argument('--f-cut', type=float, default=30.0, help='low-pass on the wrench [Hz], 0 = off')
    g.add_argument('--f-order', type=int, choices=(1, 2), default=1,
                   help='1 = first order; 2 = Butterworth, -40 dB/decade: cuts the 25 Hz mode harder '
                        'for the same lag at low frequency')
    g.add_argument('--stiff-gain', type=float, default=5.0,
                   help='position gain on stiff axes [1/s]')
    g.add_argument('--ff-release', type=csvn, default=csvn('3,0.3'),
                   help='target-velocity feedforward fades to zero as the environment pushes back '
                        'against the motion by this much [N, Nm]; 0 = never fade')
    g.add_argument('--ff-recover', type=float, default=0.3,
                   help='the fade drops at once but takes this long to recover from 0 to 1 [s]. A '
                        'symmetric fade modulated a 40 mm/s feedforward at 25 Hz -- ~17 mm/s per N of '
                        'force wobble -- and rang in teleop (fdcc-teleop-20260922-171759)')

    g = ap.add_argument_group('safety')
    g.add_argument('--vmax', type=csv6, default=csv6('0.08,0.9'),
                   help='speed clamp m/s, rad/s (norms; first and fourth used)')
    g.add_argument('--amax', type=csv6, default=csv6('2,4'),
                   help='acceleration clamp m/s^2, rad/s^2. It also bounds how fast contact is shed: '
                        '0.5 could not back off a 27 mm/s impact before 45 N (fdcc-teleop-20260922-164226)')
    g.add_argument('--fmax', type=float, default=25.0, help='force clamp on the controller input [N]')
    g.add_argument('--tmax', type=float, default=2.0, help='torque clamp [Nm]')
    g.add_argument('--f-abort', type=float, default=45.0, help='stop the run above this |F| [N]')
    g.add_argument('--t-abort', type=float, default=5.0, help='stop the run above this |tau| [Nm]')
    g.add_argument('--leash', type=csv6, default=csv6('0.03,0.25'),
                   help='teleop: the stick cannot push the target further than this from the arm [m, rad]. '
                        'It never drags the target after the arm')
    g.add_argument('--max-drift', type=float, default=0.25,
                   help='stop the run if the TCP gets this far from where it started [m]')
    g.add_argument('--speedl-cycles', type=float, default=1,
                   help='speedL time argument, in control cycles. Keep it 1: the control script only '
                        'takes a new target every t, and 4 (8 ms) buzzed at 125 Hz. Not a stall guard')
    g.add_argument('--watchdog-hz', type=float, default=20.0,
                   help='RTDE watchdog: the controller stops the program if the host is silent '
                        'for 1/hz. 0 = off')
    g.add_argument('--stop-acc', type=float, default=1.0, help='speedStop deceleration [m/s^2]')
    g.add_argument('--expect-payload', type=float, default=EXPECT_PAYLOAD)
    g.add_argument('--expect-tcp', type=csv6, default=np.array(EXPECT_TCP))
    g.add_argument('--ignore-mismatch', action='store_true',
                   help='run even if the controller payload / TCP differ from --expect-*')
    g.add_argument('--no-zero', action='store_true', help='skip zeroFtSensor at start')

    g = ap.add_argument_group('teleop')
    g.add_argument('--teleop-hz', type=float, default=100.0)
    g.add_argument('--teleop-accel', type=csv6, default=csv6('0.5,2.0'),
                   help='slew limit on the stick target m/s^2, rad/s^2. Unshaped, the stick went '
                        '0 -> 49 mm/s in 25 ms: the arm buzzed (~125 Hz) on every start and the '
                        'fade tripped on the arm\'s own acceleration (fdcc-teleop-20260922-172321). '
                        'Contact reaction is --amax, not this')
    g.add_argument('--teleop-speed', type=csv6, default=csv6('0.08,0.9'),
                   help='DualSense full-stick target speed m/s, rad/s')

    g = ap.add_argument_group('center / free')
    g.add_argument('--offsets', type=offsets_arg, default=offsets_arg('tcp=0,flange=-0.154'),
                   help='center: push points name=offset along tool z from the TCP [m], outward positive. '
                        'Only points you can actually push at')
    g.add_argument('--seconds', type=float, default=3.0, help='length of one push window [s]')
    g.add_argument('--reps', type=int, default=4, help='push windows per point')
    g.add_argument('--seg-drift', type=float, default=0.05,
                   help='end a push window when the TCP is this far from its start [m]')
    g.add_argument('--f-min', type=float, default=3.0, help='use samples pushed harder than this [N]')
    g.add_argument('--t-min', type=float, default=0.2, help='free: rotation fit threshold [Nm]')
    g.add_argument('--v-min', type=float, default=1e-3, help='center: ignore push-point speeds below [m/s]')

    g = ap.add_argument_group('tap')
    g.add_argument('--tap-dir', type=csv3, default=csv3('0,0,-1'), help='approach direction, base frame')
    g.add_argument('--tap-speeds', type=csvn, default=csvn('2,4,6,8'), help='approach speeds [mm/s]')
    g.add_argument('--tap-reps', type=int, default=2)
    g.add_argument('--tap-travel', type=float, default=0.03, help='give up without contact after [m]')
    g.add_argument('--tap-press', type=float, default=5.0,
                   help='after contact the target sits press/K inside the surface: steady force [N]')
    g.add_argument('--tap-hold', type=float, default=2.0, help='hold after contact [s]')
    g.add_argument('--f-contact', type=float, default=5.0, help='contact threshold [N]')
    g.add_argument('--tap-ramp', type=float, default=0.3,
                   help='ramp up to the approach speed over this long [s]. A step reads as a false '
                        'contact: the F/T sees the arm\'s own acceleration as ~20 kg')

    g = ap.add_argument_group('ramp (open-loop speedL, no admittance: does the ARM buzz?)')
    g.add_argument('--ramp-dir', type=csv3, default=csv3('0,1,0'), help='base frame; out and back')
    g.add_argument('--ramp-speed', type=float, default=0.05, help='cruise speed [m/s]')
    g.add_argument('--ramp-accels', type=csvn, default=csvn('0.1,0.5,2'), help='ramp accelerations [m/s^2]')
    g.add_argument('--ramp-cruise', type=float, default=0.3, help='time at cruise speed [s]')

    g = ap.add_argument_group('stall')
    g.add_argument('--stall-dir', type=csv3, default=csv3('0,0,1'), help='base frame; up by default')
    g.add_argument('--stall-speed', type=float, default=0.005, help='[m/s]')
    g.add_argument('--stall-gap', type=float, default=0.3, help='how long the host goes silent [s]')

    ap.add_argument('--out', default=None, help='log file (default fdcc-<mode>-<time>.npz)')
    return ap


# ---------------------------------------------------------------------- geometry

def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def pose_error(target, actual):
    """target - actual in the base frame: [dp, rotvec(R_t R_a^T)]."""
    dp = np.asarray(target[:3]) - np.asarray(actual[:3])
    dr = (R.from_rotvec(target[3:]) * R.from_rotvec(actual[3:]).inv()).as_rotvec()
    return np.r_[dp, dr]


def leash(target, actual, max_dp, max_dr):
    """
    Pull the target back so it is at most max_dp / max_dr from the actual pose.
    Without this, pushing the stick into a wall winds the target up and the arm
    lunges when contact breaks.
    """
    e = pose_error(target, actual)
    n = np.linalg.norm(e[:3])
    if n > max_dp:
        target[:3] = np.asarray(actual[:3]) + e[:3] * (max_dp / n)
    n = np.linalg.norm(e[3:])
    if n > max_dr:
        dr = e[3:] * (max_dr / n)
        target[3:] = (R.from_rotvec(dr) * R.from_rotvec(actual[3:])).as_rotvec()
    return target


def clamp_wrench(w, fmax, tmax):
    """Scale the linear and angular halves to at most fmax / tmax. Used for twists too."""
    w = np.array(w, float)
    n = np.linalg.norm(w[:3])
    if n > fmax:
        w[:3] *= fmax / n
    n = np.linalg.norm(w[3:])
    if n > tmax:
        w[3:] *= tmax / n
    return w


def soft_deadband(w, fdb, tdb):
    """Shrink |F| by fdb and |tau| by tdb, keeping direction; zero inside. Works on (..., 6)."""
    w = np.array(w, float)
    for sl, db in ((slice(0, 3), fdb), (slice(3, 6), tdb)):
        n = np.linalg.norm(w[..., sl], axis=-1, keepdims=True)
        w[..., sl] *= np.clip(n - db, 0.0, None) / np.maximum(n, 1e-12)
    return w


def point_position(pose, off):
    """Base-frame position of a point fixed at `off` in the TCP frame. (6,) or (N, 6)."""
    pose = np.asarray(pose, float)
    return pose[..., :3] + R.from_rotvec(pose[..., 3:]).apply(np.asarray(off, float))


def flange_position(pose, tcp_offset):
    """T_base_flange = T_base_tcp * T_tcp_offset^-1, position part. (6,) or (N, 6)."""
    pose, tcp_offset = np.asarray(pose, float), np.asarray(tcp_offset, float)
    Rf = R.from_rotvec(pose[..., 3:]) * R.from_rotvec(tcp_offset[3:]).inv()
    return pose[..., :3] - Rf.apply(tcp_offset[:3])


def ref_point(pose, tcp_offset, which):
    return np.asarray(pose, float)[..., :3] if which == 'tcp' else flange_position(pose, tcp_offset)


def shift_wrench(w, a, b):
    """
    Wrench referenced at point a -> the same wrench referenced at point b:
        F_b = F_a        tau_b = tau_a + (a - b) x F
    """
    w = np.asarray(w, float)
    F = w[..., :3]
    return np.concatenate([F, w[..., 3:] + np.cross(np.asarray(a) - np.asarray(b), F)], axis=-1)


def shift_twist(t, a, b):
    """
    Twist of body point a -> twist of body point b on the same rigid body:
        v_b = v_a + omega x (b - a)        omega unchanged
    Dual of shift_wrench: F.v + tau.omega is the same at every point.
    """
    t = np.asarray(t, float)
    w = t[..., 3:]
    return np.concatenate([t[..., :3] + np.cross(w, np.asarray(b) - np.asarray(a)), w], axis=-1)


def twist_between(a, b, h):
    """Constant twist taking TCP pose a to pose b in h seconds."""
    dr = (R.from_rotvec(b[3:]) * R.from_rotvec(a[3:]).inv()).as_rotvec()
    return np.r_[(np.asarray(b[:3]) - np.asarray(a[:3])) / h, dr / h]


def frame_mats(pose, frame):
    """Compliance-frame rotation(s): columns are the frame axes in base."""
    pose = np.asarray(pose, float)
    if frame == 'tool':
        return R.from_rotvec(pose[..., 3:]).as_matrix()
    return np.broadcast_to(np.eye(3), pose.shape[:-1] + (3, 3))


def to_frame(x, Rc):
    """Base-frame 6-vector(s) -> compliance-frame components (R^T x on both halves)."""
    return np.concatenate([np.einsum('...ji,...j->...i', Rc, x[..., :3]),
                           np.einsum('...ji,...j->...i', Rc, x[..., 3:])], axis=-1)


def from_frame(x, Rc):
    return np.concatenate([np.einsum('...ij,...j->...i', Rc, x[..., :3]),
                           np.einsum('...ij,...j->...i', Rc, x[..., 3:])], axis=-1)


# -------------------------------------------------------------------- controller

class LowPass:
    """Per-component low-pass: first order, or a 2nd-order Butterworth (bilinear, prewarped)."""

    def __init__(self, f_cut, order, dt):
        self.off = f_cut <= 0
        if self.off:
            return
        if order == 1:
            a = 1.0 - np.exp(-2 * np.pi * f_cut * dt)
            self.b, self.a = np.array([a, 0, 0]), np.array([1.0, a - 1.0, 0])
        else:
            k = np.tan(np.pi * f_cut * dt)
            n = 1.0 / (1.0 + np.sqrt(2) * k + k * k)
            self.b = np.array([k * k, 2 * k * k, k * k]) * n
            self.a = np.array([1.0, 2 * (k * k - 1) * n, (1 - np.sqrt(2) * k + k * k) * n])
        self.reset()

    def reset(self):
        self.x = self.y = None

    def __call__(self, u):
        if self.off:
            return u
        if self.x is None:                   # start at rest on the first sample
            self.x, self.y = [u, u], [u, u]
        y = self.b[0] * u + self.b[1] * self.x[0] + self.b[2] * self.x[1] - self.a[1] * self.y[0] - self.a[2] * self.y[1]
        self.x, self.y = [u, self.x[0]], [y, self.y[0]]
        return y


class Admittance:
    """
    Per cycle, at the compliance point p and in the compliance frame C:

        M dv/dt + D (v - v_t) + K e = F_p,     e = x_p - x_t

    discretised implicitly in the damping term,

        v+ = (M v + dt (F_p - K e + D v_t)) / (M + D dt),

    which is stable for any D >= 0 and still defined at M = 0 (a pure damper).
    Stiff axes (sel 0) ignore F and track: v = v_t - stiff_gain * e. Then the
    speed and acceleration clamps, then the twist of p is moved to the TCP.

    e uses the MEASURED pose; v is the controller's own state (the measured twist
    is noisy and lags). self.v is the commanded twist of p, base frame.
    """

    def __init__(self, K, D, M, sel, frame, point, tcp_offset, ft_ref, ft_sign, vmax, amax,
                 deadband, f_cut, stiff_gain, fmax, tmax, dt, ff_release=(0.0, 0.0), f_order=1,
                 ff_recover=0.0):
        self.K, self.D, self.M = (np.array(x, float) for x in (K, D, M))
        self.sel = np.array(sel, float)
        self.frame, self.point = frame, np.array(point, float)
        self.tcp, self.ft_ref, self.ft_sign = np.array(tcp_offset, float), ft_ref, float(ft_sign)
        self.vmax, self.amax = np.array(vmax, float), np.array(amax, float)
        self.deadband = np.array(deadband, float)
        self.lp = LowPass(f_cut, f_order, dt)
        self.stiff_gain, self.fmax, self.tmax, self.dt = stiff_gain, fmax, tmax, dt
        self.ff_release = np.array(ff_release, float)
        self.ff_recover = float(ff_recover)
        self.ff_gain = np.ones(2)          # feedforward scale, translation / rotation (logged)
        bad = (self.sel != 0) & (self.M + self.D * dt <= 0)
        if bad.any():
            raise ValueError(f'compliant axes {np.flatnonzero(bad)} have M = D = 0')
        self.reset()

    def reset(self):
        self.v = np.zeros(6)
        self.lp.reset()

    def wrench_at_point(self, pose, w_meas):
        p = point_position(pose, self.point)
        w = shift_wrench(self.ft_sign * np.asarray(w_meas, float),
                         ref_point(pose, self.tcp, self.ft_ref), p)
        return w, p

    def step(self, pose, w_meas, target, vt_tcp, k_scale=1.0):
        pose, target = np.asarray(pose, float), np.asarray(target, float)
        # 1. measured wrench -> p, filter, deadband, clamp
        w, p = self.wrench_at_point(pose, w_meas)
        w = clamp_wrench(soft_deadband(self.lp(w), *self.deadband), self.fmax, self.tmax)
        # 2. error and target twist at p, everything into C
        pt = point_position(target, self.point)
        e = -pose_error(np.r_[pt, target[3:]], np.r_[p, pose[3:]])
        vt = shift_twist(vt_tcp, target[:3], pt)
        Rc = frame_mats(pose, self.frame)
        wc, ec, vtc, vc = (to_frame(x, Rc) for x in (w, e, vt, self.v))
        # 3. dynamics. The feedforward D v_t is a FORCE in disguise: in contact it
        # presses at D |v_t| whatever the spring and leash allow -- 27 N at 27 mm/s,
        # above the 25 N clamp, so the arm could not back off
        # (fdcc-teleop-20260922-164226). Fade it out as the environment pushes back
        # against the direction of travel; free-space tracking is unchanged.
        K, D = self.K * k_scale, self.D * np.sqrt(k_scale)
        # Fast attack, slow release: a symmetric fade turns any force wobble into a
        # feedforward modulation of |v_t| / release per N (17 mm/s per N at 50 mm/s),
        # which is what rang at 25 Hz in fdcc-teleop-20260922-171759.
        vff = vtc.copy()
        for j, (sl, rel) in enumerate(((slice(0, 3), self.ff_release[0]), (slice(3, 6), self.ff_release[1]))):
            n = np.linalg.norm(vff[sl])
            g = 1.0
            if rel > 0 and n > 0:
                g = float(np.clip(1.0 + (wc[sl] @ vff[sl]) / (n * rel), 0.0, 1.0))
            if self.ff_recover > 0:
                g = min(g, self.ff_gain[j] + self.dt / self.ff_recover)
            self.ff_gain[j] = g
            vff[sl] *= g
        v_new = (self.M * vc + self.dt * (wc - K * ec + D * vff)) / (self.M + D * self.dt)
        stiff = self.sel == 0
        v_new[stiff] = vtc[stiff] - self.stiff_gain * ec[stiff]
        v_new = from_frame(v_new, Rc)
        # 4. clamps. The state is clamped too, so it cannot wind up behind them.
        v_new = clamp_wrench(v_new, self.vmax[0], self.vmax[3])
        self.v = self.v + clamp_wrench(v_new - self.v, self.amax[0] * self.dt, self.amax[3] * self.dt)
        v_tcp = clamp_wrench(shift_twist(self.v, p, pose[:3]), self.vmax[0], self.vmax[3])
        return v_tcp, {'w': w, 'e': ec}


def make_admittance(args, tcp, dt):
    K, sel, db = args.k.copy(), args.sel.copy(), np.array(args.deadband, float)
    if args.mode in ('center', 'free'):
        K[:], sel[:] = 0.0, 1.0            # no spring, everything compliant
    if args.mode == 'center':
        db[:] = 0.0                        # a deadband bends s(d); pushes are short, drift is not an issue
    return Admittance(K, args.d, args.m, sel, args.frame, args.point, tcp, args.ft_ref, args.ft_sign,
                      args.vmax, args.amax, db, args.f_cut, args.stiff_gain, args.fmax, args.tmax, dt,
                      getattr(args, 'ff_release', (0.0, 0.0)), getattr(args, 'f_order', 1),
                      getattr(args, 'ff_recover', 0.0))


# ----------------------------------------------------------------------- session

COLS = [('t', 1), ('pose', 6), ('target', 6), ('twist', 6), ('v_cmd', 6), ('v_point', 6),
        ('tcp_force', 6), ('ft_raw', 6), ('wrench_point', 6), ('err', 6), ('q', 6),
        ('k_scale', 1), ('sel', 6), ('loop_dt', 1), ('seg', 1), ('rep', 1), ('phase', 1), ('ff_gain', 2)]
NCOL = sum(w for _, w in COLS)
NAN6 = np.full(6, np.nan)


def unpack(L):
    out, i = {}, 0
    for name, w in COLS:
        out[name] = L[:, i] if w == 1 else L[:, i:i + w]
        i += w
    return out


class Abort(Exception):
    pass


class HardwareIO:
    def prompt(self, msg, kick, **ctx):
        """input(), but kicking the watchdog every 10 ms while waiting."""
        print(msg, end='', flush=True)
        while True:
            kick()
            if select.select([sys.stdin], [], [], 0.01)[0]:
                sys.stdin.readline()
                return

    def sleep(self, s):
        time.sleep(s)

    def event(self, name, **ctx):
        pass


class Session:
    """
    One control cycle = read state, safety checks, admittance step, speedL, log.
    Log columns are COLS: base frame except err (compliance frame), v_cmd is the
    TCP twist sent, v_point the controller's twist of p, wrench_point the wrench
    at p after sign, filter, deadband and clamp.
    """

    def __init__(self, ctrl, recv, adm, args, io, clock=time.perf_counter):
        self.ctrl, self.recv, self.adm, self.args, self.io, self.clock = ctrl, recv, adm, args, io, clock
        self.dt = 1.0 / args.hz
        self.speedl_time = args.speedl_cycles * self.dt
        self.k_scale = 1.0
        self.watchdog = False
        self.buf, self.n = np.full((1 << 15, NCOL), np.nan), 0
        self.t0, self.t_last, self.max_gap, self.slow = clock(), None, 0.0, 0
        self.v_last = np.zeros(6)          # last TCP twist actually sent
        try:
            recv.getFtRawWrench()
            self.has_raw = True
        except Exception:
            self.has_raw = False
        self.home = self.pose()

    def pose(self):
        return np.array(self.recv.getActualTCPPose(), float)

    def data(self):
        return unpack(self.buf[:self.n])

    def kick(self):
        if self.watchdog:
            self.ctrl.kickWatchdog()

    def idle(self, seconds):
        """Wait, never blocking more than 10 ms without feeding the watchdog."""
        t_end = self.clock() + seconds
        while self.clock() < t_end:
            self.kick()
            self.io.sleep(min(0.01, max(t_end - self.clock(), 1e-4)))

    def prompt(self, msg, **ctx):
        self.io.prompt(msg, self.kick, **ctx)

    def ramp_down(self, timeout=0.5):
        """
        Bring the arm to rest with speedL, feeding the watchdog every cycle.

        speedStop() alone is NOT safe with the watchdog on: the control script runs
        stopl() synchronously and ur_rtde waits for it, so the host sends nothing
        for the whole deceleration. From 10 mm/s and 0.079 rad/s that outlasted the
        50 ms watchdog and protective-stopped the arm (C207A0 Fieldbus input
        disconnected, fdcc-center-20260922-162200.npz). Stopped this way first,
        the speedStop that follows has nothing left to decelerate.
        """
        a, dt = self.args, self.dt
        v = np.array(self.v_last, float)
        t0 = self.clock()
        while self.clock() - t0 < timeout:
            t_start = self.ctrl.initPeriod()
            if self.recv.isProtectiveStopped() or self.recv.isEmergencyStopped():
                return
            v = v - clamp_wrench(v, a.amax[0] * dt, a.amax[3] * dt)
            if self.ctrl.speedL(v.tolist(), a.amax[0], self.speedl_time) is False:
                return
            self.ctrl.waitPeriod(t_start)
            tw = np.array(self.recv.getActualTCPSpeed(), float)
            if not v.any() and np.linalg.norm(tw[:3]) < 5e-4 and np.linalg.norm(tw[3:]) < 5e-3:
                break

    def stop(self):
        """Ramp to rest through speedL, then speedStop (see ramp_down)."""
        self.ramp_down()
        self.ctrl.speedStop(self.args.stop_acc)
        self.v_last = np.zeros(6)
        self.adm.reset()
        self.t_last = None

    def zero(self):
        """Stop, let it settle, zero. ONLY out of contact."""
        self.stop()
        self.idle(0.3)
        self.ctrl.zeroFtSensor()
        self.idle(0.3)

    def cycle(self, target, vt, tag=(0, 0, 0), v_override=None, send=True, check_stop=True):
        a = self.args
        t_start = self.ctrl.initPeriod()
        now = self.clock()
        loop_dt = np.nan if self.t_last is None else now - self.t_last
        self.t_last = now
        if loop_dt > self.max_gap:
            self.max_gap = loop_dt
        if loop_dt > 2 * self.dt:
            self.slow += 1
        self.stopped = self.recv.isProtectiveStopped() or self.recv.isEmergencyStopped()
        if self.stopped and check_stop:
            raise Abort('protective/emergency stop')
        pose = self.pose()
        twist = np.array(self.recv.getActualTCPSpeed(), float)
        wm = np.array(self.recv.getActualTCPForce(), float)
        f, tq = np.linalg.norm(wm[:3]), np.linalg.norm(wm[3:])
        if f > a.f_abort or tq > a.t_abort:
            raise Abort(f'wrench over the abort limit: |F| {f:.1f} N  |tau| {tq:.2f} Nm')
        drift = np.linalg.norm(pose[:3] - self.home[:3])
        if drift > a.max_drift:
            raise Abort(f'TCP {1e3 * drift:.0f} mm from where the run started (--max-drift)')
        target = pose if target is None else target
        if v_override is None and send:
            v_cmd, info = self.adm.step(pose, wm, target, vt, self.k_scale)
        else:
            v_cmd, info = np.asarray(v_override, float), {'w': NAN6, 'e': NAN6}
        if send:
            if self.ctrl.speedL(v_cmd.tolist(), a.amax[0], self.speedl_time) is False:
                raise Abort('speedL refused -- control script not running (watchdog stop?)')
            self.v_last = v_cmd
        else:
            v_cmd = NAN6
        raw = self.recv.getFtRawWrench() if self.has_raw else NAN6
        row = np.r_[now - self.t0, pose, target, twist, v_cmd, self.adm.v, wm, raw, info['w'],
                    info['e'], self.recv.getActualQ(), self.k_scale, self.adm.sel, loop_dt, tag,
                    self.adm.ff_gain]
        if self.n == len(self.buf):
            self.buf = np.concatenate([self.buf, np.full_like(self.buf, np.nan)])
        self.buf[self.n] = row
        self.n += 1
        self.ctrl.waitPeriod(t_start)
        return pose, twist, wm, info

    def goto(self, target, tag):
        """
        Position-controlled return through the same speedL path, all axes stiff
        (moveL would block without feeding the watchdog). F/T aborts stay live.
        """
        target = np.array(target, float)
        sel = self.adm.sel.copy()
        self.adm.sel[:] = 0.0
        t0 = self.clock()
        try:
            while True:
                pose = self.cycle(target, np.zeros(6), tag)[0]
                e = pose_error(target, pose)
                if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < 3e-3:
                    break
                if self.clock() - t0 > 30.0:
                    raise Abort('return to start did not converge')
        finally:
            self.adm.sel[:] = sel
            self.stop()


# ------------------------------------------------------------------------- modes

def make_dualsense(args, pose):
    """
    Slow (interface.py -> env.py imports cv2, h5py, camera, wsg; then HID open):
    seconds of host silence. run() calls this BEFORE arming the watchdog -- doing it
    inside the mode tripped C207A0 before the first cycle.
    """
    from interface import DualSenseInterface
    print(args.teleop_speed)
    return DualSenseInterface(pose, xyzspeed=args.teleop_speed[0], rpyspeed=args.teleop_speed[3],
                              enable_zadaptive=False)


def mode_teleop(sess, args, io):
    iface = sess.iface
    pose = sess.pose()
    iface.targ_pose = pose.copy()          # built before zeroing; start from where the arm is now
    edge = Edge()
    adm, dt = sess.adm, sess.dt
    every = max(1, round(args.hz / args.teleop_hz))
    sel0, rot_locked = adm.sel.copy(), False
    vt, v_slew = np.zeros(6), np.zeros(6)
    print('running -- Ctrl-C to stop')
    i = 0
    while True:
        if i % every == 0:
            before = iface.targ_pose.copy()
            if iface.update(every * dt) == -1:
                iface.targ_pose[:] = before
            # Slew-limit the stick: move the target at a rate-limited copy of what
            # the stick asked for, rather than letting it step.
            h = every * dt
            stick = twist_between(before, iface.targ_pose, h)
            v_slew = v_slew + clamp_wrench(stick - v_slew, args.teleop_accel[0] * h, args.teleop_accel[3] * h)
            iface.targ_pose[:3] = before[:3] + v_slew[:3] * h
            iface.targ_pose[3:] = (R.from_rotvec(v_slew[3:] * h) * R.from_rotvec(before[3:])).as_rotvec()
            st = iface.dualsense.state
            if edge(st, 'DpadUp'):
                sess.k_scale *= 1.5
                print(f'\nstiffness x{sess.k_scale:.2f}  K {np.round(adm.K * sess.k_scale, 2)}')
            if edge(st, 'DpadDown'):
                sess.k_scale /= 1.5
                print(f'\nstiffness x{sess.k_scale:.2f}  K {np.round(adm.K * sess.k_scale, 2)}')
            if edge(st, 'Cross'):
                iface.targ_pose = pose.copy()
                before = pose.copy()             # a jump, not a velocity
                v_slew[:] = 0
                print('\ntarget re-anchored')
            if edge(st, 'Square'):
                rot_locked = not rot_locked
                if rot_locked:
                    adm.sel[3:] = 0.0
                    iface.targ_pose[3:] = pose[3:]      # hold where it is, not where the target was
                    before[3:] = pose[3:]              # a jump, not a velocity
                else:
                    adm.sel[3:] = sel0[3:]
                print(f'\nrotation {"LOCKED" if rot_locked else "free"}  sel {adm.sel}')
            if edge(st, 'DpadLeft'):
                print('\nzeroing F/T -- hands off, out of contact')
                sess.zero()
                print('F/T zeroed')
            # Leash: stop the STICK winding the target away from the arm, but never
            # drag the target after the arm. Dragging moved the equilibrium when the
            # arm was pushed by hand, and the dragged target's velocity went into the
            # feedforward, cancelled the damping and ran the arm to --vmax, 80 mm/s
            # (fdcc-teleop-20260922-174455).
            e0, e1 = pose_error(before, pose), pose_error(iface.targ_pose, pose)
            for sl, lim in ((slice(0, 3), args.leash[0]), (slice(3, 6), args.leash[3])):
                if np.linalg.norm(e1[sl]) > lim and np.linalg.norm(e1[sl]) > np.linalg.norm(e0[sl]):
                    iface.targ_pose[sl] = before[sl]
                    v_slew[sl] = 0.0
            # only the stick's own motion of the target feeds forward
            vt = twist_between(before, iface.targ_pose, every * dt)
        pose, twist, wm, info = sess.cycle(iface.targ_pose, vt)
        if i % int(args.hz / 5) == 0:
            e = info['e']
            print(f'|e| {1e3 * np.linalg.norm(e[:3]):5.1f} mm {np.degrees(np.linalg.norm(e[3:])):5.2f} deg'
                  f'  F@p {np.round(info["w"][:3], 1)}  v {1e3 * np.linalg.norm(adm.v[:3]):4.1f} mm/s'
                  f'  K x{sess.k_scale:.2f}   ', end='\r')
        i += 1


def push_windows(sess, args, io, seg, label, d, home):
    """Timed push windows at one point, returning to `home` between them."""
    for rep in range(args.reps):
        print(f'    push {rep + 1}/{args.reps}: shove at "{label}" NOW ({args.seconds:.0f} s)', flush=True)
        io.event('push_start', d=d, rep=rep)
        t0 = sess.clock()
        try:
            while sess.clock() - t0 < args.seconds:
                pose = sess.cycle(home, np.zeros(6), (seg, rep, 0))[0]
                if np.linalg.norm(pose[:3] - home[:3]) > args.seg_drift:
                    print(f'      push {rep + 1} ended early: {1e3 * args.seg_drift:.0f} mm from start')
                    break
        finally:
            io.event('push_end')
            sess.stop()
        print('      hands off -- returning', flush=True)
        sess.idle(0.8)
        sess.goto(home, (seg, rep, 2))


def mode_center(sess, args, io):
    tr, rot = args.m[0] / args.d[0], args.m[3] / args.d[3]
    if abs(tr - rot) > 0.2 * max(tr, rot):
        print(f'NOTE: M/D differs between translation ({1e3 * tr:.0f} ms) and rotation '
              f'({1e3 * rot:.0f} ms); transients will bias the spin ratio a little.')
    print('push GENTLY (< ~8 N): samples where a speed clamp bites are excluded from the fit')
    for seg, (label, d) in enumerate(args.offsets):
        sess.prompt(f'\n[{seg + 1}/{len(args.offsets)}] ready to push at "{label}" '
                    f'({1e3 * d:+.0f} mm from TCP along tool z, sideways)? Enter ', d=d)
        print('    hands OFF while the F/T zeroes')
        sess.zero()
        push_windows(sess, args, io, seg, label, d, sess.pose())
        try:
            analyze_center(sess.data(), args, {'tcp_offset': sess.adm.tcp}, only=seg)
        except Exception as ex:          # never lose the run over a report bug
            print(f'    (segment report failed: {ex!r})')


def mode_free(sess, args, io):
    print('Shove the tool in varied directions -- sideways, along, twisting -- and let go.\n'
          'Short shoves are fine. Keep under ~8 N so the speed clamp does not bite.')
    sess.prompt('ready? Enter ', d=None)
    print('    hands OFF while the F/T zeroes')
    sess.zero()
    push_windows(sess, args, io, 0, 'anywhere', None, sess.pose())


def mode_tap(sess, args, io):
    dirn = unit(args.tap_dir)
    speeds = np.asarray(args.tap_speeds, float) * 1e-3
    if np.any(speeds > args.vmax[0]):
        print(f'NOTE: speeds above --vmax {1e3 * args.vmax[0]:.0f} mm/s are skipped')
    sess.prompt(f'Put the tool 5-20 mm from a hard surface along {np.round(dirn, 2)} (base). '
                f'Hand on the e-stop. Enter ')
    home = sess.pose()
    adm = sess.adm
    for seg, v in enumerate(speeds):
        if v > args.vmax[0]:
            continue
        for rep in range(args.tap_reps):
            print(f'  tap {1e3 * v:.0f} mm/s  #{rep + 1}', flush=True)
            sess.zero()                                   # at home, out of contact
            target, contact, phase, t0 = home.copy(), False, 0, sess.clock()
            try:
                while True:
                    if not contact:
                        vr = v * min(1.0, (sess.clock() - t0) / max(args.tap_ramp, 1e-9))
                        vt = np.r_[dirn * vr, 0, 0, 0]
                        target[:3] += dirn * vr * sess.dt
                    pose, twist, wm, info = sess.cycle(target, vt, (seg, rep, phase))
                    fc = -args.ft_sign * wm[:3] @ dirn
                    if not contact:
                        if fc > args.f_contact:
                            contact, phase, t_c = True, 1, sess.clock()
                            dc = frame_mats(pose, args.frame).T @ dirn
                            k_dir = float(np.sum(adm.K[:3] * adm.sel[:3] * dc ** 2))
                            target = pose.copy()
                            if k_dir > 0:
                                target[:3] += dirn * args.tap_press / k_dir
                            vt = np.zeros(6)
                        elif np.linalg.norm(pose[:3] - home[:3]) > args.tap_travel:
                            print(f'    no contact within {1e3 * args.tap_travel:.0f} mm')
                            break
                        leash(target, pose, args.leash[0], args.leash[3])
                    elif sess.clock() - t_c > args.tap_hold:
                        break
            finally:
                sess.stop()
            sess.idle(0.3)
            sess.goto(home, (seg, rep, 2))


def mode_stall(sess, args, io):
    """
    A: speedL at --stall-speed, then the host goes silent for --stall-gap with NO
       watchdog. ur_rtde's speed_thread predicts the arm keeps moving.
    B: the same with the watchdog on. The controller should stop the program
       after 1/--watchdog-hz and the arm decelerate. This ends the run.
    """
    dirn = unit(args.stall_dir)
    v = np.r_[dirn * args.stall_speed, 0, 0, 0]
    wd = args.watchdog_hz if args.watchdog_hz > 0 else 20.0
    sess.prompt(f'The arm moves ~{1e3 * args.stall_speed * (0.5 + args.stall_gap):.0f} mm along '
                f'{np.round(dirn, 2)} (base), twice. Clear path, hand on the e-stop. Enter ')
    home = sess.pose()
    for part, use_wd in enumerate((False, True)):
        if use_wd:
            sess.goto(home, (part, 0, 2))
            sess.ctrl.setWatchdog(wd)
            sess.watchdog = True
            print(f'  B: watchdog ON at {wd:.0f} Hz ({1e3 / wd:.0f} ms)')
        else:
            print('  A: watchdog OFF')
        t0 = sess.clock()
        while sess.clock() - t0 < 0.5:
            sess.cycle(None, None, (part, 0, 0), v_override=v)
        # Silent gap. Keep READING through a stop (phase 3 once the stop flag is
        # up) -- aborting on it would cut the log before the deceleration.
        t0 = sess.clock()
        while sess.clock() - t0 < args.stall_gap:
            sess.cycle(None, None, (part, 0, 3 if getattr(sess, 'stopped', False) else 1),
                       send=False, check_stop=False)
        running = sess.ctrl.isProgramRunning()
        print(f'     after the gap: program running = {running}, protective stop = {sess.stopped}')
        if sess.stopped:
            if use_wd:
                print('     (the watchdog raises a PROTECTIVE stop on this controller -- clear it on the pendant)')
                return
            raise Abort('protective stop during the gap with the watchdog OFF -- not expected')
        if not use_wd:
            sess.stop()


def mode_ramp(sess, args, io):
    """
    Trapezoidal speedL profiles with NO force feedback: out along --ramp-dir, then
    back, at each --ramp-accels. If the ~125 Hz buzz seen at teleop starts shows up
    here too, it is the arm / UR's speed loop, not the admittance.
    phase 0 = ramping, 1 = cruising, 2 = return to start.
    """
    dirn = unit(args.ramp_dir)
    v, dt = args.ramp_speed, sess.dt
    travel = max(v * v / a + v * args.ramp_cruise for a in args.ramp_accels)
    sess.prompt(f'The arm moves up to {1e3 * travel:.0f} mm along {np.round(dirn, 2)} (base) and back, '
                f'{len(args.ramp_accels)} times. Clear path, hand on the e-stop. Enter ')
    home = sess.pose()
    for seg, acc in enumerate(args.ramp_accels):
        for rep_, sgn in enumerate((1.0, -1.0)):
            print(f'  {acc:.2f} m/s^2, {"out" if sgn > 0 else "back"}', flush=True)
            t_ramp = v / acc
            t_total = 2 * t_ramp + args.ramp_cruise
            t0 = sess.clock()
            try:
                while (tt := sess.clock() - t0) < t_total:
                    sp = v * min(1.0, tt / t_ramp, max(0.0, (t_total - tt) / t_ramp))
                    ramping = tt < t_ramp or tt > t_ramp + args.ramp_cruise
                    sess.cycle(None, None, (seg, rep_, 0 if ramping else 1),
                               v_override=np.r_[sgn * dirn * sp, 0, 0, 0])
            finally:
                sess.stop()
            sess.idle(0.3)
        sess.goto(home, (seg, 2, 2))


def buzz(err, dt, lo=95.0, hi=160.0, L=32):
    """rms of err (N, k) in [lo, hi] Hz, from 32-sample Hann windows; per window."""
    h = np.hanning(L)[:, None]
    f = np.fft.rfftfreq(L, dt)
    band = (f > lo) & (f < hi)
    out = []
    for s in range(0, len(err) - L + 1, L // 4):
        e = err[s:s + L] - err[s:s + L].mean(0)
        out.append(np.sqrt((np.abs(np.fft.rfft(e * h, axis=0))[band] ** 2).sum()) / L * 2)
    return np.array(out)


def analyze_ramp(d, args, meta):
    dt = float(np.nanmedian(np.diff(d['t'])))
    err = d['twist'][:, :3] - d['v_cmd'][:, :3]
    print('\n  95-160 Hz tracking error (measured - commanded), mm/s: median / 90%')
    print('  accel m/s^2     ramping           cruising')
    out = {}
    for seg, acc in enumerate(args.ramp_accels):
        cells = []
        for ph in (0, 1):
            parts = [buzz(err[b], dt) for b in blocks(d, ph) if d['seg'][b[0]] == seg and len(b) >= 32]
            x = np.concatenate(parts) if parts else np.array([np.nan])
            cells.append((np.nanmedian(x), np.nanpercentile(x, 90)))
        out[acc] = cells
        print(f'  {acc:5.2f}         {1e3 * cells[0][0]:5.2f} / {1e3 * cells[0][1]:5.2f}      '
              f'{1e3 * cells[1][0]:5.2f} / {1e3 * cells[1][1]:5.2f}')
    print('  For scale, fdcc-teleop-20260922-172321 (admittance on): speed changes 4.3 / 6.3, cruise 0.3 / 0.5.')
    return out


MODES = {'teleop': mode_teleop, 'center': mode_center, 'free': mode_free, 'ramp': mode_ramp,
         'tap': mode_tap, 'stall': mode_stall}


# ---------------------------------------------------------------------- analysis

def spin_ratio(pose, twist, force, d, v_min, f_min):
    """
    s = |omega| / |v| at the PUSH POINT, for one segment (from test-forcemode-center.py).

    This is the observable to fit, not the pivot location: pushing exactly at the
    compliance centre gives pure translation, where the pivot runs off to infinity
    but s simply goes to zero. Model:

        s(d) = |d - c| / (k + (d - c)^2),     k = a_t / a_r  (= D_rot / D_trans here)

    Returns (s, n_used, pivot_z_or_nan).
    """
    v, w = twist[:, :3], twist[:, 3:]
    Rb = R.from_rotvec(pose[:, 3:]).as_matrix()
    arm = np.einsum('nij,j->ni', Rb, np.array([0.0, 0.0, d]))
    v_push = v + np.cross(w, arm)                      # velocity of the pushed point
    vn, wn = np.linalg.norm(v_push, axis=1), np.linalg.norm(w, axis=1)
    m = (vn > v_min) & (np.linalg.norm(force[:, :3], axis=1) > f_min)
    if m.sum() < 50:
        return np.nan, m.sum(), np.nan
    s = float(np.median(wn[m] / vn[m]))
    mr = m & (wn > 0.02)
    if mr.sum() < 50:
        return s, m.sum(), np.nan
    r = np.cross(w[mr], v[mr]) / (wn[mr] ** 2)[:, None]
    pz = float(np.median(np.einsum('nij,nj->ni', Rb[mr].transpose(0, 2, 1), r)[:, 2]))
    return s, m.sum(), pz


def solve_centre(d, s):
    """
    Fit s_i = |d_i - c| / (k + (d_i - c)^2) over segments. Returns list of
    distinct (c, k, cost), best first -- |d - c| is symmetric about c, so two
    push points can admit two solutions.
    """
    def resid(x):
        c, k = x
        return s - np.abs(d - c) / (max(k, 1e-6) + (d - c) ** 2)

    sols = []
    for c0 in np.linspace(d.min() - 0.25, d.max() + 0.25, 40):
        for k0 in (0.002, 0.02, 0.2):
            try:
                r = least_squares(resid, [c0, k0], bounds=([-1.0, 1e-6], [1.0, 10.0]))
            except Exception:
                continue
            if not np.isfinite(r.cost):
                continue
            if not any(abs(r.x[0] - c) < 5e-3 and abs(r.x[1] - k) < 1e-3 for c, k, _ in sols):
                sols.append((float(r.x[0]), float(r.x[1]), float(r.cost)))
    sols.sort(key=lambda t: t[2])
    return sols[:3]


def analyze_center(d, args, meta, only=None):
    tcp = np.asarray(meta['tcp_offset'], float)
    vc = d['v_cmd']
    # s is pure geometry only while no clamp is shaping the motion
    unclamped = ((np.linalg.norm(vc[:, :3], axis=1) < 0.9 * args.vmax[0]) &
                 (np.linalg.norm(vc[:, 3:], axis=1) < 0.9 * args.vmax[3]))
    rows = []
    for seg, (label, dd) in enumerate(args.offsets):
        if only is not None and seg != only:
            continue
        seg_m = (d['seg'] == seg) & (d['phase'] == 0)
        m = seg_m & unclamped
        if not seg_m.any():
            continue
        s, n, pz = spin_ratio(d['pose'][m], d['twist'][m], d['tcp_force'][m], dd, args.v_min, args.f_min)
        clipped = int(seg_m.sum() - m.sum())
        if not np.isfinite(s):
            print(f'    "{label}": only {n} usable samples (need >50 with |F| > {args.f_min:.0f} N while '
                  f'moving, {clipped} dropped at a speed clamp) -- push steadier')
            continue
        print(f'    "{label}" ({1e3 * dd:+.0f} mm): spin ratio |w|/|v| {s:6.2f} rad/m  (n={n}, '
              f'{clipped} clamped samples dropped)  pivot '
              + (f'{1e3 * pz:+.0f} mm' if np.isfinite(pz) else 'none -- pure translation, centre HERE'))
        rows.append((label, dd, s))
    if only is not None or len(rows) < 2:
        return {'s': np.array([r[2] for r in rows])}
    dd, s = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    sols = solve_centre(dd, s)
    pz, fz = args.point[2], flange_position(np.r_[0, 0, 0, 0, 0, 0], tcp)[2]
    print(f'\ncompliance centre (tool z from TCP; configured {1e3 * pz:+.0f} mm, flange {1e3 * fz:+.0f} mm):')
    for i, (c, k, cost) in enumerate(sols):
        print(f'  {"best" if i == 0 else "also"}: {1e3 * c:+6.0f} mm   a_t/a_r = {k:.4f} m^2 '
              f'(expect D_rot/D_trans = {args.d[3] / args.d[0]:.4f})   cost {cost:.3g}')
    if len(sols) > 1 and abs(sols[0][0] - sols[1][0]) > 0.02 and len(rows) < 3:
        print('  NOTE: |d - c| is symmetric; add a third push point to break the tie.')
    c = sols[0][0]
    # centre = point + (q_true - q_assumed): a wrong reference shifts it by the difference
    q_assumed = 0.0 if args.ft_ref == 'tcp' else fz
    q_implied = q_assumed + (c - pz)
    if abs(c - pz) < 0.015:
        print(f'  PASS: centre is within {1e3 * abs(c - pz):.0f} mm of the configured point '
              f'(--ft-ref {args.ft_ref} is consistent)')
    else:
        print(f'  FAIL: centre is {1e3 * (c - pz):+.0f} mm from the configured point. That implies '
              f'getActualTCPForce\'s moment is about {1e3 * q_implied:+.0f} mm (TCP 0, flange '
              f'{1e3 * fz:+.0f}). ' + ('Try --ft-ref flange.' if abs(q_implied - fz) < 0.03 else
                                     'Try --ft-ref tcp.' if abs(q_implied) < 0.03 else
                                     'Matches neither -- check --point and the TCP offset.'))
    return {'s': s, 'c': c, 'k': sols[0][1], 'q_implied': q_implied}


def blocks(d, phase=0):
    """Index arrays of contiguous runs with one (seg, rep) and the given phase."""
    idx = np.flatnonzero(d['phase'] == phase)
    if not len(idx):
        return []
    key = d['seg'][idx] * 1000 + d['rep'][idx]
    cut = np.flatnonzero((np.diff(idx) != 1) | (np.diff(key) != 0)) + 1
    return np.split(idx, cut)


def fit_gain_lag(parts, dt, max_lag=0.3):
    """
    y(t) = g * x(t - L), pooled over parts [(x (N,3), y (N,3), mask (N,))] without
    pairing across part boundaries. Picks L by R^2; returns dict or None.
    """
    best = None
    for n in range(int(max_lag / dt) + 1):
        xs, ys = [], []
        for x, y, m in parts:
            if len(x) <= n + 10:
                continue
            mm = m[:len(x) - n]
            xs.append(x[:len(x) - n][mm])
            ys.append(y[n:][mm])
        if not xs:
            continue
        X, Y = np.concatenate(xs), np.concatenate(ys)
        ok = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
        X, Y = X[ok], Y[ok]
        if len(X) < 100:
            continue
        g = np.sum(X * Y) / np.sum(X * X)
        r2 = 1 - np.sum((Y - g * X) ** 2) / np.sum(Y * Y)
        if best is None or r2 > best['r2']:
            gx = np.sum(X * Y, axis=0) / np.maximum(np.sum(X * X, axis=0), 1e-12)
            best = {'g': g, 'g_axes': gx, 'lag': n * dt, 'r2': r2, 'n': len(X)}
    return best


def analyze_free(d, args, meta):
    """
    End-to-end: measured twist of p vs the measured wrench at p (deadband applied,
    since the controller never sees what is inside it). Then split into our
    controller (commanded twist vs wrench: M/D and the filter) and UR's tracking
    (measured vs commanded twist).
    """
    tcp = np.asarray(meta['tcp_offset'], float)
    pose = d['pose']
    p = point_position(pose, args.point)
    w = shift_wrench(args.ft_sign * d['tcp_force'], ref_point(pose, tcp, args.ft_ref), p)
    w = soft_deadband(w, *args.deadband)
    vm = shift_twist(d['twist'], pose[:, :3], p)
    Rc = frame_mats(pose, args.frame)
    w, vm, vc = (to_frame(x, Rc) for x in (w, vm, d['v_point']))
    dt = float(np.nanmedian(np.diff(d['t'])))
    B = blocks(d, 0)
    out = {}
    for name, sl, thr, unit_s, sc in (('translation', slice(0, 3), args.f_min, 'mm/s per N', 1e3),
                                      ('rotation', slice(3, 6), args.t_min, 'deg/s per Nm', np.degrees(1.0))):
        push = np.linalg.norm(w[:, sl], axis=1) > thr
        moving = np.linalg.norm(vc[:, sl], axis=1) > 0.2 * (args.vmax[sl.start] * 0.1)
        e2e = fit_gain_lag([(w[b, sl], vm[b, sl], push[b]) for b in B], dt)
        ctl = fit_gain_lag([(w[b, sl], vc[b, sl], push[b]) for b in B], dt)
        trk = fit_gain_lag([(vc[b, sl], vm[b, sl], moving[b]) for b in B], dt)
        expect = 1.0 / args.d[sl]
        print(f'\n{name} ({args.frame} frame axes):')
        if e2e is None:
            print(f'  not enough pushed samples (|.| > {thr}) -- push harder or longer')
            out[name] = None
            continue
        print(f'  end to end   {sc * e2e["g"]:6.3f} {unit_s}  per axis {np.round(sc * e2e["g_axes"], 3)}'
              f'  (expect {np.round(sc * expect, 3)})  lag {1e3 * e2e["lag"]:4.0f} ms  R^2 {e2e["r2"]:.3f}')
        if ctl:
            print(f'  controller   {sc * ctl["g"]:6.3f} {unit_s}  lag {1e3 * ctl["lag"]:4.0f} ms  '
                  f'(M/D {1e3 * np.mean(args.m[sl] / args.d[sl]):.0f} ms + filter)')
        if trk:
            print(f'  UR tracking  gain {trk["g"]:5.3f}  lag {1e3 * trk["lag"]:4.0f} ms  R^2 {trk["r2"]:.3f}')
        if name == 'translation':
            print(f'  forceMode    {1e3 * FM_GAIN:6.3f} {unit_s}  lag {1e3 * FM_LAG:4.0f} ms  (baseline)')
            if e2e['g'] < 0:
                print('  NEGATIVE gain: the arm moved against the push. Wrench sign is inverted -- '
                      'rerun with --ft-sign -1.')
        out[name] = {'e2e': e2e, 'controller': ctl, 'tracking': trk}
    return out


def analyze_tap(d, args, meta):
    dirn = unit(args.tap_dir)
    speeds = np.asarray(args.tap_speeds, float)
    rows = []
    print(f'\n  cmd    approach  peak   rise    lost     swings  settle   final')
    print(f'  mm/s   mm/s      N      ms      contact  >2 N    ms       N')
    for seg in range(len(speeds)):
        for rep in range(args.tap_reps):
            m = (d['seg'] == seg) & (d['rep'] == rep) & (d['phase'] <= 1)
            if m.sum() < 20:
                continue
            t = d['t'][m]
            fc = -args.ft_sign * d['tcp_force'][m, :3] @ dirn
            va = d['twist'][m, :3] @ dirn
            hit = np.flatnonzero(fc > args.f_contact)
            if not len(hit):
                print(f'  {speeds[seg]:4.0f}   no contact')
                continue
            ic = hit[0]
            v_app = float(np.median(va[max(0, ic - 50):max(1, ic - 5)]))
            win = (t >= t[ic]) & (t <= t[ic] + 0.5)
            ip = np.flatnonzero(win)[np.argmax(fc[win])]
            peak = fc[ip]
            i3 = np.flatnonzero(fc[:ip + 1] > 3.0)
            rise = t[ip] - t[i3[0]] if len(i3) else np.nan
            after = (t >= t[ip]) & (t <= t[ip] + 0.5)
            lost = float(np.mean(fc[after] < 1.0))
            fs = np.convolve(fc[after], np.ones(5) / 5, mode='valid')
            ext = np.flatnonzero(np.diff(np.sign(np.diff(fs))) != 0) + 1
            vals = fs[np.r_[0, ext, len(fs) - 1]] if len(fs) > 2 else fs
            swings = int(np.sum(np.abs(np.diff(vals)) > 2.0))
            final = float(np.median(fc[t >= t[-1] - 0.2]))
            off = np.flatnonzero(np.abs(fc - final) > max(2.0, 0.2 * abs(final)))
            settle = t[off[-1]] - t[ic] if len(off) else 0.0
            rows.append((speeds[seg], 1e3 * v_app, peak, rise, lost, swings, settle, final))
            print(f'  {speeds[seg]:4.0f}   {1e3 * v_app:6.1f}    {peak:5.1f}  {1e3 * rise:5.0f}   '
                  f'{100 * lost:4.0f} %    {swings:3d}     {1e3 * settle:5.0f}    {final:5.1f}')
    out = {'rows': np.array(rows)}
    if len({r[0] for r in rows}) >= 2:
        A = np.array(rows)
        slope, icpt = np.polyfit(A[:, 1], A[:, 2], 1)
        print(f'\n  peak = {icpt:.1f} N + {slope:.2f} N per mm/s of approach   '
              f'(forceMode: commanded + {FM_TAP_SLOPE:.1f} N per mm/s)')
        print(f'  steady press should be {args.tap_press:.1f} N + the {args.deadband[0]:.1f} N deadband')
        print('  ringing: "swings" counts force reversals > 2 N in the 0.5 s after the peak; '
              '"lost contact" is the time spent below 1 N. More damping (--d) suppresses both.')
        out.update(slope=slope, intercept=icpt)
    return out


def analyze_stall(d, args, meta):
    """
    Speeds are reported only at times the log actually covers: interpolating past
    the last sample repeats it, which once reported a stopped arm as still moving.
    """
    out = {}
    wd = args.watchdog_hz if args.watchdog_hz > 0 else 20.0
    for part, name in ((0, 'A watchdog off'), (1, 'B watchdog on ')):
        m = (d['seg'] == part) & ((d['phase'] == 1) | (d['phase'] == 3))
        if not m.any():
            print(f'  {name}: no gap samples')
            continue
        t = d['t'][m] - d['t'][m][0]
        sp = np.linalg.norm(d['twist'][m, :3], axis=1)
        at = [float(np.interp(x, t, sp)) if x <= t[-1] else np.nan for x in (0.0, 0.05, 0.1, 0.2)]
        stop = np.flatnonzero(d['phase'][m] == 3)
        t_stop = t[stop[0]] if len(stop) else np.nan
        slow = np.flatnonzero(sp < 0.2 * args.stall_speed)
        t_slow = t[slow[0]] if len(slow) else np.nan
        print(f'  {name}: gap logged for {1e3 * t[-1]:.0f} ms; TCP speed at 0/50/100/200 ms: '
              + ' '.join('  -- ' if np.isnan(x) else f'{1e3 * x:4.1f}' for x in at)
              + f' mm/s, last {1e3 * sp[-1]:.1f}')
        print(f'      below 20 % of commanded at {1e3 * t_slow:.0f} ms' if np.isfinite(t_slow) else
              '      never slowed below 20 % of commanded',
              f'; protective stop seen at {1e3 * t_stop:.0f} ms' if np.isfinite(t_stop) else '')
        out[part] = {'speeds': at, 'last': float(sp[-1]), 'covered': float(t[-1]),
                     't_slow': float(t_slow), 't_stop': float(t_stop)}
    if 0 in out:
        a = out[0]
        if a['covered'] < 0.9 * args.stall_gap:
            print('  A: gap not fully logged -- inconclusive.')
        else:
            print('  A: ' + ('the arm KEPT MOVING with no host input -- speedL\'s time argument is not a '
                             'stall guard.' if a['last'] > 0.5 * args.stall_speed else 'the arm stopped by itself.'))
    if 1 in out:
        b = out[1]
        if np.isfinite(b['t_slow']):
            print(f'  B: the watchdog stopped it: below 20 % at {1e3 * b["t_slow"]:.0f} ms after the last input '
                  f'(watchdog period {1e3 / wd:.0f} ms)' + (', via a protective stop' if np.isfinite(b['t_stop']) else ''))
        elif b['covered'] < 0.9 * args.stall_gap:
            print('  B: gap not fully logged -- inconclusive.')
        else:
            print('  B: still moving at the end of the gap -- the watchdog did NOT stop it.')
    return out


ANALYSES = {'center': analyze_center, 'free': analyze_free, 'tap': analyze_tap, 'stall': analyze_stall,
            'ramp': analyze_ramp}


# ----------------------------------------------------------------------- logging

class Edge:
    """Rising-edge detector for DualSense buttons."""

    def __init__(self):
        self.last = {}

    def __call__(self, state, name):
        now = bool(getattr(state, name))
        rose = now and not self.last.get(name, False)
        self.last[name] = now
        return rose


def git_rev():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        rev = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True,
                             text=True, timeout=2).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', os.path.basename(__file__)],
                               cwd=here, capture_output=True, text=True, timeout=2).stdout.strip()
        return rev + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def args_to_json(args):
    return json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in vars(args).items()})


def args_from_json(s):
    raw = json.loads(s)
    for k, v in raw.items():
        if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            raw[k] = np.array(v, float)
        elif k == 'offsets':
            raw[k] = [tuple(x) for x in v]
    return argparse.Namespace(**raw)


def save_log(sess, args, meta, out):
    d = sess.data()
    extra = {}
    for k, v in vars(args).items():         # every argument, one key each, as well as the JSON
        try:
            a = np.asarray(v)
            extra[f'arg_{k}'] = a if a.dtype != object else np.array(json.dumps(v))
        except Exception:
            extra[f'arg_{k}'] = np.array(str(v))
        if v is None:
            extra[f'arg_{k}'] = np.array('None')
    np.savez(out, **d, **meta, args_json=args_to_json(args), **extra)
    return out


def load_log(path):
    z = np.load(path, allow_pickle=False)
    n = len(z['t'])
    d = {name: z[name] if name in z.files else np.full((n, w) if w > 1 else n, np.nan)   # older logs
         for name, w in COLS}
    meta = {k: z[k] for k in z.files if not k.startswith('arg_') and k not in d and k != 'args_json'}
    return d, args_from_json(str(z['args_json'])), meta


def replay(path):
    d, args, meta = load_log(path)
    print(f'{path}: mode {args.mode}, {len(d["t"])} samples, payload {float(meta["payload"]):.3f} kg, '
          f'tcp {np.round(meta["tcp_offset"], 4)}, rev {meta.get("git_rev", "?")}')
    fn = ANALYSES.get(args.mode)
    return fn(d, args, meta) if fn else None


# --------------------------------------------------------------------------- run

def run(args, ctrl, recv, io, clock=time.perf_counter, out=None, make_iface=make_dualsense):
    """Everything after connecting. Shared by the hardware path and the selftest."""
    mass, cog = float(recv.getPayload()), np.array(recv.getPayloadCog(), float)
    tcp = np.array(ctrl.getTCPOffset(), float)
    print(f'payload     : {mass:.3f} kg  cog {np.round(cog, 4)}')
    print(f'TCP offset  : {np.round(tcp, 4)}')
    bad = []
    if abs(mass - args.expect_payload) > 0.02:
        bad.append(f'payload {mass:.3f} kg, expected {args.expect_payload:.3f}')
    if np.linalg.norm(tcp - np.asarray(args.expect_tcp)) > 1e-3:
        bad.append(f'TCP {np.round(tcp, 4)}, expected {np.round(args.expect_tcp, 4)}')
    if bad:
        msg = 'controller settings differ from expected: ' + '; '.join(bad)
        if not args.ignore_mismatch:
            raise SystemExit(msg + '\n(unsaved pendant edit? fix it, or pass --ignore-mismatch)')
        print('WARNING: ' + msg)

    dt = 1.0 / args.hz
    adm = make_admittance(args, tcp, dt)
    print(f'mode {args.mode}: frame {args.frame}  point {np.round(args.point, 4)}  ft-ref {args.ft_ref}  '
          f'sign {args.ft_sign:+.0f}')
    print(f'  K {adm.K}\n  D {adm.D}\n  M {adm.M}\n  sel {adm.sel}')
    print(f'  vmax {args.vmax[[0, 3]]}  amax {args.amax[[0, 3]]}  fmax {args.fmax} N  '
          f'abort {args.f_abort} N  deadband {adm.deadband}')
    sess = Session(ctrl, recv, adm, args, io, clock)
    meta = {'payload': mass, 'payload_cog': cog, 'tcp_offset': tcp, 'git_rev': git_rev(),
            'wall_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'dt': dt,
            'speedl_time': sess.speedl_time, 'K_used': adm.K, 'sel_used': adm.sel,
            'deadband_used': adm.deadband}
    reason = 'finished'
    try:
        if args.mode == 'teleop':          # slow setup: before the watchdog is armed
            print('starting DualSense ...')
            sess.iface = make_iface(args, sess.pose())
        if not args.no_zero and args.mode not in ('center', 'free', 'tap'):   # those zero per segment
            print('zeroing F/T -- arm must be free of contact')
            sess.zero()
        if args.watchdog_hz > 0 and args.mode != 'stall':
            ctrl.setWatchdog(args.watchdog_hz)
            sess.watchdog = True
            print(f'watchdog    : {args.watchdog_hz:.0f} Hz')
        MODES[args.mode](sess, args, io)
    except KeyboardInterrupt:
        reason = 'interrupted'
        print('\nstopping')
    except Abort as ex:
        reason = f'ABORT: {ex}'
        print(f'\n{reason}')
    finally:
        try:
            try:
                sess.ramp_down()           # never speedStop from speed with the watchdog on
            except Exception as ex:
                print(f'ramp down failed: {ex!r}')
            try:
                ctrl.speedStop(args.stop_acc)
            except Exception as ex:
                print(f'speedStop failed: {ex!r}')
        finally:
            ctrl.stopScript()
    meta.update(end_reason=reason, max_loop_gap=sess.max_gap, slow_cycles=sess.slow)
    res = None
    if sess.n:
        out = out or time.strftime(f'fdcc-{args.mode}-%Y%m%d-%H%M%S.npz')
        save_log(sess, args, meta, out)
        d = sess.data()
        print(f'wrote {out}  ({sess.n} samples, {sess.n / max(d["t"][-1], 1e-9):.0f} Hz, '
              f'max loop gap {1e3 * sess.max_gap:.1f} ms, {sess.slow} cycles > 2 dt)')
        fn = ANALYSES.get(args.mode)
        if fn:
            res = fn(d, args, meta)        # after the save: a report bug never costs the data
    return out, res, reason


# -------------------------------------------------------------------- fake robot

class FakeRobot:
    """
    Kinematic stand-in for RTDEControlInterface + RTDEReceiveInterface, for
    --mode selftest. A velocity source: it tracks the commanded TCP twist through
    a first-order lag and speedL's acceleration (UR's joint loops, stiction
    absorbed), re-issues the last speedL until speedStop as ur_rtde's speed_thread
    does, and implements the RTDE watchdog. External forces come from callables
    fn(fake) -> [(F, point, couple)]; the wrench it reports has its moment about
    `sensor_ref` ('tcp' or 'flange') -- set that unlike --ft-ref to see a wrong
    reference.
    """

    def __init__(self, pose, dt, tcp_offset=EXPECT_TCP, sensor_ref='flange', tau=0.02,
                 payload=EXPECT_PAYLOAD, noise=0.0, seed=0, cmd_delay=0.0, m_sense=0.0, sense_delay=0.0):
        self.t, self.dt, self.tau = 0.0, dt, tau
        # Optional measured dynamics (see ur16e()): speedL targets reach the arm
        # cmd_delay late, and the TCP-force estimate reads the arm's OWN acceleration
        # as -m_sense * a, sense_delay late.
        self.cmd_delay, self.m_sense = cmd_delay, m_sense
        self.cmd_hist, self._ci = [], 0
        self.a_hist = deque([np.zeros(3)] * (int(round(sense_delay / 1e-3)) + 1),
                            maxlen=int(round(sense_delay / 1e-3)) + 1)
        self.pose, self.twist, self.cmd = np.array(pose, float), np.zeros(6), np.zeros(6)
        self.tcp, self.sensor_ref, self.payload = np.array(tcp_offset, float), sensor_ref, payload
        self.acc, self.stop_acc = 1.0, 2.0
        self.speeding, self.running, self.wd_hz, self.last_input = False, True, 0.0, 0.0
        self.forces, self.calls, self.speedl_times = [], [], []
        self.zero, self.noise, self.rng = np.zeros(6), noise, np.random.default_rng(seed)
        self.pstop_at, self.push = np.inf, None
        self.wd_action = 'protective'  # what the UR16e does (C207A0); 'stop' = plain program stop

    @classmethod
    def ur16e(cls, pose, dt, **kw):
        """
        Fitted to the 25 Hz limit cycle in fdcc-tap-20260922-165218 (M 10 kg): UR
        tracking ~ unity gain with ~26 ms delay; getActualTCPForce ~ -20 kg x the
        arm's acceleration, ~34 ms late. Loop gain there was 1.55. Only the 25 Hz
        point is measured -- trust it near there, not as a full model.
        """
        # NOT a contact model: with it every M/D chatters on a 40 N/mm wall, while
        # the arm settled cleanly at 5-15 mm/s. Use it for the 25 Hz question only.
        kw = {'tau': 0.004, 'cmd_delay': 0.022, 'm_sense': 20.0, 'sense_delay': 0.034, **kw}
        return cls(pose, dt, **kw)

    def now(self):
        return self.t

    def advance(self, seconds):
        n = max(1, int(round(seconds / 1e-3)))
        for _ in range(n):
            self._step(seconds / n)

    def _step(self, h):
        if h <= 0:
            return
        if self.running and self.wd_hz > 0 and self.t - self.last_input > 1.0 / self.wd_hz:
            self.running, self.speeding = False, False
            self.calls.append('watchdog-stop')
            if self.wd_action == 'protective':
                self.pstop_at, self.stop_acc = self.t, 10.0
        go = self.speeding and self.running
        cmd = self.cmd
        if self.cmd_delay > 0:
            while self._ci + 1 < len(self.cmd_hist) and self.cmd_hist[self._ci + 1][0] <= self.t - self.cmd_delay:
                self._ci += 1
            cmd = (self.cmd_hist[self._ci][1] if self.cmd_hist and self.cmd_hist[self._ci][0] <= self.t - self.cmd_delay
                   else np.zeros(6))
        target, acc = (cmd, self.acc) if go else (np.zeros(6), self.stop_acc)
        dv = clamp_wrench((target - self.twist) * min(1.0, h / self.tau), acc * h,
                          (acc if go else self.stop_acc) * h * (4 if go else 1))
        self.twist = self.twist + dv
        self.a_hist.append(dv[:3] / h)
        self.pose[:3] += self.twist[:3] * h
        self.pose[3:] = (R.from_rotvec(self.twist[3:] * h) * R.from_rotvec(self.pose[3:])).as_rotvec()
        self.t += h

    def _wrench(self):
        q = ref_point(self.pose, self.tcp, self.sensor_ref)
        w = np.zeros(6)
        for fn in self.forces:
            for F, pt, couple in fn(self):
                w[:3] += F
                w[3:] += np.cross(pt - q, F) + couple
        if self.m_sense:
            F = -self.m_sense * self.a_hist[0]            # phantom, applied at the flange
            w[:3] += F
            w[3:] += np.cross(flange_position(self.pose, self.tcp) - q, F)
        return w

    def _input(self, name):
        self.calls.append(name)
        if self.running:
            self.last_input = self.t
        return self.running

    # control interface
    def initPeriod(self):
        return self.t

    def waitPeriod(self, t0):
        self.advance(self.dt)

    def speedL(self, xd, acceleration=0.25, time=0.0):
        if not self._input('speedL'):
            return False
        self.cmd, self.acc, self.speeding = np.array(xd, float), acceleration, True
        if self.cmd_delay > 0:
            self.cmd_hist.append((self.t, self.cmd))
        self.speedl_times.append(time)
        return True

    def speedStop(self, a=10.0):
        if not self._input('speedStop'):
            return False
        self.speeding, self.stop_acc = False, a
        # stopl(a) runs synchronously and ur_rtde waits on it: the host is silent
        # until the arm is at rest (rotation decelerates at a rad/s^2 too).
        self.advance(max(np.linalg.norm(self.twist[:3]), np.linalg.norm(self.twist[3:])) / a)
        return self.running

    def stopScript(self):
        self.calls.append('stopScript')
        self.running = False

    def zeroFtSensor(self):
        self._input('zeroFtSensor')
        self.zero = self._wrench()
        return True

    def setWatchdog(self, min_frequency=10.0):
        self._input('setWatchdog')
        self.wd_hz = min_frequency
        return True

    def kickWatchdog(self):
        return self._input('kick')

    def isProgramRunning(self):
        return self.running

    def getTCPOffset(self):
        return list(self.tcp)

    # receive interface
    def getActualTCPPose(self):
        return list(self.pose)

    def getActualTCPSpeed(self):
        return list(self.twist)

    def getActualTCPForce(self):
        return list(self._wrench() - self.zero + self.noise * self.rng.standard_normal(6) * [1, 1, 1, .05, .05, .05])

    def getFtRawWrench(self):
        return self.getActualTCPForce()

    def getActualQ(self):
        return [0.0] * 6

    def isProtectiveStopped(self):
        return self.t >= self.pstop_at

    def isEmergencyStopped(self):
        return False

    def getPayload(self):
        return self.payload

    def getPayloadCog(self):
        return [0.006, 0.006, 0.048]


class SimIO:
    """Prompts return at once; push windows switch a scripted hand on and off."""

    def __init__(self, fake, hand=None):
        self.fake, self.hand = fake, hand

    def prompt(self, msg, kick, **ctx):
        kick()

    def sleep(self, s):
        self.fake.advance(s)

    def event(self, name, **ctx):
        if self.hand is None:
            return
        if name == 'push_start':
            self.fake.forces = [self.hand(ctx.get('d'))]
        elif name == 'push_end':
            self.fake.forces = []


def hand_at(d, f_tool, couple_fn=None, point=None):
    """Push with force f_tool (tool frame) at tool-z offset d from the TCP, or at `point`."""
    def fn(fk):
        Rt = R.from_rotvec(fk.pose[3:])
        off = np.array([0, 0, d]) if d is not None else np.asarray(point, float)
        F = f_tool(fk.t) if callable(f_tool) else Rt.apply(f_tool)
        c = couple_fn(fk.t) if couple_fn else np.zeros(3)
        return [(F, fk.pose[:3] + Rt.apply(off), c)]
    return fn


def wall(point, n_into, k):
    """Hard surface: pushes the TCP back out along -n_into, k N/m of penetration."""
    point, n_into = np.asarray(point, float), unit(n_into)

    def fn(fk):
        pen = (fk.pose[:3] - point) @ n_into
        return [(-n_into * k * pen, fk.pose[:3].copy(), np.zeros(3))] if pen > 0 else []
    return fn


# ---------------------------------------------------------------------- selftest

POSE0 = [0.45, -0.15, 0.30, 2.6, -1.0, 0.3]      # deliberately not axis-aligned


def selftest():
    """Offline. The maths, the signs, the clamps, the stop order, and every mode's analysis."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix='fdcc-selftest-')
    ap = build_parser()
    results = []

    def check(name, ok, detail=''):
        results.append(bool(ok))
        print(f'  {"PASS" if ok else "FAIL"}  {name:40s} {detail}')

    def args_for(*extra):
        return ap.parse_args(['--no-zero', '--watchdog-hz', '0', '--deadband', '0,0', *extra])

    def quiet(fn, *a, **k):
        buf = _io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                return fn(*a, **k)
        except BaseException:
            print(buf.getvalue())
            raise

    def sim(args, fake, seconds, target_fn=None, vt=np.zeros(6)):
        adm = make_admittance(args, fake.tcp, 1 / args.hz)
        sess = Session(fake, fake, adm, args, SimIO(fake), fake.now)
        target = sess.pose()
        for i in range(int(seconds * args.hz)):
            tg = target if target_fn is None else target_fn(fake.t, target)
            sess.cycle(tg, vt)
        return sess

    rng = np.random.default_rng(1)
    tcp = np.array(EXPECT_TCP)

    print('geometry')
    pose = np.r_[rng.normal(size=3), R.random(random_state=2).as_rotvec()]
    fl = flange_position(pose, tcp)
    zt = R.from_rotvec(pose[3:]).apply([0, 0, 1])
    check('flange is 153.7 mm up tool z', np.allclose(fl, pose[:3] - 0.1537 * zt, atol=1e-12))
    a, b, w, tw = rng.normal(size=3), rng.normal(size=3), rng.normal(size=6), rng.normal(size=6)
    p_a = w[:3] @ tw[:3] + w[3:] @ tw[3:]
    wb, tb = shift_wrench(w, a, b), shift_twist(tw, a, b)
    check('power F.v + tau.w invariant under shift', np.isclose(p_a, wb[:3] @ tb[:3] + wb[3:] @ tb[3:]))
    check('shift_wrench round trip', np.allclose(shift_wrench(wb, b, a), w))
    p = point_position(pose, [0, 0, 0.03])
    F = rng.normal(size=3)
    for ref in ('tcp', 'flange'):
        q = ref_point(pose, tcp, ref)
        adm = Admittance(np.zeros(6), np.ones(6), np.zeros(6), np.ones(6), 'tool', [0, 0, 0.03], tcp,
                         ref, 1.0, np.ones(6), np.ones(6), [0, 0], 0, 1, 1e9, 1e9, 0.002)
        wp, _ = adm.wrench_at_point(pose, np.r_[F, np.cross(p - q, F)])
        check(f'force through p has no moment at p ({ref})', np.allclose(wp[3:], 0, atol=1e-12))
    # vectorised forms agree with the scalar ones
    P = np.stack([pose, np.r_[pose[:3] + 0.1, pose[3:] * 0.5]])
    check('vectorised flange/point', np.allclose(flange_position(P, tcp)[0], fl) and
          np.allclose(point_position(P, [0, 0, .03])[0], p))
    tg = pose.copy()
    tg[:3] += [0.05, 0, 0]
    leash(tg, pose, 0.03, 0.25)
    check('leash pulls a 50 mm target to 30 mm', np.isclose(np.linalg.norm(tg[:3] - pose[:3]), 0.03))

    print('dynamics (fake robot, no noise, no deadband)')
    # 1. pure damper: push +x base at the TCP, K = 0 -> v = F / D along +x
    args = args_for('--mode', 'free')
    fk = FakeRobot(POSE0, 1 / args.hz)
    fk.forces = [hand_at(0.0, lambda t: np.array([6.0, 0, 0]))]
    s = quiet(sim, args, fk, 1.0)
    v = s.data()['twist'][-1]
    check('yield: pushed +x -> moves +x at F/D', np.isclose(v[0], 6e-3, rtol=0.02) and
          np.linalg.norm(v[1:3]) < 1e-5, f'v {np.round(1e3 * v[:3], 3)} mm/s')
    # 2. spring, force THROUGH an off-TCP compliance point -> pure translation, F/K
    args = args_for('--mode', 'teleop', '--k', '2000,20', '--d', '1000,10', '--m', '10,0.1',
                    '--point', '0,0,0.03')
    fk = FakeRobot(POSE0, 1 / args.hz)
    fk.forces = [hand_at(0.03, np.array([8.0, 0, 0]))]
    s = quiet(sim, args, fk, 4.0)
    d = s.data()
    x_tool = R.from_rotvec(POSE0[3:]).apply([1, 0, 0])
    p0, p1 = point_position(d['pose'][0], args.point), point_position(d['pose'][-1], args.point)
    ang = np.linalg.norm(pose_error(d['pose'][-1], d['pose'][0])[3:])
    check('spring: push through p -> F/K, no rotation', np.isclose((p1 - p0) @ x_tool, 4e-3, rtol=0.02)
          and ang < 1e-4, f'{1e3 * (p1 - p0) @ x_tool:.3f} mm (expect 4.000), {1e3 * ang:.3f} mrad')
    # 3. same force at the TCP, 30 mm below p: tau_p = (-0.03 z) x (8 x) = -0.24 Nm about tool y
    fk = FakeRobot(POSE0, 1 / args.hz)
    fk.forces = [hand_at(0.0, np.array([8.0, 0, 0]))]
    s = quiet(sim, args, fk, 4.0)
    d = s.data()
    rot_tool = R.from_rotvec(POSE0[3:]).inv().apply(pose_error(d['pose'][-1], d['pose'][0])[3:])
    check('off-centre push rotates -tool y by tau/K', np.isclose(rot_tool[1], -0.24 / 20, rtol=0.03)
          and abs(rot_tool[0]) + abs(rot_tool[2]) < 1e-4, f'rot (tool) {np.round(1e3 * rot_tool, 3)} mrad '
          f'(expect [0, -12, 0])')
    # 4. selection: lock rotation -> the same off-centre push does not rotate
    args_l = args_for('--mode', 'teleop', '--k', '2000,20', '--d', '1000,10', '--m', '10,0.1',
                      '--point', '0,0,0.03', '--sel', '1,0')
    fk = FakeRobot(POSE0, 1 / args.hz)
    fk.forces = [hand_at(0.0, np.array([8.0, 0, 0]))]
    s = quiet(sim, args_l, fk, 4.0)
    d = s.data()
    ang = np.linalg.norm(pose_error(d['pose'][-1], d['pose'][0])[3:])
    dp = (point_position(d['pose'][-1], args.point) - point_position(d['pose'][0], args.point)) @ x_tool
    check('sel 1,0: rotation held, translation compliant', ang < 2e-4 and np.isclose(dp, 4e-3, rtol=0.05),
          f'{1e3 * ang:.3f} mrad, {1e3 * dp:.3f} mm')
    args_t = args_for('--mode', 'teleop', '--k', '2000,20', '--d', '1000,10', '--m', '10,0.1',
                      '--sel', '0,1')
    fk = FakeRobot(POSE0, 1 / args.hz)
    fk.forces = [hand_at(0.0, np.array([8.0, 0, 0]))]
    d = quiet(sim, args_t, fk, 2.0).data()
    check('sel 0,1: translation held', np.linalg.norm(d['pose'][-1, :3] - d['pose'][0, :3]) < 1e-4)
    # 5. torque sign
    fk = FakeRobot(POSE0, 1 / args.hz)
    zt0 = R.from_rotvec(POSE0[3:]).apply([0, 0, 1])
    fk.forces = [lambda f: [(np.zeros(3), f.pose[:3], 0.2 * zt0)]]
    d = quiet(sim, args, fk, 4.0).data()
    rz = pose_error(d['pose'][-1], d['pose'][0])[3:] @ zt0
    check('torque +tool z -> rotates +tool z by tau/K', np.isclose(rz, 0.2 / 20, rtol=0.03), f'{1e3 * rz:.3f} mrad')
    # 6. moving target with velocity feedforward: no D/K lag
    args_m = args_for('--mode', 'teleop')
    fk = FakeRobot(POSE0, 1 / args_m.hz)
    vt = np.r_[0, 5e-3, 0, 0, 0, 0]
    d = quiet(sim, args_m, fk, 3.0, target_fn=lambda t, t0: np.r_[t0[:3] + vt[:3] * t, t0[3:]], vt=vt).data()
    lag = np.linalg.norm(d['target'][-1, :3] - d['pose'][-1, :3])
    check('moving target tracked (feedforward)', lag < 5e-4,
          f'{1e3 * lag:.2f} mm behind (a plain spring: {1e3 * 5e-3 * 1000 / 300:.1f} mm)')
    # 7. clamps
    args_c = args_for('--mode', 'free')
    fk = FakeRobot(POSE0, 1 / args_c.hz)
    fk.forces = [hand_at(0.0, lambda t: np.array([30.0, 0, 0]) * (t < 1.0) - np.array([30.0, 0, 0]) * (t >= 1.0))]
    s = quiet(sim, args_c, fk, 2.0)
    d = s.data()
    vl = np.linalg.norm(d['v_cmd'][:, :3], axis=1)
    acc = np.linalg.norm(np.diff(d['v_point'][:, :3], axis=0), axis=1) * args_c.hz
    st = np.array(fk.speedl_times)
    check('speed clamp holds at 30 N', vl.max() <= args_c.vmax[0] * (1 + 1e-9), f'max {1e3 * vl.max():.3f} mm/s')
    check('acceleration clamp holds on reversal', acc.max() <= args_c.amax[0] * (1 + 1e-9),
          f'max {acc.max():.3f} m/s^2')
    check('every speedL has 0 < time <= 10 cycles', np.all(st > 0) and np.all(st <= 10 / args_c.hz))

    print('run-level: stop order, aborts, log contents')
    args = args_for('--mode', 'free', '--reps', '1', '--seconds', '1')
    fk = FakeRobot(POSE0, 1 / args.hz, payload=1.67)
    try:
        quiet(run, args, fk, fk, SimIO(fk), fk.now, os.path.join(tmp, 'x.npz'))
        check('payload 1.67 refused', False)
    except SystemExit:
        check('payload 1.67 refused', True)
    args = ap.parse_args(['--mode', 'free', '--reps', '2', '--seconds', '1'])     # real defaults
    fk = FakeRobot(POSE0, 1 / args.hz, noise=0.2)
    fk.pstop_at = 1.5
    out, _, reason = quiet(run, args, fk, fk, SimIO(fk, hand=lambda d: hand_at(0.0, np.array([6.0, 0, 0]))),
                           fk.now, os.path.join(tmp, 'pstop.npz'))
    c = fk.calls
    last = lambda n: max(i for i, x in enumerate(c) if x == n)
    check('protective stop aborts the run', reason.startswith('ABORT: protective'), reason)
    check('speedStop after last speedL, stopScript last', last('speedL') < last('speedStop') < last('stopScript')
          and c[-1] == 'stopScript')
    check('watchdog never tripped while idle/prompting', 'watchdog-stop' not in c)
    z = np.load(out)
    missing = [k for k in vars(args) if f'arg_{k}' not in z.files]
    missing += [k for k in ('payload', 'payload_cog', 'tcp_offset', 'args_json', 'git_rev', 'ft_raw',
                            'tcp_force', 'q', 'v_cmd', 'target', 'twist', 'pose', 't') if k not in z.files]
    check('log has every arg + payload/TCP + all signals', not missing, f'missing {missing}' if missing else '')
    args_hi = ap.parse_args(['--mode', 'free', '--no-zero', '--watchdog-hz', '0', '--reps', '1', '--seconds', '2'])
    fk = FakeRobot(POSE0, 1 / args_hi.hz)
    _, _, reason = quiet(run, args_hi, fk, fk, SimIO(fk, hand=lambda d: hand_at(0.0, np.array([60.0, 0, 0]))),
                         fk.now, os.path.join(tmp, 'fabort.npz'))
    check('60 N aborts the run', reason.startswith('ABORT: wrench'), reason)

    fk = FakeRobot(POSE0, 1 / 500)
    fk.setWatchdog(20.0)
    fk.speedL([0.01, 0, 0, 0, 0, 0.079], 1.0, 0.008)
    fk.advance(0.2)
    fk.speedStop(1.0)
    check('fake: bare speedStop at 0.079 rad/s trips C207A0', 'watchdog-stop' in fk.calls and fk.isProtectiveStopped(),
          '(reproduces fdcc-center-20260922-162200)')
    args = ap.parse_args(['--mode', 'free', '--reps', '2', '--seconds', '2'])
    fk = FakeRobot(POSE0, 1 / args.hz, noise=0.1)
    out, _, reason = quiet(run, args, fk, fk, SimIO(fk, hand=lambda d: hand_at(-0.154, np.array([12.0, 0, 0]))),
                           fk.now, os.path.join(tmp, 'fast-release.npz'))
    wz = np.abs(np.load(out)['v_cmd'][:, 3:]).max()
    check('push ending at speed returns without a watchdog stop', reason == 'finished' and 'watchdog-stop' not in fk.calls,
          f'{reason}, peak rotation {wz:.3f} rad/s')

    class FakePad:
        """DualSense stand-in: slow to build, holds the stick +x, then Ctrl-C."""
        def __init__(self, fk, pose, n):
            fk.advance(3.0)                                   # the slow import + HID open
            self.targ_pose, self.n = np.array(pose, float), n
            self.dualsense = type('D', (), {'state': type('S', (), {k: False for k in
                                  ('DpadUp', 'DpadDown', 'DpadLeft', 'Cross', 'Square')})()})()

        def update(self, h):
            self.n -= 1
            if self.n < 0:
                raise KeyboardInterrupt
            self.targ_pose[0] += 5e-3 * h

    args = ap.parse_args(['--mode', 'teleop'])
    fk = FakeRobot(POSE0, 1 / args.hz, noise=0.1)
    out, _, reason = quiet(run, args, fk, fk, SimIO(fk), fk.now, os.path.join(tmp, 'teleop.npz'),
                           make_iface=lambda a, p: FakePad(fk, p, 200))
    moved = np.load(out)['pose'][-1, 0] - POSE0[0] if out and os.path.exists(out) else np.nan
    check('teleop: slow DualSense setup does not trip the watchdog',
          reason == 'interrupted' and 'watchdog-stop' not in fk.calls and moved > 5e-3,
          f'{reason}, followed the stick {1e3 * moved:.1f} mm (target 10)')

    class WallPad(FakePad):
        """Stick held straight down at 27 mm/s into a 40 N/mm surface 20 mm below."""
        def update(self, h):
            self.n -= 1
            if self.n < 0:
                raise KeyboardInterrupt
            self.targ_pose[2] -= 0.01 * h if self.slow else 0.027 * h

    def wall_teleop(*extra):
        a = ap.parse_args(['--mode', 'teleop', '--vmax', '0.08,0.9', '--teleop-speed', '0.08,0.9', *extra])
        WallPad.slow = '0.01,0.9' in extra
        fk = FakeRobot(POSE0, 1 / a.hz, noise=0.1)
        fk.forces = [wall(np.array(POSE0[:3]) + [0, 0, -0.02], [0, 0, -1], 40000.0)]
        out, _, reason = quiet(run, a, fk, fk, SimIO(fk), fk.now, os.path.join(tmp, 'wall.npz'),
                               make_iface=lambda aa, p: WallPad(fk, p, 800 if WallPad.slow else 400))
        z = np.load(out)
        F, t = z['tcp_force'][:, 2], z['t']
        last = F[t > t[-1] - 1.0]              # slow case: target reaches the leash ~5 s in, 8 s run
        return reason, F.max(), np.median(last), last.std()

    r, pk, st, sd = wall_teleop('--teleop-speed', '0.01,0.9')
    press = 300 * 0.03 + 1.5                           # K * leash + deadband
    check('teleop into a wall at 10 mm/s: contact capped at K*leash+db', r == 'interrupted' and
          abs(st - press) < 3.0 and pk < 20, f'steady {st:.1f} +- {sd:.1f} N, expect {press:.1f} (peak {pk:.1f}) [{r}]')
    r, pk, st, sd = wall_teleop()
    check('teleop into a wall at 27 mm/s: no abort (simple fake)', r == 'interrupted' and pk < 35,
          f'peak {pk:.1f} N, steady {st:.1f} +- {sd:.1f} N [{r}]')
    r, pk, st, sd = wall_teleop('--ff-release', '0,0', '--amax', '0.5,2', '--m', '30,0.6')
    check('old settings reproduce the hardware abort', r.startswith('ABORT: wrench'),
          f'peak {pk:.1f} N [{r[:40]}]  (fdcc-teleop-20260922-164226: 43.6 N then abort)')

    # 25 Hz: the measured sensor/arm loop (FakeRobot.ur16e). A 30 ms shove on top of
    # a steady 5 N push (steady force takes the deadband out of the loop, as the
    # tap's spring did on the arm) must die out at the defaults. M 10 must not: on
    # the arm it held 2.5-2.9 mm/s rms at 25 Hz.
    def kick(*extra):
        a = ap.parse_args(['--mode', 'free', '--reps', '1', '--seconds', '2.5', *extra])
        fk = FakeRobot.ur16e(POSE0, 1 / a.hz, noise=0.1)

        def hand(d):
            t0 = fk.t
            return hand_at(None, lambda t: np.array([0, 5.0 + 15.0 * (t - t0 < 0.03), 0]), point=a.point)
        out, _, reason = quiet(run, a, fk, fk, SimIO(fk, hand=hand), fk.now, os.path.join(tmp, 'kick.npz'))
        z = np.load(out)
        m = (z['phase'] == 0) & (z['t'] > z['t'][z['phase'] == 0][0] + 1.0)
        v = z['v_cmd'][m, 1]
        return reason, 1e3 * (v - v.mean()).std()
    r10, v10 = kick('--m', '10,0.2')
    rd, vd = kick()
    check('25 Hz: M 10 limit-cycles like the arm did', r10 == 'finished' and 1.5 < v10 < 4.0,
          f'{v10:.2f} mm/s rms (arm: 2.5-2.9)')
    check('25 Hz: shove on a 5 N push dies out at default M', rd == 'finished' and vd < 0.5, f'{vd:.2f} mm/s rms')

    print('validation modes on the fake (analysis code paths)')
    for sensor, flag, point, expect in (('tcp', 'tcp', 0.0, 0.0), ('tcp', 'tcp', 0.03, 0.03),
                                        ('flange', 'flange', 0.0, 0.0), ('flange', 'tcp', 0.0, -0.1537)):
        args = ap.parse_args(['--mode', 'center', '--reps', '1', '--seconds', '1.5', '--ft-ref', flag,
                              '--point', f'0,0,{point}'])
        fk = FakeRobot(POSE0, 1 / args.hz, sensor_ref=sensor, noise=0.1)
        out, res, reason = quiet(run, args, fk, fk, SimIO(fk, hand=lambda d: hand_at(d, np.array([6.0, 0, 0]))),
                                 fk.now, os.path.join(tmp, f'center-{sensor}-{flag}-{point}.npz'))
        c = res.get('c', np.nan) if res else np.nan
        check(f'center: sensor@{sensor} flag {flag} point {1e3 * point:+.0f}', abs(c - expect) < 0.01,
              f'fit {1e3 * c:+.1f} mm, expect {1e3 * expect:+.1f}' + ('' if reason == 'finished' else f' [{reason}]'))

    args = ap.parse_args(['--mode', 'free', '--reps', '2', '--seconds', '4'])
    fk = FakeRobot(POSE0, 1 / args.hz, noise=0.1)
    ff = lambda t: 4.0 * np.array([np.sin(2 * np.pi * .6 * t), np.sin(2 * np.pi * .9 * t + 1), np.sin(2 * np.pi * .4 * t + 2)])
    cf = lambda t: 0.3 * np.array([np.sin(2 * np.pi * .5 * t + .5), np.sin(2 * np.pi * .7 * t + 1.5), np.sin(2 * np.pi * .3 * t)])
    out, res, reason = quiet(run, args, fk, fk, SimIO(fk, hand=lambda d: hand_at(None, ff, cf, point=args.point)),
                             fk.now, os.path.join(tmp, 'free.npz'))
    tr, ro = res['translation'], res['rotation']
    check('free: translational gain = 1/D', np.isclose(tr['e2e']['g'], 1 / args.d[0], rtol=0.05),
          f'{1e3 * tr["e2e"]["g"]:.3f} mm/s/N, lag {1e3 * tr["e2e"]["lag"]:.0f} ms, R^2 {tr["e2e"]["r2"]:.3f}')
    check('free: rotational gain = 1/D_rot', np.isclose(ro['e2e']['g'], 1 / args.d[3], rtol=0.05),
          f'{ro["e2e"]["g"]:.4f} rad/s/Nm, lag {1e3 * ro["e2e"]["lag"]:.0f} ms')
    check('free: lags split into controller + tracking',
          abs(tr['controller']['lag'] - args.m[0] / args.d[0]) < 0.015 and abs(tr['tracking']['lag'] - fk.tau) < 0.012,
          f'controller {1e3 * tr["controller"]["lag"]:.0f} ms, tracking {1e3 * tr["tracking"]["lag"]:.0f} ms '
          f'(fake: M/D {1e3 * args.m[0] / args.d[0]:.0f} + filter, tau {1e3 * fk.tau:.0f})')
    rp = quiet(replay, out)
    check('replay reproduces the analysis', np.isclose(rp['translation']['e2e']['g'], tr['e2e']['g']))

    args = ap.parse_args(['--mode', 'tap', '--tap-reps', '1'])
    fk = FakeRobot(POSE0, 1 / args.hz, noise=0.1)
    fk.forces = [wall(np.array(POSE0[:3]) + unit(args.tap_dir) * 0.008, args.tap_dir, 20000.0)]
    out, res, reason = quiet(run, args, fk, fk, SimIO(fk), fk.now, os.path.join(tmp, 'tap.npz'))
    rows = res['rows'] if res else np.zeros((0, 8))
    check('tap: contact at every speed', len(rows) == len(args.tap_speeds), reason)
    check('tap: peak rises with speed', len(rows) > 1 and rows[-1, 2] > rows[0, 2] and res['slope'] > 0,
          f'peaks {np.round(rows[:, 2], 1)} N, slope {res.get("slope", np.nan):.2f} N per mm/s' if len(rows) else '')
    press = args.tap_press + args.deadband[0]       # the deadband adds to the steady contact force
    check('tap: settles to press + deadband', len(rows) and np.allclose(rows[:, 7], press, atol=0.5),
          f'final {np.round(rows[:, 7], 1)} N, expect {press:.1f}' if len(rows) else '')

    for action in ('stop', 'protective'):
        args = ap.parse_args(['--mode', 'stall'])
        fk = FakeRobot(POSE0, 1 / args.hz)
        fk.wd_action = action
        out, res, reason = quiet(run, args, fk, fk, SimIO(fk), fk.now, os.path.join(tmp, f'stall-{action}.npz'))
        ok = (res and res[0]['last'] > 0.9 * args.stall_speed and res[1]['covered'] > 0.9 * args.stall_gap
              and res[1]['last'] < 1e-4 and 0.05 < res[1]['t_slow'] < 0.1 and reason == 'finished')
        check(f'stall ({action}): A moves on, B stops ~1/hz', ok,
              f'B slowed at {1e3 * res[1]["t_slow"]:.0f} ms, logged {1e3 * res[1]["covered"]:.0f} ms [{reason}]'
              if res and 1 in res else reason)

    n_fail = results.count(False)
    print(f'\n{len(results) - n_fail}/{len(results)} passed' + (f', {n_fail} FAILED' if n_fail else '')
          + f'   (logs in {tmp})')
    return 1 if n_fail else 0


# -------------------------------------------------------------------------- main

def main():
    args = build_parser().parse_args()
    if args.replay:
        replay(args.replay)
        return
    if args.mode == 'selftest':
        sys.exit(selftest())

    import rtde_control
    import rtde_receive
    ctrl = rtde_control.RTDEControlInterface(args.ip, args.hz)
    recv = rtde_receive.RTDEReceiveInterface(args.ip)
    run(args, ctrl, recv, HardwareIO(), out=args.out)


if __name__ == '__main__':
    main()
