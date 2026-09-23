"""
Per-cycle Cartesian admittance over UR speedL. Minimal API for Env._control_loop().

    imp = Impedance(ImpedanceParams.from_toml(), tcp_offset=ctrl.getTCPOffset())
    imp.reset()                                  # after any motion not commanded by imp
    ...every 2 ms:
    imp.set_gains(K=..., D=...)                  # optional, per timestep
    v = imp.step(recv.getActualTCPPose(), recv.getActualTCPForce(), pose_target)
    ctrl.speedL(v, imp.p.accel[0], imp.p.dt)     # time argument = ONE cycle

The law is FDCC-ADMITTANCE.md §3, in body coordinates at the compliance frame c:

    M dV/dt + D (V - g V_t) = F + K xi,      xi = log(T_sc^-1 T_sc*)^v

Twists/wrenches are [linear; angular]. Poses in and twists out use UR's conventions:
pose = [p; rotvec] of the TCP in `base`; twist = TCP-point velocity in base axes
(getActualTCPSpeed / speedL); wrench = getActualTCPForce, base axes, moment about the
FLANGE on this arm.

NOT handled here (the caller's job; see FDCC-ADMITTANCE.md §7-8):
  * teleop slew limit and leash. The leash must not DRAG the target after the arm: a
    dragged target's velocity feeds forward, cancels the damping and ran the arm to
    vmax. env.py's clamp() (8 mm / 0.05 rad from actual) drags -- don't feed its output
    here as the target;
  * wrench aborts ([limits] abort_wrench), protective-stop checks, F/T zeroing;
  * stopping: ramp the command to zero through speedL, THEN speedStop. With the RTDE
    watchdog armed, speedStop-from-speed and moveL block the host and trip C207A0;
  * speedL's time argument must be one cycle: 8 ms fed an 8 ms staircase (125 Hz buzz).
"""
from dataclasses import dataclass, field
import os

import numpy as np
from scipy.spatial.transform import Rotation

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, 'fdcc.toml')


def load_config(path=CONFIG_PATH):
    """The whole fdcc.toml as a dict (the caller needs [limits], [rtde], [teleop] too)."""
    import tomllib
    with open(path, 'rb') as f:
        return tomllib.load(f)


def _six(x):
    """scalar, [trans, rot], or 6 values -> 6-vector."""
    x = np.atleast_1d(np.asarray(x, float))
    if x.size == 1:
        return np.repeat(x, 6)
    if x.size == 2:
        return np.repeat(x, 3)
    assert x.size == 6, x
    return x.copy()


@dataclass
class ImpedanceParams:
    dt: float = 0.002
    tcp_offset: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0.1537, 0, 0, 0.]))
    compliance_point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame: str = 'tool'                 # K, D, M, sel on TCP axes; 'base' = fixed base axes
    ft_ref: str = 'flange'              # point getActualTCPForce's moment is about
    ft_sign: float = 1.0
    lowpass_hz: float = 30.0
    deadband: np.ndarray = field(default_factory=lambda: np.array([1.5, 0.15]))
    clamp: np.ndarray = field(default_factory=lambda: np.array([25.0, 2.0]))
    K: np.ndarray = field(default_factory=lambda: _six([300.0, 10.0]))
    D: np.ndarray = field(default_factory=lambda: _six([1000.0, 20.0]))
    M: np.ndarray = field(default_factory=lambda: _six([15.0, 0.6]))
    sel: np.ndarray = field(default_factory=lambda: np.ones(6))
    stiff_gain: float = 5.0
    ff_release: np.ndarray = field(default_factory=lambda: np.array([3.0, 0.3]))
    ff_recover: float = 0.3
    ff_filter_hz: float = 0.0           # low-pass on the force the fade sees; 0 = off
    speed: np.ndarray = field(default_factory=lambda: np.array([0.08, 0.9]))
    accel: np.ndarray = field(default_factory=lambda: np.array([2.0, 4.0]))

    @classmethod
    def from_toml(cls, path=CONFIG_PATH):
        return cls.from_config(load_config(path))

    @classmethod
    def from_config(cls, c):
        w, a, ff, lim = c['wrench'], c['admittance'], c['feedforward'], c['limits']
        return cls(dt=1.0 / c['robot']['rate_hz'], tcp_offset=np.array(c['robot']['tcp_offset'], float),
                   compliance_point=np.array(a['compliance_point'], float), frame=a['frame'],
                   ft_ref=w['reference_point'], ft_sign=float(w['sign']), lowpass_hz=w['lowpass_hz'],
                   deadband=np.array(w['deadband'], float), clamp=np.array(w['clamp'], float),
                   K=_six(a['stiffness']), D=_six(a['damping']), M=_six(a['mass']), sel=_six(a['selection']),
                   stiff_gain=a['stiff_gain'], ff_release=np.array(ff['release'], float),
                   ff_recover=ff['recover_s'], ff_filter_hz=ff.get('filter_hz', 0.0),
                   speed=np.array(lim['speed'], float),
                   accel=np.array(lim['accel'], float))


# ------------------------------------------------------------------- SE(3) helpers

def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _T(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def _rotvec_to_R(w):
    """Rodrigues. Closed form: scipy's Rotation costs ~50 us per call at 500 Hz."""
    th = np.sqrt(w @ w)
    W = _skew(w)
    if th < 1e-8:
        return np.eye(3) + W
    return np.eye(3) + np.sin(th) / th * W + (1 - np.cos(th)) / th ** 2 * W @ W


def _R_to_rotvec(Rm):
    c = np.clip((np.trace(Rm) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th > np.pi - 1e-3:                    # near pi the closed form loses precision
        return Rotation.from_matrix(Rm).as_rotvec()
    a = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    return a * (0.5 if th < 1e-8 else th / (2 * np.sin(th)))


def pose_to_T(pose):
    """UR [p; rotvec] -> 4x4."""
    pose = np.asarray(pose, float)
    return _T(_rotvec_to_R(pose[3:]), pose[:3])


def _inv(T):
    R, p = T[:3, :3], T[:3, 3]
    return _T(R.T, -R.T @ p)


def adjoint(T):
    """Ad_T for [v; w] ordering: V_a = Ad_{T_ab} V_b, F_b = Ad_{T_ab}^T F_a."""
    R, p = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3], A[:3, 3:], A[3:, 3:] = R, _skew(p) @ R, R
    return A


def _B(R):
    Z = np.zeros((6, 6))
    Z[:3, :3] = Z[3:, 3:] = R
    return Z


def se3_log(T):
    """log(T)^v = [v; w], with v = V^-1(w) p."""
    w = _R_to_rotvec(T[:3, :3])
    th = np.linalg.norm(w)
    W = _skew(w)
    if th < 1e-6:
        c = 1.0 / 12.0
    else:
        c = (1.0 - th * np.sin(th) / (2.0 * (1.0 - np.cos(th)))) / th ** 2
    Vinv = np.eye(3) - 0.5 * W + c * W @ W
    return np.r_[Vinv @ T[:3, 3], w]


def se3_exp(xi):
    v, w = np.asarray(xi[:3], float), np.asarray(xi[3:], float)
    th = np.linalg.norm(w)
    W = _skew(w)
    if th < 1e-6:
        V = np.eye(3) + 0.5 * W
    else:
        V = np.eye(3) + (1 - np.cos(th)) / th ** 2 * W + (th - np.sin(th)) / th ** 3 * W @ W
    return _T(_rotvec_to_R(w), V @ v)


def _clamp_halves(x, lin, ang):
    x = np.array(x, float)
    for sl, lim in ((slice(0, 3), lin), (slice(3, 6), ang)):
        n = np.linalg.norm(x[sl])
        if n > lim:
            x[sl] *= lim / n
    return x


def _deadband_halves(x, fdb, tdb):
    x = np.array(x, float)
    for sl, db in ((slice(0, 3), fdb), (slice(3, 6), tdb)):
        n = np.linalg.norm(x[sl])
        x[sl] *= max(n - db, 0.0) / n if n > 0 else 0.0
    return x


def leash_step(prev, des, actual, radius, max_step):
    """
    Non-dragging leash: move the leashed target `prev` toward `des`, at most `max_step`
    ([m, rad] per call), without ending further from `actual` than
    max(radius, where prev already is) -- position and rotation separately.

    It never moves the target toward the arm unless `des` lies that way, so pushing the
    arm by hand does not move the equilibrium, and a held target has no velocity to feed
    forward. It always chases `des`, so absolute targets (scripted moves, policies) are
    reached exactly. Poses are UR [p; rotvec]. Returns (new_pose, (lin_held, ang_held)).
    """
    prev, des, actual = (np.asarray(x, float) for x in (prev, des, actual))
    out, held = prev.copy(), [False, False]

    # position: largest s in [0, 1] with |u + s w| <= r
    w = _clamp_halves(np.r_[des[:3] - prev[:3], 0, 0, 0], max_step[0], 1)[:3]
    u = prev[:3] - actual[:3]
    r = max(radius[0], np.linalg.norm(u))
    if np.linalg.norm(u + w) > r:
        ww, uw = w @ w, u @ w
        s = (-uw + np.sqrt(max(uw * uw - ww * (u @ u - r * r), 0.0))) / ww if ww > 0 else 0.0
        w, held[0] = np.clip(s, 0.0, 1.0) * w, True
    out[:3] = prev[:3] + w

    # rotation: along the geodesic prev -> des, bisect for the same condition on the angle
    Rp, Ra = _rotvec_to_R(prev[3:]), _rotvec_to_R(actual[3:])
    d = _R_to_rotvec(_rotvec_to_R(des[3:]) @ Rp.T)
    th = np.linalg.norm(d)
    if th > max_step[1]:
        d *= max_step[1] / th
    ang = lambda s: np.linalg.norm(_R_to_rotvec(_rotvec_to_R(s * d) @ Rp @ Ra.T))
    r = max(radius[1], ang(0.0))
    s = 1.0
    if ang(1.0) > r:
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ang(mid) <= r else (lo, mid)
        s, held[1] = lo, True
    out[3:] = _R_to_rotvec(_rotvec_to_R(s * d) @ Rp)
    return out, tuple(held)


# ---------------------------------------------------------------------- controller

class Impedance:
    def __init__(self, params=None, tcp_offset=None):
        self.p = params or ImpedanceParams()
        off = np.asarray(self.p.tcp_offset if tcp_offset is None else tcp_offset, float)
        self.T_fe = pose_to_T(off)                                   # flange -> TCP
        self.T_ec = _T(np.eye(3), self.p.compliance_point)           # TCP -> compliance frame
        self.Ad_ec = adjoint(self.T_ec)
        self.Ad_ec_inv = adjoint(_inv(self.T_ec))
        lp = lambda hz: 1.0 if hz <= 0 else 1.0 - np.exp(-2 * np.pi * hz * self.p.dt)
        self.alpha = lp(self.p.lowpass_hz)
        self.alpha_ff = lp(self.p.ff_filter_hz)
        self.K, self.D, self.M, self.sel = (_six(x) for x in (self.p.K, self.p.D, self.p.M, self.p.sel))
        self._check()
        self.reset()

    def reset(self):
        """Zero the state. Call after any motion this object did not command."""
        self.V = np.zeros(6)            # commanded twist of c, body coordinates
        self.F_filt = None
        self.F_ff = None                # slow copy of F for the fade
        self.g = np.ones(2)             # feedforward fade, [lin, ang]
        self.T_target_prev = None
        self.last = {}

    def set_gains(self, K=None, D=None, M=None, sel=None):
        """
        Replace gains from the next step on. Each: scalar, [trans, rot] or 6 values;
        None keeps the current value. Axes are those of `frame`. Scaling K by s keeps
        the damping ratio if D is scaled by sqrt(s) -- that is the caller's choice.
        """
        new = [self.K, self.D, self.M, self.sel]
        for i, x in enumerate((K, D, M, sel)):
            if x is not None:
                new[i] = _six(x)
        old = self.K, self.D, self.M, self.sel
        self.K, self.D, self.M, self.sel = new
        try:
            self._check()
        except ValueError:
            self.K, self.D, self.M, self.sel = old
            raise

    def _check(self):
        bad = (self.sel != 0) & (self.M + self.D * self.p.dt <= 0)
        if bad.any():
            raise ValueError(f'compliant axes {np.flatnonzero(bad)} need M + D dt > 0')

    def step(self, pose, wrench, pose_target, twist_target=None):
        """
        pose, pose_target: TCP [p; rotvec] in base. wrench: raw getActualTCPForce().
        twist_target: the target's TCP twist, speedL convention; None = finite
        difference of successive pose_targets. Returns the twist for speedL.
        """
        p, dt = self.p, self.p.dt
        T_se, T_st = pose_to_T(pose), pose_to_T(pose_target)
        T_sc, T_sct = T_se @ self.T_ec, T_st @ self.T_ec
        R_sc = T_sc[:3, :3]

        # 1. wrench: body at c, filtered, deadbanded, clamped
        T_sf = T_se @ _inv(self.T_fe)
        T_q = T_sf if p.ft_ref == 'flange' else T_se
        F_q = _B(T_q[:3, :3]).T @ (p.ft_sign * np.asarray(wrench, float))
        F = adjoint(_inv(T_q) @ T_sc).T @ F_q
        self.F_filt = F if self.F_filt is None else self.F_filt + self.alpha * (F - self.F_filt)
        F = _clamp_halves(_deadband_halves(self.F_filt, *p.deadband), *p.clamp)
        self.F_ff = F if self.F_ff is None else self.F_ff + self.alpha_ff * (F - self.F_ff)

        # 2. error and target twist, body at c
        T_cct = _inv(T_sc) @ T_sct
        xi = se3_log(T_cct)
        if twist_target is None:
            Vb_et = (np.zeros(6) if self.T_target_prev is None
                     else se3_log(_inv(self.T_target_prev) @ T_st) / dt)
            Vb_et = _clamp_halves(Vb_et, *p.speed)
        else:
            Vb_et = _B(T_st[:3, :3]).T @ np.asarray(twist_target, float)
        self.T_target_prev = T_st
        Vt = adjoint(T_cct) @ (self.Ad_ec_inv @ Vb_et)

        # compliance coordinates: body (tool axes) or base axes at c
        Q = np.eye(6) if p.frame == 'tool' else _B(R_sc)
        Fq, xq, Vtq, Vq = Q @ F, Q @ xi, Q @ Vt, Q @ self.V
        Fff = Q @ self.F_ff

        # 3. feedforward fade: instant drop, slow recovery. It reads a slow copy of F:
        # an impact kicked the ~28 Hz mode, its force wobble (4 N rms) kept tripping
        # the fade, the lost feedforward left a 10-40 mm lag, and that steady spring
        # force held the deadband open so the mode kept going
        # (logs-debug-fdcc/episode000001, 2026-09-23).
        Vff = Vtq.copy()
        for j, (sl, rel) in enumerate(((slice(0, 3), p.ff_release[0]), (slice(3, 6), p.ff_release[1]))):
            n = np.linalg.norm(Vff[sl])
            g = 1.0
            if rel > 0 and n > 0:
                g = float(np.clip(1.0 + (Fff[sl] @ Vff[sl]) / (n * rel), 0.0, 1.0))
            if p.ff_recover > 0:
                g = min(g, self.g[j] + dt / p.ff_recover)
            self.g[j] = g
            Vff[sl] *= g

        # 4. dynamics, implicit in the damping
        Vn = (self.M * Vq + dt * (Fq + self.K * xq + self.D * Vff)) / (self.M + self.D * dt)
        stiff = self.sel == 0
        Vn[stiff] = Vtq[stiff] + p.stiff_gain * xq[stiff]
        Vn = Q.T @ Vn

        # 5. clamps (state too), then to the TCP in speedL's convention
        Vn = _clamp_halves(Vn, *p.speed)
        self.V = self.V + _clamp_halves(Vn - self.V, p.accel[0] * dt, p.accel[1] * dt)
        v_tcp = _clamp_halves(_B(T_se[:3, :3]) @ (self.Ad_ec @ self.V), *p.speed)

        self.last = {'F_c': F, 'xi': xi, 'V': self.V.copy(), 'ff_gain': self.g.copy(), 'V_t': Vt}
        return v_tcp
