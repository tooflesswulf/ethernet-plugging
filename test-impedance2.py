#!/usr/bin/env python3
"""
Chirp system-identification for the impedance controller.

Everything so far has been tuned from offline models, and those models have been
wrong more than once (task inertia measured at tool0 rather than the TCP; a
discrete stability bound that says the old damping diverges when in practice it
only wiggled). This measures the actual plant instead.

Two experiments:

  chirp --mode setpoint   sweep the EQUILIBRIUM pose with the impedance loop
                          running. Gives the closed-loop response eq -> actual:
                          where it rings, how damped it really is, where it goes
                          unstable.

  chirp --mode wrench     hold position with a low stiffness and inject a chirp
                          WRENCH on top. Gives the plant response F -> accel,
                          i.e. whether the task-inertia model is right at all.

Then:

  analyze <file.npz>      frequency response, resonances, damping estimates.

Logs t, q, qd, actual_pose, twist, raw force, equilibrium pose, commanded wrench
and commanded torque every tick into preallocated arrays -- no allocation in the
control loop, because allocation pressure is one of the things that makes ticks
run late.

Amplitudes are deliberately small. Keep a hand on the e-stop.
"""
from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse
import time

from impedance import CartesianImpedance, saturate_direction, pose_error

# rtde_* and kinematics (pinocchio) are imported lazily inside cmd_chirp, so
# `analyze` runs on a machine that has neither the robot nor pinocchio.

ROBOT_IP = "192.168.0.100"
AXES = ['x', 'y', 'z', 'rx', 'ry', 'rz']


# ======================================================================
# connection
# ======================================================================
def make_torque_fn(ctrl):
    """1.6.3 takes friction_comp: bool, 1.6.5 takes per-joint scale vectors."""
    doc = ctrl.directTorque.__doc__ or ''
    if 'viscous' in doc:
        print('directTorque : 1.6.5 scale-vector form (its script drops the scales)')
        return ctrl.directTorque
    print('directTorque : friction_comp=True (1.6.3 form)')
    return lambda t: ctrl.directTorque(t, True)


def connect():
    import rtde_control
    import rtde_receive
    # No FLAG_UPPER_RANGE_REGISTERS -- it hangs construction on this controller.
    ctrl = rtde_control.RTDEControlInterface(ROBOT_IP)
    recv = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    tcp = ctrl.getTCPOffset()
    dt = ctrl.getStepTime()
    if dt <= 0:
        print(f'getStepTime() returned {dt}; assuming 0.002 s')
        dt = 0.002
    print(f'TCP offset : {np.round(tcp, 5)}')
    print(f'payload    : {recv.getPayload()} kg  cog {np.round(recv.getPayloadCog(), 4)}')
    print(f'dt         : {dt} s')
    return ctrl, recv, tcp, dt


# ======================================================================
# chirp
# ======================================================================
def log_chirp_phase(t, f0, f1, T):
    """
    Exponential sweep. Logarithmic spacing spends comparable time per octave,
    which is what you want when the interesting dynamics span 1-100 Hz.
    """
    k = (f1 / f0) ** (t / T)
    return 2 * np.pi * f0 * T / np.log(f1 / f0) * (k - 1.0)


def envelope(t, T, ramp=0.5):
    """Fade in/out so the sweep does not start or stop with a step."""
    a = min(t / ramp, 1.0) if t < ramp else 1.0
    b = min((T - t) / ramp, 1.0) if t > T - ramp else 1.0
    return min(a, b)


def amp_at(f, amp, v_max, axis):
    """
    Displacement amplitude, tapered so peak velocity stays under v_max.

    A constant-displacement sweep is unusable across a wide band: the amplitude
    needed to push past this arm's ~5 N friction floor (13 mm at K=1500) implies
    2*pi*40*0.013 = 3 m/s at the top of a 40 Hz sweep. Tapering as v_max/(2 pi f)
    keeps it safe -- but note the consequence: above a few Hz the commanded force
    drops back under the friction floor, so the honestly measurable band on this
    arm is low frequency. That is where the closed-loop poles are anyway
    (w_n ~ 11 rad/s = 1.8 Hz).
    """
    if axis >= 3:
        return amp                      # rotations: rad, velocity limit n/a here
    return min(amp, v_max / max(2 * np.pi * f, 1e-9))


def offset_pose(base, axis, delta):
    """Apply a scalar displacement along one of the 6 task axes."""
    p = np.array(base, float)
    if axis < 3:
        p[axis] += delta
        return p
    rv = np.zeros(3)
    rv[axis - 3] = delta
    p[3:] = (R.from_rotvec(rv) * R.from_rotvec(p[3:])).as_rotvec()
    return p


def cmd_chirp(args):
    from kinematics import URKin
    ctrl, recv, tcp, dt = connect()
    kin = URKin(tcp)
    imp = CartesianImpedance(tau_rated=kin.tau_rated)

    # Apparent inertia is payload + residual, not the arm's task inertia -- the
    # firmware compensates its own dynamics. --inertia overrides it for
    # experiments that want to probe a different value.
    payload = recv.getPayload()
    if args.inertia:
        ref = np.array([float(v) for v in args.inertia.split(',')])
        print(f'inertia    : OVERRIDE {np.round(ref, 3)}')
    else:
        ref = imp.effective_inertia(payload, tcp)
        print(f'inertia    : {np.round(ref, 3)}  (payload {payload:.3f} kg + residual)')
    imp.calibrate(ref, zeta=args.zeta)
    if args.fc_nm:
        imp.f_c = np.array([float(x) for x in args.fc_nm.split(',')])
    if args.no_friction:
        imp.f_c = np.zeros(6)

    axis = AXES.index(args.axis)
    mode = args.mode
    if args.amp is None:
        # Different units AND different gains: 0.015 is 15 mm (22 N at K=1500,
        # well clear of the 5 N floor) but only 0.015 rad (0.75 Nm at K_rot=50,
        # BELOW the ~0.9 Nm rotational floor). One default cannot serve both.
        args.amp = 0.015 if axis < 3 else 0.15
    if mode == 'wrench':
        # Hold position weakly so it cannot drift, and inject the wrench on top.
        k = args.hold_k
        imp.K_free = np.array([k, k, k, k / 20, k / 20, k / 20], float)
        imp.calibrate(ref, zeta=args.zeta)

    K, D = imp.gains(0.0)
    print(f'\nmode       : {mode}')
    print(f'axis       : {args.axis}')
    print(f'amplitude  : {args.amp}{" m" if axis < 3 else " rad" if mode == "setpoint" else " N/Nm"}')
    print(f'sweep      : {args.f0} -> {args.f1} Hz over {args.duration} s')
    print(f'K          : {np.round(K, 1)}')
    print(f'D          : {np.round(D, 1)}')
    print(f'f_c        : {np.round(imp.f_c, 2)}')
    if mode == 'setpoint':
        floor = args.friction_floor if axis < 3 else args.friction_floor_rot
        unit = 'N' if axis < 3 else 'Nm'
        f_lo = K[axis] * amp_at(args.f0, args.amp, args.v_max, axis)
        f_hi = K[axis] * amp_at(args.f1, args.amp, args.v_max, axis)
        print(f'drive      : {f_lo:.2f} {unit} at {args.f0} Hz -> {f_hi:.2f} {unit} at {args.f1} Hz'
              f'   (friction floor ~{floor} {unit})')
        if f_lo < 3 * floor:
            print(f'\n!! drive {f_lo:.2f} {unit} is not >> the ~{floor} {unit} friction floor.'
                  f'\n!! The sweep will sit inside the deadband and measure stiction,'
                  f'\n!! not dynamics. Raise --amp to at least '
                  f'{3*floor/K[axis]:.3f} ({"m" if axis < 3 else "rad"}).\n')
        if f_hi < floor:
            print(f'   note: the velocity taper pushes the drive under the friction floor'
                  f'\n   toward the top of the sweep; data up there is not meaningful.')
        if axis >= 3 and args.amp > 0.2:
            print(f'   note: amp {args.amp} rad exceeds MAX_ORIENTATION_ERROR (0.2), so the'
                  f'\n   error clamps and the drive flattens. Keep amp <= 0.2 rad.')

    n = int(args.duration / dt) + 100
    L = {k: np.zeros((n, d)) for k, d in (
        ('q', 6), ('qd', 6), ('pose', 6), ('twist', 6), ('force', 6),
        ('eq', 6), ('wrench', 6), ('tau', 6))}
    L['t'] = np.zeros(n)
    L['drive'] = np.zeros(n)          # the injected signal itself
    L['dt'] = np.zeros(n)

    torque_cmd = make_torque_fn(ctrl)
    base = np.array(recv.getActualTCPPose())
    print(f'base pose  : {np.round(base, 4)}')
    input('\nenter to start, ctrl-C to abort: ')
    ctrl.setWatchdog(args.watchdog)      # arm AFTER the prompt, never before

    i = 0
    abort = None
    t0 = time.perf_counter()
    t_prev = t0
    try:
        while True:
            ts = ctrl.initPeriod()
            now = time.perf_counter()
            t = now - t0
            if t > args.duration:
                break

            q = np.array(recv.getActualQ())
            qd = np.array(recv.getActualQd())
            pose = np.array(recv.getActualTCPPose())
            twist = np.array(recv.getActualTCPSpeed())
            force = np.array(recv.getActualTCPForce())

            f_now = args.f0 * (args.f1 / args.f0) ** (t / args.duration)
            a_now = amp_at(f_now, args.amp, args.v_max, axis)
            drive = (a_now * envelope(t, args.duration, args.ramp)
                     * np.sin(log_chirp_phase(t, args.f0, args.f1, args.duration)))

            if mode == 'setpoint':
                eq = offset_pose(base, axis, drive)
                ff = np.zeros(6)
            else:
                eq = base
                ff = np.zeros(6)
                ff[axis] = drive

            J = kin.jacobian(q)
            tau, F, _ = imp.compute(q, qd, pose, eq, twist, J)
            if mode == 'wrench':
                F = saturate_direction(F + ff, imp.F_sat)
                tau = saturate_direction(J.T @ F - imp.d_q * qd, imp.tau_sat)

            # ---- abort checks (raw signals; the filtered force is far too slow)
            if np.max(np.abs(force[:3])) > args.force_max:
                abort = f'force {np.max(np.abs(force[:3])):.1f} N'
            elif np.linalg.norm(twist[:3]) > args.speed_max:
                abort = f'speed {np.linalg.norm(twist[:3]):.2f} m/s'
            elif np.max(np.abs(pose[:3] - base[:3])) > args.excursion:
                abort = f'excursion {np.max(np.abs(pose[:3] - base[:3]))*1000:.0f} mm'
            elif not np.all(np.isfinite(tau)):
                abort = 'non-finite torque'
            if abort:
                break

            for k, v in (('q', q), ('qd', qd), ('pose', pose), ('twist', twist),
                         ('force', force), ('eq', eq), ('wrench', F), ('tau', tau)):
                L[k][i] = v
            L['t'][i] = t
            L['drive'][i] = drive
            L['dt'][i] = now - t_prev
            t_prev = now
            i += 1

            torque_cmd(tau.tolist())
            if i % 250 == 0:
                print(f'  t={t:5.1f}s  f={f_now:6.2f}Hz  '
                      f'|dev|={np.max(np.abs(pose[:3]-base[:3]))*1000:5.1f}mm  '
                      f'|F|={np.linalg.norm(force[:3]):5.1f}N', end='\r')
            ctrl.waitPeriod(ts)
    except KeyboardInterrupt:
        abort = 'interrupted'
    finally:
        for _ in range(5):
            torque_cmd([0.0] * 6)
        ctrl.stopJ(2.0)
        ctrl.stopScript()

    print(f'\n{"ABORTED: " + abort if abort else "complete"}   {i} samples')
    if i < 100:
        print('too few samples to be useful')
        return

    out = {k: v[:i] for k, v in L.items()}
    out.update(dict(dt_nominal=dt, axis=axis, mode=mode, f0=args.f0, f1=args.f1,
                    duration=args.duration, amp=args.amp, base=base, ramp=args.ramp,
                    K=K, D=D, f_c=imp.f_c, ref_inertia=ref, tcp=tcp,
                    aborted=abort or ''))
    np.savez_compressed(args.out, **out)
    late = np.sum(out['dt'][1:] > 3 * dt)
    print(f'rate {i/out["t"][-1]:.0f} Hz, {late} late ticks '
          f'(worst {out["dt"][1:].max()*1000:.1f} ms)')
    print(f'saved {args.out}')
    print(f'\n  python test-impedance2.py analyze {args.out}')


# ======================================================================
# analyze
# ======================================================================
def cmd_analyze(args):
    from scipy import signal
    d = np.load(args.file, allow_pickle=True)
    axis = int(d['axis'])
    mode = str(d['mode'])
    fs = 1.0 / float(d['dt_nominal'])
    name = AXES[axis]
    print(f'{args.file}: mode={mode} axis={name} '
          f'{float(d["f0"])}-{float(d["f1"])} Hz, {len(d["t"])} samples @ {fs:.0f} Hz')
    if str(d['aborted']):
        print(f'  NOTE: run aborted ({d["aborted"]}) -- the sweep is incomplete')
    print(f'  K = {np.round(d["K"], 1)}')
    print(f'  D = {np.round(d["D"], 1)}')

    # Exclude the fade windows. On a LOG sweep the fade-out spans a huge
    # frequency range at the top (1 s of a 0.2-40 Hz sweep covers 33-40 Hz), so
    # including it manufactures a fake resonance right at f1.
    ramp = float(d['ramp']) if 'ramp' in d.files else 1.0
    T = float(d['duration'])
    keep = (d['t'] > ramp * 1.2) & (d['t'] < T - ramp * 1.5)
    if keep.sum() < 500:
        keep = np.ones(len(d['t']), bool)
    f_top = float(d['f0']) * (float(d['f1']) / float(d['f0'])) ** (d['t'][keep].max() / T)
    print(f'  usable window: {d["t"][keep].min():.1f}-{d["t"][keep].max():.1f} s '
          f'-> up to {f_top:.1f} Hz (fade windows excluded)')

    if mode == 'setpoint':
        u = d['eq'][:, axis] - d['base'][axis] if axis < 3 else d['drive']
        y = (d['pose'][:, axis] - d['base'][axis]) if axis < 3 else None
        if y is None:
            # rotation: project the orientation error onto the driven axis
            # pose_error(base, pose) already gives the rotation of pose wrt base.
            # Negating it here inverted every rotational DC gain.
            y = np.array([pose_error(d['base'], p)[3:][axis - 3] for p in d['pose']])
        label = 'eq -> actual  (closed loop)'
        u, y = u[keep], y[keep]
    else:
        u = d['wrench'][:, axis]
        v = d['twist'][:, axis]
        y = np.gradient(v, d['t'])           # acceleration
        label = 'wrench -> accel  (plant)'
        u, y = u[keep], y[keep]

    nper = min(4096, len(u) // 4)
    f, Puu = signal.welch(u, fs, nperseg=nper)
    _, Puy = signal.csd(u, y, fs, nperseg=nper)
    H = Puy / np.maximum(Puu, 1e-30)
    _, Pyy = signal.welch(y, fs, nperseg=nper)
    coh = np.abs(Puy) ** 2 / np.maximum(Puu * Pyy, 1e-30)

    band = (f >= float(d['f0'])) & (f <= f_top) & (coh > args.min_coh)
    if not band.any():
        print('  no frequency bins with usable coherence; longer run or larger amplitude')
        return

    print(f'\n{label}      (bins with coherence > {args.min_coh})')
    print(f'{"f [Hz]":>8} {"|H|":>12} {"dB":>8} {"phase":>8} {"coh":>6}')
    fb, Hb, cb = f[band], H[band], coh[band]
    mag = np.abs(Hb)
    for k in np.unique(np.linspace(0, len(fb) - 1, min(18, len(fb))).astype(int)):
        print(f'{fb[k]:8.2f} {mag[k]:12.4g} {20*np.log10(max(mag[k],1e-12)):8.1f} '
              f'{np.degrees(np.angle(Hb[k])):8.0f} {cb[k]:6.2f}')

    pk = np.argmax(mag)
    print(f'\npeak |H| at {fb[pk]:.2f} Hz  ({fb[pk]*2*np.pi:.1f} rad/s)')

    if mode == 'setpoint' and cb.mean() > 0.6:
        # Fit a real second-order model rather than eyeballing a Q factor.
        # Bounded, because an unbounded fit on a response with no resonant peak
        # runs away (one axis returned zeta=361, inertia=3e5).
        from scipy import optimize
        w = 2 * np.pi * fb
        def mdl(p_, w_):
            s_ = 1j * w_
            return p_[0] * p_[1] ** 2 / (s_ ** 2 + 2 * p_[2] * p_[1] * s_ + p_[1] ** 2)
        dc0 = float(np.mean(mag[:max(1, len(mag) // 20)]))
        # seed w_n where |H| falls to dc0/sqrt(2) -- works with or without a peak
        below = np.nonzero(mag < dc0 / np.sqrt(2))[0]
        w0 = 2 * np.pi * (fb[below[0]] if len(below) else fb[pk])
        r = optimize.least_squares(
            lambda p_: np.r_[(mdl(p_, w) - Hb).real, (mdl(p_, w) - Hb).imag],
            [max(dc0, 1e-3), w0, 0.5],
            bounds=([1e-4, 2 * np.pi * fb[0], 0.02],
                    [5.0, 2 * np.pi * fb[-1], 3.0]))
        g_, wn_, z_ = r.x[0], abs(r.x[1]), abs(r.x[2])

        resonant = mag.max() > 1.05 * dc0
        at_bound = (wn_ < 2 * np.pi * fb[0] * 1.05) or (wn_ > 2 * np.pi * fb[-1] * 0.95)
        if at_bound or not resonant:
            print(f'\n  ** fit is NOT reliable for this axis: '
                  f'{"no resonant peak in band" if not resonant else "w_n pinned at a band edge"}.')
            print(f'  ** |H| falls monotonically from DC, so w_n and zeta are not')
            print(f'  ** separately identifiable here. Treat the numbers below as a')
            print(f'  ** lower bound on w_n, and widen the sweep to see the corner.')
        Kx = float(d['K'][axis]); Dx = float(d['D'][axis])
        Im = float(d['ref_inertia'][axis]); Ie = Kx / wn_ ** 2
        print(f'\nfitted 2nd-order:  w_n {wn_:.2f} rad/s ({wn_/2/np.pi:.2f} Hz)   '
              f'zeta {z_:.3f}   DC {g_:.3f}')
        print(f'  effective inertia K/w_n^2 = {Ie:.2f}   model says {Im:.2f}   '
              f'({Im/max(Ie,1e-9):.2f}x off)')
        print(f'  zeta predicted from measured inertia: {Dx/(2*np.sqrt(Kx*Ie)):.3f}')
        print(f'  measured zeta is higher if friction is adding dissipation')
        print(f'\n  to reach zeta=0.7 at this K, using the MEASURED inertia:')
        print(f'    D[{name}] = {2*0.7*np.sqrt(Kx*Ie):.0f}   (currently {Dx:.0f})')
        print(f'    verify by re-running the sweep -- do not trust the model bound')
    if mode == 'setpoint':
        dc = mag[:max(1, len(mag) // 20)].mean()
        Q = mag[pk] / max(dc, 1e-12)
        print(f'  DC gain {dc:.3f}   peak/DC {Q:.2f}')
        if Q > 1.05:
            print(f'  => resonant, zeta ~ {1/(2*Q):.2f}   (design assumed '
                  f'{float(d["D"][axis])/(2*np.sqrt(float(d["K"][axis])*float(d["ref_inertia"][axis]))):.2f})')
        else:
            print('  => no resonant peak: overdamped or the drive is too weak')
        if dc < 0.5:
            cmd = np.abs(d['eq'][:, axis] - d['base'][axis]).max()
            act = np.abs(d['pose'][:, axis] - d['base'][axis]).max()
            print(f'\n  ** NOT TRACKING: DC gain {dc:.3f}. Commanded {cmd*1000:.2f} mm, '
                  f'moved {act*1000:.2f} mm ({act/max(cmd,1e-12)*100:.0f}%).')
            print(f'  ** Drive force was {float(d["K"][axis])*cmd:.1f} N. If that is not well')
            print(f'  ** above the friction floor (~5 N) the sweep measured stiction,')
            print(f'  ** not dynamics. Re-run with a larger --amp.')
    else:
        lo = mag[:max(1, len(mag) // 10)].mean()
        print(f'  low-frequency |accel/F| = {lo:.4g}  -> apparent mass {1/max(lo,1e-12):.2f}')
        print(f'  model says inertia_d[{name}] = {float(d["ref_inertia"][axis]):.3f}')

    late = np.sum(d['dt'][1:] > 3 * float(d['dt_nominal']))
    print(f'\nloop: {late} late ticks, worst {d["dt"][1:].max()*1000:.1f} ms')

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        ax[0].loglog(f[1:], np.abs(H[1:])); ax[0].set_ylabel('|H|')
        ax[1].semilogx(f[1:], np.degrees(np.angle(H[1:]))); ax[1].set_ylabel('phase [deg]')
        ax[2].semilogx(f[1:], coh[1:]); ax[2].set_ylabel('coherence')
        ax[2].set_xlabel('Hz'); ax[0].set_title(f'{label}  axis {name}')
        for a in ax:
            a.grid(True, which='both', alpha=0.3)
            a.axvline(float(d['f0']), color='k', ls=':'); a.axvline(float(d['f1']), color='k', ls=':')
        plt.tight_layout(); plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    c = sub.add_parser('chirp', help='run a frequency sweep and log it')
    c.add_argument('--mode', choices=('setpoint', 'wrench'), default='setpoint')
    c.add_argument('--axis', choices=AXES, default='x')
    c.add_argument('--amp', type=float, default=None,
                   help='m (translation) or rad (rotation). Default 0.015 m / 0.15 rad.')
    c.add_argument('--f0', type=float, default=0.2)
    c.add_argument('--f1', type=float, default=10.0)
    c.add_argument('--duration', type=float, default=40.0)
    c.add_argument('--ramp', type=float, default=1.0, help='fade in/out [s]')
    c.add_argument('--v-max', type=float, default=0.15,
                   help='taper amplitude to keep peak TCP speed under this [m/s]')
    c.add_argument('--friction-floor', type=float, default=5.0,
                   help='measured translational friction floor [N], used to warn')
    c.add_argument('--friction-floor-rot', type=float, default=0.9,
                   help='measured rotational friction floor [Nm], used to warn')
    c.add_argument('--hold-k', type=float, default=200.0,
                   help='wrench mode: stiffness that holds position while driving')
    c.add_argument('--zeta', type=float, default=1.0)
    c.add_argument('--inertia', default=None,
                   help='6 task inertias, comma separated. Overrides the '
                        'payload+residual estimate, for probing other values.')
    c.add_argument('--fc-nm', default=None, help='6 per-joint Coulomb torques [Nm]')
    c.add_argument('--no-friction', action='store_true',
                   help='disable friction feedforward (to identify it)')
    c.add_argument('--force-max', type=float, default=60.0)
    c.add_argument('--speed-max', type=float, default=0.5)
    c.add_argument('--excursion', type=float, default=0.05, help='abort beyond [m]')
    c.add_argument('--watchdog', type=float, default=10.0)
    c.add_argument('--out', default='chirp.npz')

    a = sub.add_parser('analyze', help='frequency response from a logged sweep')
    a.add_argument('file')
    a.add_argument('--min-coh', type=float, default=0.5)
    a.add_argument('--plot', action='store_true')

    args = p.parse_args()
    (cmd_chirp if args.cmd == 'chirp' else cmd_analyze)(args)


if __name__ == '__main__':
    main()
