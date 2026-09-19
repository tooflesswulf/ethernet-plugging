"""
Why does a +x teleop command move y, z and the rotations?

Three mechanisms can do it and they need different fixes, so measure before
modelling:

  1. FRICTION gating. tau = J^T F spreads the commanded wrench over six joints,
     each with its own stiction threshold. Only the joints that clear threshold
     move, so the arm travels along J restricted to that subset -- a direction
     unrelated to +x. With f_c = 0 (env.py's default today) joint 1 alone needs
     6-19 mm of equilibrium offset before it breaks loose, against a 26.7 mm
     leash, so small joystick commands sit inside the deadband.

  2. TCP-CoM lever arm. The TCP is 12.1 cm from the payload CoM, so a pure force
     at the TCP makes a moment r x F ABOUT THE COM -- 0.121 Nm per N here. No
     choice of frame removes this; commanding zero moment about the TCP is not
     the same as commanding zero angular acceleration.

  3. Task-space inertia. q'' = H(q)^-1 tau and H is not diagonal, so a pure x
     force accelerates every axis (measured: off-axis/on-axis = 0.63 in
     translation, 3.66 rad/s^2 per N in rotation). directTorque compensates
     gravity and friction but cannot cancel the arm's own inertia; that needs
     inertia shaping, which impedance.py removed.

HOW THEY SEPARATE
-----------------
In free space the impedance controller drives e -> 0, so at steady state F -> 0
and with no force there is no tilt and no acceleration. 2 and 3 are therefore
TRANSIENT. What leaves a persistent off-axis error is friction: the arm stops
where the commanded force drops below stiction, in whichever direction the stuck
joints dictate.

So sweep the commanded amplitude and watch the SETTLED off-axis error:

  constant magnitude, fraction falling as 1/A   -> friction (mechanism 1)
  magnitude proportional to A, constant fraction -> linear coupling (2 or 3)
  large in transit, gone once settled            -> inertia (3)

The script fits both models and reports which one the data supports.

Each amplitude is run in both directions and repeated: friction is direction
dependent and stochastic, so a single step tells you very little.

SAFETY
------
Much gentler than test-scale-sweep.py: no breakaway ramps and no post-breakaway
observation. This is the impedance controller doing its normal job with bounded
equilibrium offsets, ramped in rather than stepped. Aborts on force, speed and
excursion, same as the chirp harness. The arm WILL move by the commanded
amplitude -- clear that space.
"""
import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from impedance import CartesianImpedance, pose_error
from kinematics import URKin

ROBOT_IP = '192.168.0.100'

FORCE_MAX = 60.0      # N, raw
SPEED_MAX = 0.40      # m/s -- raised: breaking out of a deep deadband snaps
EXCURSION = 0.06      # m from base, any translational axis
SETTLE_V = 0.002      # m/s -- "stopped"
SETTLE_W = 0.01       # rad/s
SETTLE_HOLD = 0.5     # s below those before we call it settled
SETTLE_TIMEOUT = 4.0  # s


def offset_pose(base, axis, delta):
    """Displace one of the 6 task axes. Rotations compose, never subtract."""
    p = np.array(base, float)
    if axis < 3:
        p[axis] += delta
        return p
    rv = np.zeros(3)
    rv[axis - 3] = delta
    p[3:] = (R.from_rotvec(rv) * R.from_rotvec(p[3:])).as_rotvec()
    return p


def split(d, axis):
    """(on-axis, off-axis translation, off-axis rotation) of a 6-vector."""
    lin, rot = d[:3].copy(), d[3:].copy()
    if axis < 3:
        on = lin[axis]
        lin[axis] = 0.0
    else:
        on = rot[axis - 3]
        rot[axis - 3] = 0.0
    return on, float(np.linalg.norm(lin)), float(np.linalg.norm(rot))


def run_step(ctrl, recv, kin, imp, base, axis, amp, ramp_t, dt, trace):
    """
    Ramp the equilibrium to base+amp, hold until settled, return the result.

    Ramped rather than stepped: a step commands K*amp instantly, which is a jolt
    and excites exactly the transient we are trying to separate out.
    """
    imp.reset()
    t0 = time.perf_counter()
    quiet_since = None
    peak_off_t = peak_off_r = 0.0
    abort = None
    while True:
        ts = ctrl.initPeriod()
        t = time.perf_counter() - t0
        frac = min(1.0, t / ramp_t) if ramp_t > 0 else 1.0
        eq = offset_pose(base, axis, amp * frac)

        q = np.array(recv.getActualQ())
        qd = np.array(recv.getActualQd())
        pose = np.array(recv.getActualTCPPose())
        twist = np.array(recv.getActualTCPSpeed())
        force = np.array(recv.getActualTCPForce())

        J = kin.jacobian(q)
        tau, F, e = imp.compute(q, qd, pose, eq, twist, J)
        ctrl.directTorque(tau.tolist(), [0.0] * 6, [0.0] * 6)

        d = pose_error(base, pose)          # realised displacement from base
        _, ot, orr = split(d, axis)
        if frac >= 1.0:
            peak_off_t = max(peak_off_t, ot)
            peak_off_r = max(peak_off_r, orr)
        trace.append(np.r_[t, amp, pose, eq, F, d])

        if np.max(np.abs(force[:3])) > FORCE_MAX:
            abort = f'force {np.max(np.abs(force[:3])):.1f} N'
        elif np.linalg.norm(twist[:3]) > SPEED_MAX:
            abort = f'speed {np.linalg.norm(twist[:3]):.2f} m/s'
        elif np.max(np.abs(pose[:3] - base[:3])) > EXCURSION:
            abort = f'excursion {1000*np.max(np.abs(pose[:3]-base[:3])):.0f} mm'
        elif not np.all(np.isfinite(tau)):
            abort = 'non-finite torque'
        if abort:
            raise RuntimeError(abort)

        now = time.perf_counter()
        if frac >= 1.0 and (np.linalg.norm(twist[:3]) < SETTLE_V
                            and np.linalg.norm(twist[3:]) < SETTLE_W):
            quiet_since = quiet_since if quiet_since is not None else now
            if now - quiet_since >= SETTLE_HOLD:
                break
        else:
            quiet_since = None
        if t > ramp_t + SETTLE_TIMEOUT:
            break
        ctrl.waitPeriod(ts)

    pose = np.array(recv.getActualTCPPose())
    d = pose_error(base, pose)
    on, ot, orr = split(d, axis)
    eq_full = offset_pose(base, axis, amp)
    err = pose_error(pose, eq_full)         # what the spring is still pushing on
    return dict(amp=amp, on=on, off_t=ot, off_r=orr,
                peak_off_t=peak_off_t, peak_off_r=peak_off_r,
                resid=float(np.linalg.norm(err[:3])), settled=quiet_since is not None)


def fit_models(A, Y):
    """
    Compare Y = const against Y = slope*A. Returns (r2_const, r2_prop, verdict).

    These are the two signatures: friction leaves a roughly fixed residual
    regardless of how hard you push, a linear coupling scales with the command.
    """
    A, Y = np.asarray(A, float), np.asarray(Y, float)
    m = np.isfinite(A) & np.isfinite(Y)
    A, Y = A[m], Y[m]
    if len(A) < 3 or np.ptp(A) < 1e-9:
        return np.nan, np.nan, 'not enough amplitudes'
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    if ss_tot <= 0:
        return 1.0, 0.0, 'constant (exactly)'
    r2_const = 0.0                                    # const model = the mean
    k = float(np.sum(A * Y) / np.sum(A * A))          # proportional, no intercept
    r2_prop = 1.0 - float(np.sum((Y - k * A) ** 2)) / ss_tot
    spread = float(np.std(Y, ddof=1) / max(abs(Y.mean()), 1e-12))
    if r2_prop > 0.6 and k * np.ptp(A) > 0.5 * abs(Y.mean()):
        return r2_const, r2_prop, 'proportional to amplitude'
    if spread < 0.35:
        return r2_const, r2_prop, 'constant with amplitude'
    return r2_const, r2_prop, 'neither cleanly'


def report(rows, axis):
    name = ['x', 'y', 'z', 'rx', 'ry', 'rz'][axis]
    print('\n' + '=' * 78)
    print(f'  OFF-AXIS RESPONSE to a commanded +/-{name} step')
    print('=' * 78)
    print('  amp[mm]  dir  on[mm]  off-trans[mm]  off-rot[mrad]  peak-trans[mm]'
          '  resid[mm]')
    for r in rows:
        print(f'   {1000*r["amp"]:6.1f}   {"+" if r["amp"]>0 else "-"}  '
              f'{1000*r["on"]:6.2f}      {1000*r["off_t"]:7.2f}       '
              f'{1000*r["off_r"]:7.2f}        {1000*r["peak_off_t"]:7.2f}   '
              f'{1000*r["resid"]:7.2f}'
              + ('' if r['settled'] else '   (did not settle)'))

    A = [abs(r['amp']) for r in rows]
    realised = np.array([abs(r['on']) / max(abs(r['amp']), 1e-12) for r in rows])
    print(f'\n  realised / commanded displacement: median {np.median(realised):.2f}, '
          f'max {realised.max():.2f}')
    if realised.max() < 0.5:
        print('\n' + '!' * 78)
        print('  THIS RUN CANNOT ANSWER THE QUESTION.')
        print('!' * 78)
        print('  The arm never left the friction deadband -- it realised at most '
              f'{100*realised.max():.0f}% of\n  the commanded displacement, and '
              'the holding force was still climbing at the\n  largest '
              'amplitude. Everything measured here is PRESLIDING deflection: '
              'the\n  arm flexing in place while stuck, along whatever '
              'direction its stuck\n  compliance points. That deflection is '
              'elastic, so it scales with the command\n  and mimics the '
              '"linear coupling" signature exactly -- the verdict below would\n'
              '  be an artefact.\n')
        print('  Push harder before reading anything into it:')
        print('    --f-sat 80 --amps 10,20,30,40,50   (F_sat, not K, is the cap)')
        print('  or turn the feedforward on, which is what it is for:')
        print('    --f-c 10.34,9.52,6.96,2.78,2.94,2.07\n')
    print()
    for label, key in (('settled off-axis translation', 'off_t'),
                       ('settled off-axis rotation   ', 'off_r'),
                       ('PEAK off-axis translation   ', 'peak_off_t')):
        Y = [r[key] for r in rows]
        _, r2p, verdict = fit_models(A, Y)
        unit = 'mrad' if 'rotation' in label else 'mm'
        print(f'  {label}: mean {1000*np.mean(Y):6.2f} {unit},  '
              f'R^2(prop) {r2p:5.2f}  ->  {verdict}')

    st = np.array([r['off_t'] for r in rows])
    pk = np.array([r['peak_off_t'] for r in rows])
    frac = float(np.mean(st / np.maximum(pk, 1e-12)))
    _, _, v_settled = fit_models(A, st)
    _, _, v_peak = fit_models(A, pk)
    print(f'\n  settled / peak off-axis translation: {frac:.2f}')
    print('\n' + '-' * 78)
    print('  READ AS')
    print('-' * 78)

    # Order matters. Inertia looks like "constant settled error" too -- the
    # difference is that its error is SMALL and the PEAK scales with amplitude,
    # because the coupling lives in the acceleration and is gone once the arm
    # stops. Check that before falling through to the friction reading.
    if v_peak == 'proportional to amplitude' and frac < 0.6:
        print('  Off-axis motion is mostly TRANSIENT: the peak scales with the '
              'command and\n  most of it is gone once the arm settles '
              f'(settled/peak {frac:.2f}). That is\n  task-space inertia '
              '(mechanism 3). More f_c will not touch it -- it needs\n  '
              'inertia shaping, which impedance.py removed for good reasons, '
              'or gentler\n  command ramps so less of it is excited.')
    elif v_settled == 'constant with amplitude':
        print('  Settled off-axis error is roughly CONSTANT however hard you '
              'push, and it\n  survives settling. That is the friction '
              'signature: a fixed residual of\n  about stiction/K, pointing '
              'wherever the stuck joints left it. Mechanism 1,\n  and f_c is '
              'currently zero -- that is the fix.')
    elif v_settled == 'proportional to amplitude':
        print('  Settled off-axis error scales WITH the command and persists '
              'at rest. That\n  is a linear coupling, not friction -- the '
              'TCP-CoM lever arm (mechanism 2).\n  f_c will not touch it. '
              'K_rot, or moving the TCP nearer the payload CoM, is\n  where '
              'the effort goes.')
    else:
        print('  Mixed or noisy. Add repeats (--reps) before concluding; '
              'friction is\n  stochastic and a handful of steps will not '
              'separate these.')
    print(f'\n  (settled: {v_settled};  peak: {v_peak})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', default=ROBOT_IP)
    ap.add_argument('--script', default='rtde_control-1.6.5-frictionfix.script')
    ap.add_argument('--axis', type=int, default=0, help='0=x 1=y 2=z 3=rx 4=ry 5=rz')
    ap.add_argument('--amps', default='2,4,8,12,16,20',
                    help='mm (or mrad for a rotary axis)')
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--ramp', type=float, default=1.0,
                    help='s to ramp the equilibrium in; not a step')
    ap.add_argument('--f-sat', type=float, default=None,
                    help='translational force saturation [N]. THIS is what caps '
                         'the command, not K: impedance.compute saturates F at '
                         'F_sat, so raising K alone just hits the cap sooner. '
                         'Default 40 (impedance.py).')
    ap.add_argument('--k-scale', type=float, default=1.0,
                    help='scale K, with D scaled by sqrt of it so zeta is '
                         'preserved. Raising K alone silently changes zeta -- '
                         'see impedance.calibrate.')
    ap.add_argument('--f-c', default=None,
                    help='6 comma-separated Nm to enable the friction '
                         'feedforward. Default OFF, matching env.py today.')
    ap.add_argument('--out', default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    axis = args.axis
    scale = 1e-3
    amps = [float(a) * scale for a in args.amps.split(',')]
    out = args.out or time.strftime('offaxis-%Y%m%d-%H%M%S.npz')
    f_c = (np.array([float(x) for x in args.f_c.split(',')]) if args.f_c
           else np.zeros(6))

    print(f'OFF-AXIS AMPLITUDE SWEEP, axis {axis}')
    print(f'  amplitudes {[round(1000*a,1) for a in amps]} mm, both directions, '
          f'{args.reps} reps')
    print(f'  equilibrium ramped in over {args.ramp} s, then held to rest')
    print(f'  f_c {np.round(f_c, 2)}' + ('   (OFF -- as env.py runs today)'
                                         if not args.f_c else ''))
    print(f'  firmware scales 0 on every joint (coulomb 0.8 is past the '
          f'over-compensation\n    threshold; see test-scale-sweep.py)')
    n = len(amps) * 2 * args.reps
    print(f'\n  {n} steps, ~{n * (args.ramp + 2.5) / 60:.0f} min')
    if args.dry_run:
        return

    import rtde_control
    import rtde_receive
    print('\nThe arm moves by the commanded amplitude on each step. Clear space.')
    input('enter to start, ctrl-C to abort: ')

    ctrl = rtde_control.RTDEControlInterface(args.ip, 500.0)
    recv = rtde_receive.RTDEReceiveInterface(args.ip)
    ctrl.setCustomScriptFile(args.script)
    deadline = time.monotonic() + 5.0
    while not ctrl.isProgramRunning():
        if time.monotonic() > deadline:
            raise RuntimeError('control script did not start')
        time.sleep(0.01)
    tcp = ctrl.getTCPOffset()
    kin = URKin(tcp)
    imp = CartesianImpedance(f_c=f_c, tau_rated=kin.tau_rated)
    if args.k_scale != 1.0:
        # D = 2 zeta sqrt(K I), so D must go as sqrt(K) to hold zeta.
        imp.K_free = imp.K_free * args.k_scale
        imp.D_free = imp.D_free * np.sqrt(args.k_scale)
        imp.K_contact = imp.K_contact * args.k_scale
        imp.D_contact = imp.D_contact * np.sqrt(args.k_scale)
    if args.f_sat is not None:
        imp.F_sat = np.array([args.f_sat] * 3 + list(imp.F_sat[3:]))
    dt = ctrl.getStepTime() or 0.002

    base = np.array(recv.getActualTCPPose())
    print(f'\nbase pose  : {np.round(base, 4)}')
    print(f'K          : {np.round(imp.K_free, 0)}')
    print(f'leash      : {1000*imp.F_sat[0]/imp.K_free[0]:.1f} mm\n')

    rows, trace = [], []
    ctrl.setWatchdog(0.05)
    try:
        for rep in range(args.reps):
            for a in amps:
                for sgn in (+1.0, -1.0):
                    # Re-read base every time. The return-to-base step below
                    # cannot be assumed to work: with the arm deep in its
                    # deadband it stays where the previous step left it, and
                    # every subsequent displacement is then measured from a
                    # stale origin. That is what made rep 1 of the first run
                    # report negative on-axis motion for a positive command.
                    base = np.array(recv.getActualTCPPose())
                    r = run_step(ctrl, recv, kin, imp, base, axis, sgn * a,
                                 args.ramp, dt, trace)
                    rows.append(r)
                    print(f'  rep {rep} {1000*sgn*a:+6.1f} mm -> on '
                          f'{1000*r["on"]:6.2f}  off_t {1000*r["off_t"]:6.2f}  '
                          f'off_r {1000*r["off_r"]:6.2f} mrad'
                          + ('' if r['settled'] else '  (no settle)'))
                    # back to base the same way every time
                    run_step(ctrl, recv, kin, imp, base, axis, 0.0,
                             args.ramp, dt, [])
    except KeyboardInterrupt:
        print('\naborted by user')
    except RuntimeError as e:
        print(f'\nABORT: {e}')
    finally:
        try:
            for _ in range(5):
                ctrl.directTorque([0.0] * 6, [0.0] * 6, [0.0] * 6)
            ctrl.stopJ(2.0)
            ctrl.stopScript()
        except Exception as e:
            print(f'  !! stop failed: {e!r}')

    if trace:
        T = np.array(trace)
        np.savez(out, t=T[:, 0], amp=T[:, 1], pose=T[:, 2:8], eq=T[:, 8:14],
                 wrench=T[:, 14:20], disp=T[:, 20:26],
                 results=np.array([[r['amp'], r['on'], r['off_t'], r['off_r'],
                                    r['peak_off_t'], r['peak_off_r'],
                                    r['resid'], float(r['settled'])]
                                   for r in rows]),
                 result_cols=np.array(['amp', 'on', 'off_t', 'off_r',
                                       'peak_off_t', 'peak_off_r', 'resid',
                                       'settled']),
                 base=base, axis=axis, f_c=f_c, K=imp.K_free, D=imp.D_free)
        print(f'\nwrote {out}')
    if rows:
        report(rows, axis)


if __name__ == '__main__':
    main()
