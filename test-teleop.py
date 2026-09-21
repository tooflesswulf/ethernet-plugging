"""
Does the calibrated config actually make teleop work?

Everything in friction.toml was measured one joint at a time, in joint space,
with the impedance controller switched off. This is the first thing that runs
the whole stack together and asks the question the user actually has: when I
push the stick along one axis, does the tool go that way.

Two parts.

PART A -- fc_assist and the limit cycle
---------------------------------------
fc_assist weights the STANDSTILL term, so it decides how hard the controller
pushes on a joint that is stuck. Too little and the deadband stays; too much
and the push overshoots, the error reverses, the push reverses, and the joint
hunts. That is the limit cycle impedance.py:63 has warned about since before
any of this was measured, and NOBODY HAS EVER TESTED IT. The one run at
assist 1.0 (offaxis-20260919-165630.npz) was moving throughout and used the
old, larger f_c.

Note a limit cycle cannot start from a perfect equilibrium: with e = 0 the
commanded torque is zero, so the assist term is zero too. It needs a standing
error to feed on. So each trial commands a small offset, lets it settle, then
HOLDS and watches. Sustained velocity during the hold is the signature; a
stable joint goes quiet.

PART B -- per-axis teleop response
----------------------------------
A commanded step along each of the six task axes, both directions, measuring
how much of it the tool actually realises and how much lands on the other five.

Read the realised fraction FIRST. Every earlier attempt at this topped out at
0.43 because the arm never left its deadband, and the off-axis numbers from
inside the deadband are presliding deflection, not coupling -- they mimic a
linear coupling exactly and produced two confident wrong verdicts. Below ~0.9
the off-axis column means nothing.

Expect the axes to differ, and not because friction differs. Joint 5 has a
translational moment arm of ZERO, so no push at the TCP drives it; joint 4's is
0.117 m against joint 1's 0.934. The same joint torque costs 3x the hand force
at the wrist, which is why the wrist feels sticky in free-drive even though its
breakaway is the smallest in the arm.

WHAT THIS CANNOT TELL YOU
-------------------------
Nothing here says anything about free-drive feel. The static term is
f_c * assist * tanh(tau_cmd / t_eps) -- it keys off COMMANDED torque, and
free-drive commands none, so f_c is identically zero there. brr.py does not even
construct a CartesianImpedance. Making hand-guiding light needs intent sensing,
which is a different mechanism entirely.

SAFETY
------
The gentlest script here: no breakaway ramps, no post-breakaway observation,
just the impedance controller doing its job with bounded equilibrium offsets
ramped in rather than stepped. Aborts on force, speed and excursion. The arm
moves by the commanded amplitude -- clear that space.
"""
import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from impedance import (CartesianImpedance, pose_error, load_friction,
                       load_scales, load_gravity_residual)
from kinematics import URKin

ROBOT_IP = '192.168.0.100'
AXES = ['x', 'y', 'z', 'rx', 'ry', 'rz']

FORCE_MAX = 60.0      # N, raw
SPEED_MAX = 0.40      # m/s
EXCURSION = 0.08      # m from base, any translational axis
SETTLE_V = 0.002      # m/s
SETTLE_W = 0.01       # rad/s
SETTLE_HOLD = 0.5     # s below those to count as settled
SETTLE_TIMEOUT = 5.0  # s

# Part A: what counts as hunting rather than settling.
QUIET_QD = 0.004      # rad/s RMS per joint -- above the 0.001-0.004 noise floor
HOLD_TIME = 3.0       # s of holding still while we watch


def offset_pose(base, axis, delta):
    """Displace one task axis. Rotations compose, never subtract."""
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


def guard(pose, base, twist, force, tau):
    if np.max(np.abs(force[:3])) > FORCE_MAX:
        return f'force {np.max(np.abs(force[:3])):.1f} N'
    if np.linalg.norm(twist[:3]) > SPEED_MAX:
        return f'speed {np.linalg.norm(twist[:3]):.2f} m/s'
    if np.max(np.abs(pose[:3] - base[:3])) > EXCURSION:
        return f'excursion {1000 * np.max(np.abs(pose[:3] - base[:3])):.0f} mm'
    if not np.all(np.isfinite(tau)):
        return 'non-finite torque'
    return None


def drive(ctrl, recv, kin, imp, base, axis, amp, ramp_t, scales, theta,
          hold=0.0, trace=None):
    """
    Ramp the equilibrium to base+amp, settle, then hold for `hold` seconds.

    Returns a dict. The hold phase is what Part A reads; Part B uses the
    settled displacement. Ramped rather than stepped: a step commands K*amp
    instantly, which is a jolt and excites the very transient we are trying to
    keep out of the settled numbers.
    """
    vis, cou = scales
    imp.reset()
    t0 = time.perf_counter()
    quiet_since = None
    settled_at = None
    peak_off_t = peak_off_r = 0.0
    hold_qd = []
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
        bias = None if theta is None else kin.gravity_bias(q, theta)
        tau, F, _ = imp.compute(q, qd, pose, eq, twist, J, tau_bias=bias)
        ctrl.directTorque(tau.tolist(), vis, cou)

        d = pose_error(base, pose)
        _, ot, orr = split(d, axis)
        if frac >= 1.0:
            peak_off_t = max(peak_off_t, ot)
            peak_off_r = max(peak_off_r, orr)
        if trace is not None:
            trace.append(np.r_[t, axis, amp, pose, eq, F, d, q, qd, tau])

        why = guard(pose, base, twist, force, tau)
        if why:
            raise RuntimeError(why)

        now = time.perf_counter()
        if settled_at is None:
            still = (np.linalg.norm(twist[:3]) < SETTLE_V
                     and np.linalg.norm(twist[3:]) < SETTLE_W)
            if frac >= 1.0 and still:
                quiet_since = quiet_since if quiet_since is not None else now
                if now - quiet_since >= SETTLE_HOLD:
                    settled_at = now
            else:
                quiet_since = None
            if t > ramp_t + SETTLE_TIMEOUT:
                settled_at = now          # timed out; record anyway
                quiet_since = None
        else:
            hold_qd.append(np.abs(qd))
            if now - settled_at >= hold:
                break
        ctrl.waitPeriod(ts)

    pose = np.array(recv.getActualTCPPose())
    d = pose_error(base, pose)
    on, ot, orr = split(d, axis)
    err = pose_error(pose, offset_pose(base, axis, amp))
    H = np.array(hold_qd) if hold_qd else np.zeros((1, 6))
    return dict(axis=axis, amp=amp, on=on, off_t=ot, off_r=orr,
                peak_off_t=peak_off_t, peak_off_r=peak_off_r,
                resid=float(np.linalg.norm(err[:3])),
                settled=quiet_since is not None,
                hold_qd_rms=np.sqrt((H ** 2).mean(axis=0)),
                hold_n=len(hold_qd))


def report_assist(rows):
    """Part A: where does holding still stop being still?"""
    print('\n' + '=' * 76)
    print('  PART A -- fc_assist and the limit cycle')
    print('=' * 76)
    print(f'  hold {HOLD_TIME:.0f} s after settling; quiet means every joint '
          f'under {QUIET_QD} rad/s RMS\n')
    moved = max(abs(r['on'] / r['amp']) if r['amp'] else 0.0 for r in rows)
    if moved < 0.5:
        print('!' * 76)
        print('  THIS TESTED NOTHING.')
        print('!' * 76)
        print(f'  The arm realised at most {moved:.2f} of the commanded offset, '
              f'so the joints\n  never broke free. A limit cycle needs the '
              f'joint to break loose and then\n  OVERSHOOT; one that is still '
              f'stuck cannot hunt, however hard the assist\n  term pushes. '
              f'Raise --assist-amp until this clears ~0.9 and re-run.\n')
    print('  assist   worst joint   qd RMS [rad/s]   realised   verdict')
    limit = None
    for r in rows:
        rms = r['hold_qd_rms']
        k = int(np.argmax(rms))
        hunting = rms[k] > QUIET_QD
        if hunting and limit is None:
            limit = r['assist']
        fr = abs(r['on'] / r['amp']) if r['amp'] else np.nan
        print(f'   {r["assist"]:5.2f}      joint {k}        {rms[k]:.5f}     '
              f'{fr:5.2f}      '
              + ('HUNTING' if hunting else 'quiet')
              + ('   <-- first' if hunting and limit == r['assist'] else ''))
    print()
    if limit is None and moved < 0.5:
        print('  No hunting -- but see above: nothing moved, so this says '
              'nothing about\n  whether fc_assist is safe.')
    elif limit is None:
        print('  No hunting at any assist tested, and the arm DID move, so the '
              'standstill\n  term is not driving it into a limit cycle at '
              'these f_c. The highest value\n  tested is usable.')
    else:
        ok = [r['assist'] for r in rows if r['assist'] < limit]
        print(f'  Hunting starts at fc_assist {limit:.2f}.')
        if ok:
            print(f'  Highest quiet value: {max(ok):.2f}. Set fc_assist there '
                  f'or below in\n  friction.toml.')
        else:
            print('  Hunting even at the lowest value tested -- something is '
                  'driving the arm\n  that is not the assist term. Check the '
                  'firmware coulomb scales against\n  test-scale-sweep.py '
                  'before touching f_c.')


def report_axes(rows, kin, base_q):
    """Part B: per-axis response."""
    J = kin.jacobian(base_q)
    print('\n' + '=' * 76)
    print('  PART B -- response to a commanded step, per axis')
    print('=' * 76)
    print('  axis  dir   realised/cmd   off-trans[mm]  off-rot[mrad]   resid[mm]')
    per = {}
    for r in rows:
        f = abs(r['on'] / r['amp']) if r['amp'] else np.nan
        per.setdefault(r['axis'], []).append(f)
        print(f'   {AXES[r["axis"]]:3s}   {"+" if r["amp"] > 0 else "-"}      '
              f'{f:5.2f}        {1000 * r["off_t"]:8.2f}      '
              f'{1000 * r["off_r"]:8.2f}    {1000 * r["resid"]:7.2f}'
              + ('' if r['settled'] else '   (no settle)'))
    print('\n  axis   realised (mean)   verdict')
    bad = []
    for a in sorted(per):
        f = float(np.mean(per[a]))
        v = ('ok' if f > 0.9 else
             'DEADBAND -- off-axis numbers above are meaningless' if f < 0.6
             else 'marginal')
        if f <= 0.9:
            bad.append(AXES[a])
        print(f'   {AXES[a]:4s}      {f:5.2f}          {v}')
    print()
    if bad:
        print(f'  Axes not reaching their command: {", ".join(bad)}.')
        print('  Before blaming friction, check the lever. A commanded task-axis')
        print('  force reaches each joint through J^T, and the wrist joints have')
        print('  almost no translational arm:')
        arms = [float(np.linalg.norm(J[:3, j])) for j in range(6)]
        print('    joint:  ' + '  '.join(f'{j}' for j in range(6)))
        print('    |Jv|:   ' + '  '.join(f'{a:.2f}' for a in arms) + '  m')
        print('  A joint with a small arm needs a large TCP force for a modest')
        print('  joint torque, which no amount of f_c changes.')
    else:
        print('  Every axis reaches its command. The off-axis columns are '
              'meaningful.')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', default=ROBOT_IP)
    ap.add_argument('--script', default='rtde_control-1.6.5-frictionfix.script')
    ap.add_argument('--part', choices=('a', 'b', 'both'), default='both')
    ap.add_argument('--assists', default='0.0,0.4,0.8,1.0',
                    help='fc_assist values for part A')
    ap.add_argument('--axes', default='0,1,2,3,4,5')
    ap.add_argument('--amp', type=float, default=15.0,
                    help='translational step [mm] for part B')
    ap.add_argument('--rot-amp', type=float, default=60.0,
                    help='rotational step [mrad] for part B')
    ap.add_argument('--ramp', type=float, default=1.0)
    ap.add_argument('--hold', type=float, default=HOLD_TIME)
    ap.add_argument('--assist-amp', type=float, default=25.0,
                    help='part A offset [mm]. MUST be large enough to actually '
                         'move the joint: a limit cycle needs the joint to '
                         'break free and overshoot, and a stuck joint cannot '
                         'hunt. The 2 mm of the first run realised 0.07-0.36 mm '
                         'and tested nothing.')
    ap.add_argument('--corrected-fc', action='store_true',
                    help='use the gravity-CORRECTED f_c table, which is only '
                         'valid with the bias on -- pairs with the default '
                         '(bias enabled), invalid with --no-bias')
    ap.add_argument('--f-sat', type=float, default=None,
                    help='translational force cap [N]. THIS limits the command, '
                         'not K -- compute() saturates F at F_sat, so raising K '
                         'alone just reaches the cap sooner.')
    ap.add_argument('--no-bias', action='store_true',
                    help='skip the gravity-residual correction (tau_bias)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    cfg = load_friction()
    vis, cou = load_scales()
    if args.corrected_fc:
        if args.no_bias:
            ap.error('--corrected-fc needs the gravity bias; drop --no-bias')
        if not np.any(cfg['f_c_pos_bias']):
            ap.error('friction.toml has no f_c_pos_bias table')
        cfg['f_c_pos'] = np.asarray(cfg['f_c_pos_bias'], float)
        cfg['f_c_neg'] = np.asarray(cfg['f_c_neg_bias'], float)
    theta = None if args.no_bias else load_gravity_residual()
    axes = [int(a) for a in args.axes.split(',')]
    assists = [float(a) for a in args.assists.split(',')]
    out = args.out or time.strftime('teleop-%Y%m%d-%H%M%S.npz')

    print('TELEOP VALIDATION')
    print(f'  f_c+   {np.round(cfg["f_c_pos"], 2)}'
          + ('   (gravity-corrected)' if args.corrected_fc else ''))
    print(f'  f_c-   {np.round(cfg["f_c_neg"], 2)}')
    print(f'  f_k+   {np.round(cfg["f_k_pos"], 2)}')
    print(f'  f_k-   {np.round(cfg["f_k_neg"], 2)}')
    print(f'  firmware coulomb {np.round(cou, 2)}   viscous {np.round(vis, 2)}')
    print(f'  fc_assist {cfg["fc_assist"]} (config)'
          + (f'; part A sweeps {assists}' if args.part in 'ab'[0] + 'both'
             else ''))
    print(f'  gravity bias: ' + ('OFF' if theta is None
                                 else f'theta {np.round(theta, 3)}'))
    if args.part in ('b', 'both'):
        print(f'  part B: axes {[AXES[a] for a in axes]}, '
              f'+/-{args.amp} mm / {args.rot_amp} mrad')
    if args.dry_run:
        return

    import rtde_control
    import rtde_receive
    print('\nThe arm moves by the commanded amplitude. Clear space, hand on '
          'the e-stop.')
    input('enter to start, ctrl-C to abort: ')

    ctrl = rtde_control.RTDEControlInterface(args.ip, 500.0)
    recv = rtde_receive.RTDEReceiveInterface(args.ip)
    ctrl.setCustomScriptFile(args.script)
    deadline = time.monotonic() + 5.0
    while not ctrl.isProgramRunning():
        if time.monotonic() > deadline:
            raise RuntimeError('control script did not start')
        time.sleep(0.01)
    kin = URKin(ctrl.getTCPOffset())
    imp = CartesianImpedance(f_c=cfg['f_c_pos'], f_c_neg=cfg['f_c_neg'],
                             tau_rated=kin.tau_rated)
    imp.f_k = np.asarray(cfg['f_k_pos'], float)
    imp.f_k_neg = np.asarray(cfg['f_k_neg'], float)
    imp.fc_assist = float(cfg['fc_assist'])
    if args.f_sat is not None:
        imp.F_sat = np.array([args.f_sat] * 3 + list(imp.F_sat[3:]))
    scales = (list(vis), list(cou))

    base_q = np.array(recv.getActualQ())
    print(f'\nbase pose (rad): {np.round(base_q, 5)}')
    print(f'K {np.round(imp.K_free, 0)}   F_sat {np.round(imp.F_sat, 0)}\n')

    trace, a_rows, b_rows = [], [], []
    ctrl.setWatchdog(0.05)
    try:
        if args.part in ('a', 'both'):
            print('--- part A: holding still at each fc_assist ---')
            for asst in assists:
                imp.fc_assist = asst
                base = np.array(recv.getActualTCPPose())
                # a small offset so there IS a standing command to feed on --
                # at exactly zero error the assist term is zero and no limit
                # cycle can start
                r = drive(ctrl, recv, kin, imp, base, 0,
                          args.assist_amp * 1e-3, args.ramp,
                          scales, theta, hold=args.hold, trace=trace)
                r['assist'] = asst
                a_rows.append(r)
                k = int(np.argmax(r['hold_qd_rms']))
                print(f'  assist {asst:4.2f}: worst joint {k} '
                      f'{r["hold_qd_rms"][k]:.5f} rad/s RMS over {r["hold_n"]} '
                      f'samples   (realised '
                      f'{abs(r["on"] / (args.assist_amp * 1e-3)):.2f})')
            imp.fc_assist = float(cfg['fc_assist'])

        if args.part in ('b', 'both'):
            print(f'\n--- part B: per-axis step (fc_assist {imp.fc_assist}) ---')
            for a in axes:
                amp = (args.amp * 1e-3) if a < 3 else (args.rot_amp * 1e-3)
                for sgn in (+1.0, -1.0):
                    base = np.array(recv.getActualTCPPose())
                    r = drive(ctrl, recv, kin, imp, base, a, sgn * amp,
                              args.ramp, scales, theta, hold=0.0, trace=trace)
                    b_rows.append(r)
                    print(f'  {AXES[a]:3s} {"+" if sgn > 0 else "-"}: '
                          f'realised {abs(r["on"] / (sgn * amp)):5.2f}   '
                          f'off_t {1000 * r["off_t"]:6.2f} mm   '
                          f'off_r {1000 * r["off_r"]:6.2f} mrad')
                    # back to base, same way every time
                    drive(ctrl, recv, kin, imp, base, a, 0.0, args.ramp,
                          scales, theta, hold=0.0)
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
        np.savez(out, t=T[:, 0], axis=T[:, 1], amp=T[:, 2], pose=T[:, 3:9],
                 eq=T[:, 9:15], wrench=T[:, 15:21], disp=T[:, 21:27],
                 q=T[:, 27:33], qd=T[:, 33:39], tau=T[:, 39:45],
                 assists=np.array([r['assist'] for r in a_rows]),
                 hold_rms=np.array([r['hold_qd_rms'] for r in a_rows]
                                   ).reshape(-1, 6),
                 axis_results=np.array([[r['axis'], r['amp'], r['on'],
                                         r['off_t'], r['off_r'], r['resid'],
                                         float(r['settled'])] for r in b_rows]
                                       ).reshape(-1, 7),
                 base_q=base_q, f_c_pos=cfg['f_c_pos'], f_c_neg=cfg['f_c_neg'],
                 f_k_pos=cfg['f_k_pos'], f_k_neg=cfg['f_k_neg'],
                 fc_assist=cfg['fc_assist'], viscous=vis, coulomb=cou,
                 theta=(np.zeros(4) if theta is None else theta))
        print(f'\nwrote {out}')
    if a_rows:
        report_assist(a_rows)
    if b_rows:
        report_axes(b_rows, kin, base_q)


if __name__ == '__main__':
    main()
