"""
Find the coulomb scale at which friction compensation stops freeing a joint and
starts driving it.

WHY THIS EXISTS
---------------
Every friction script in this repo measures breakaway at coulomb 0.8 and none of
them check that 0.8 is a sane place to measure. It may not be. Observed by hand
with brr.py (zero torque, compensation live):

  * coulomb 0 on every joint -- the arm is STICKY. It cannot be pushed into
    rising even with a deliberate shove.
  * coulomb populated -- a very gentle push starts the arm rising and it keeps
    going. It still holds if it is brought to rest first and released cleanly.

That is over-compensation. Compensation cancels friction, so it pushes ALONG the
velocity a joint already has; if it over-estimates, the net torque drives the
joint instead of freeing it. At exactly zero velocity there is no velocity to
act on, so static friction still holds -- which is why "hold it still and let
go" works and why any residual velocity at torque-mode entry is the hazard.

WHAT IT MEASURES
----------------
Breakaway torque alone cannot tell good compensation from over-compensation:
both lower it, monotonically, and neither looks wrong on its own. So at each
scale this ramps to breakaway and then ZEROES the commanded torque and watches:

  decelerates            friction was merely cancelled; removing the drive stops
                         the joint. The scale is on the right side.
  sustains / accelerates the compensation is pushing along the joint's own
                         velocity. Over-compensated.

The scale where that flips is the operating limit. A breakaway measured above it
is not a friction number -- it is partly the scale driving the joint -- so any
f_c identified there has to be redone below it.

SAFETY -- READ THIS
-------------------
This script deliberately watches the joint move after breakaway, which is the
one thing every other script here refuses to do. `identify` in test-impedance.py
accelerated joint 1 into a wall doing something similar. The exposure is bounded
as tightly as the measurement allows:

  * commanded torque is already ZERO for the whole observation -- nothing here
    commands motion, it only declines to stop it for 0.30 s
  * 1.0 deg of travel budget, hard abort on exceeding it
  * hard abort on QD_ABORT on any joint
  * only the joint under test is compensated; every other joint sits at scale 0
    with its full natural stiction, pinning the pose
  * every abort zeroes the SCALES first, restoring stiction, then the torque,
    then stopJ

Watch the first one. Hand on the e-stop.

The sweep runs at whatever configuration the arm is in when you start it, so put
the arm where you want it measured first. The threshold is load dependent, so
the answer is for THAT pose -- re-run it somewhere else before generalising.
"""
import argparse
import time

import numpy as np

from impedance import load_scales

# Scales come from friction.toml -- the single copy. The repo used to carry two
# disagreeing tables (test-friction-recal's and brr.py's) with nothing saying
# which was authoritative; --viscous still overrides for a deliberate probe.
_VIS, _COU = load_scales()
VISCOUS = [float(x) for x in _VIS]
COULOMB = [float(x) for x in _COU]
OFF = [0.0] * 6

# 1.2x the largest UNCOMPENSATED breakaway, as in test-friction-recal.py.
TAU_CAP = np.array([17.0, 26.0, 14.0, 5.0, 6.0, 5.0])
RATE = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])    # Nm/s

# Breakaway is DISPLACEMENT, not a velocity sample -- the same fix
# test-breakaway.py carries. A single tick over QD_DETECT is a noise spike or a
# micro-slip: on joint 1 those trips moved 0.016-0.018 deg against 0.2-0.48 for
# real ones. Runs before 2026-09-21 used the old single-sample rule and their
# breakaway column reads HIGH by whatever travel the velocity took to build.
DQ_CONFIRM = 0.00087  # rad (0.050 deg) -- breakaway
QD_DETECT = 0.02      # rad/s, and it must HOLD for DETECT_HOT ticks
DETECT_HOT = 10       # 20 ms at 500 Hz
QD_ABORT = 0.15       # rad/s on ANY joint
DQ_ABORT = 0.026      # rad (1.5 deg) on the joint under test
DQ_OTHER = 0.026      # rad (1.5 deg) on any other joint

# Velocity noise floor, from the stationary stretches of brr-log-old.npz:
# p50 0.0004 rad/s, p95 0.0012, joint 1 noisiest at 0.0038.
QD_QUIET = 0.006      # rad/s -- "stopped"
DQ_SETTLE = 0.035     # rad (2.0 deg) allowed during torque-mode entry
SETTLE_GRACE = 0.15   # s before the settle's velocity abort arms
SETTLE_HOT = 5        # consecutive ticks over QD_ABORT before it is a runaway

COAST_WINDOW = 0.30   # s to watch after zeroing torque at breakaway
COAST_BUDGET = 0.017  # rad (1.0 deg) of travel allowed during that window
APPROACH_BACKOFF = 0.035   # rad (2 deg) detour so every arrival is identical


def safe_stop(ctrl, why=''):
    """Scales first, then torque, then leave torque mode. Order matters."""
    if why:
        print(f'      STOP: {why}')
    try:
        for _ in range(5):
            ctrl.directTorque(OFF, OFF, OFF)
        ctrl.stopJ(2.0)
    except Exception as e:
        print(f'      !! safe_stop failed: {e!r}')


def wait_for_control_script(ctrl, timeout=5.0, poll=0.01):
    """reuploadScript() omits waitForProgramRunning(); see env.py."""
    deadline = time.monotonic() + timeout
    while not ctrl.isProgramRunning():
        if time.monotonic() > deadline:
            raise RuntimeError('control script did not start')
        time.sleep(poll)
    while True:
        try:
            ctrl.getTCPOffset()
            return
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(poll)


def scales_for(j, coulomb, viscous=None):
    """
    Compensate ONLY joint j, at the given coulomb scale.

    Everything else stays at 0 so its full natural stiction pins the pose. With
    all six compensated the arm creeps at high-load poses, which moves the very
    configuration the sweep is characterising.
    """
    v, c = [0.0] * 6, [0.0] * 6
    v[j] = VISCOUS[j] if viscous is None else float(viscous)
    c[j] = float(coulomb)
    return v, c


def approach(ctrl, q, joint):
    """
    Arrive at q the same way every time: back off 2 deg on the joint of
    interest, then come in. Friction state depends on approach direction and
    distance -- it sets where the joint sits inside its presliding band -- so
    without this the scales are not comparable to each other.
    """
    back = np.asarray(q, float).copy()
    back[joint] += APPROACH_BACKOFF
    ctrl.moveJ(back.tolist(), 0.3, 0.3)
    ctrl.moveJ(np.asarray(q, float).tolist(), 0.3, 0.3)


def settle(ctrl, recv, j, coulomb, dwell, qd_quiet=QD_QUIET, timeout=8.0):
    """
    Enter torque mode at ZERO torque and wait for the arm to stop.

    This must happen before q_start is read. moveJ leaves the joint under
    position control and the first directTorque call is what releases it; the
    arm then sags into its torque-mode equilibrium. Capture q_start before that
    and the sag reads as breakaway.

    Returns (q_start, sag, settled).
    """
    vis, cou = scales_for(j, coulomb)
    q0 = np.array(recv.getActualQ())
    zeros = [0.0] * 6
    t0 = time.perf_counter()
    quiet_since, hot = None, 0
    peak = np.zeros(6)
    while True:
        ts = ctrl.initPeriod()
        ctrl.directTorque(zeros, vis, cou)
        qd = np.abs(np.array(recv.getActualQd()))
        dq = np.array(recv.getActualQ()) - q0
        now = time.perf_counter()
        peak = np.maximum(peak, qd)

        # Debounced, and not armed until the mode switch has passed: moveJ
        # returns with residual velocity and the first readings across the
        # switch can spike. 5 ticks is 10 ms and ~1.5 mrad.
        if now - t0 > SETTLE_GRACE:
            hot = hot + 1 if np.max(qd) > QD_ABORT else 0
            if hot >= SETTLE_HOT:
                k = int(np.argmax(qd))
                raise RuntimeError(
                    f'settle: joint {k} at {qd[k]:.3f} rad/s for {hot} ticks '
                    f'(limit {QD_ABORT}); peak {np.round(peak, 3)}, travel '
                    f'{np.round(np.degrees(dq), 2)} deg')
        if np.max(np.abs(dq)) > DQ_SETTLE:
            # Crept, did not run. A fact about this pose and scale, not an
            # emergency: the caller records it and moves to the next scale.
            k = int(np.argmax(np.abs(dq)))
            print(f'      will not hold: joint {k} crept '
                  f'{np.degrees(dq[k]):+.2f} deg on entry')
            return np.array(recv.getActualQ()), dq, False

        if np.max(qd) < qd_quiet:
            quiet_since = quiet_since if quiet_since is not None else now
            if now - quiet_since >= dwell:
                q = np.array(recv.getActualQ())
                return q, q - q0, True
        else:
            quiet_since = None
        if now - t0 > timeout:
            k = int(np.argmax(peak))
            print(f'      settle timed out: joint {k} at {peak[k]:.4f} rad/s '
                  f'(need < {qd_quiet:.4f}) -- raise --qd-quiet if that is '
                  f'the noise floor')
            return np.array(recv.getActualQ()), dq, False
        ctrl.waitPeriod(ts)


def ramp(ctrl, recv, j, sgn, q_start, rate, tau_cap, coulomb, log):
    """Ramp joint j until it moves. Returns breakaway [Nm], or None at the cap."""
    tau = np.zeros(6)
    vis, cou = scales_for(j, coulomb)
    t0 = time.perf_counter()
    hot = 0
    while True:
        ts = ctrl.initPeriod()
        mag = rate * (time.perf_counter() - t0)
        if mag > tau_cap:
            return None
        tau[j] = sgn * mag
        ctrl.directTorque(tau.tolist(), vis, cou)

        q = np.array(recv.getActualQ())
        qd = np.array(recv.getActualQd())
        log.append(np.r_[time.perf_counter() - t0, coulomb, sgn * mag,
                         0.0, q, qd])          # phase 0 = ramp

        dq = q - q_start
        if np.max(np.abs(qd)) > QD_ABORT:
            raise RuntimeError(f'joint speed {np.max(np.abs(qd)):.3f} rad/s')
        if abs(dq[j]) > DQ_ABORT:
            raise RuntimeError(f'joint {j} travelled {np.degrees(dq[j]):.1f} deg')
        oth = np.abs(dq).copy()
        oth[j] = 0.0
        k = int(np.argmax(oth))
        if oth[k] > DQ_OTHER:
            raise RuntimeError(
                f'joint {k} moved {np.degrees(dq[k]):+.2f} deg while joint {j} '
                f'was under test; all dq = {np.round(np.degrees(dq), 2)}')

        if abs(dq[j]) > DQ_CONFIRM:
            return mag
        hot = hot + 1 if abs(qd[j]) > QD_DETECT else 0
        if hot >= DETECT_HOT:
            return mag
        ctrl.waitPeriod(ts)


def coast(ctrl, recv, j, coulomb, q_break, log=None, sgn=1.0):
    """
    The stability test. Torque is already zero; watch what the joint does.

    Returns (qd_at_break, qd_at_end, peak, travel, verdict).
    """
    vis, cou = scales_for(j, coulomb)
    zeros = [0.0] * 6
    t0 = time.perf_counter()
    qd0 = abs(np.array(recv.getActualQd())[j])
    peak = last = qd0
    while time.perf_counter() - t0 < COAST_WINDOW:
        ts = ctrl.initPeriod()
        ctrl.directTorque(zeros, vis, cou)
        qq = np.array(recv.getActualQ())
        qd = np.array(recv.getActualQd())
        dq = qq - q_break
        last = abs(qd[j])
        peak = max(peak, last)
        if log is not None:
            # The coast IS the measurement; the first version of this script
            # printed it and threw it away, leaving the npz with ramps only.
            log.append(np.r_[time.perf_counter() - t0, coulomb, 0.0,
                             1.0, qq, qd])              # phase 1 = coast
        if np.max(np.abs(qd)) > QD_ABORT or abs(dq[j]) > COAST_BUDGET:
            return qd0, last, peak, float(dq[j]), 'RUNAWAY'
        ctrl.waitPeriod(ts)
    dq = float((np.array(recv.getActualQ()) - q_break)[j])
    ref = max(qd0, 1e-6)
    verdict = ('decelerates' if last < 0.2 * ref
               else 'sustains' if last < 1.2 * ref
               else 'accelerates')
    return qd0, last, peak, dq, verdict


def residual_kinetic(qd0, qd_end, travel, inertia):
    """
    Torque [Nm] opposing motion during the coast, from v^2 = v0^2 - 2*a*d.

    With commanded torque at zero, whatever decelerates the joint IS the
    friction the compensation did not cancel -- which is exactly the f_k that
    belongs in friction.toml for that scale. Negative means the joint was being
    DRIVEN: over-compensation, and no f_k can fix it.

    Viscous drag is lumped in, but at the ~0.02 rad/s of a coast it is a small
    part of the total.
    """
    d = abs(travel)
    if d < 1e-9 or inertia is None:
        return np.nan
    return float(inertia) * (qd0 ** 2 - qd_end ** 2) / (2.0 * d)


def joint_inertia(q, j):
    """Effective inertia [kg m^2] of joint j at q, or None without pinocchio."""
    try:
        import pinocchio as pin
        from kinematics import URKin
        kin = URKin([0.0] * 6)
        H = pin.crba(kin.model, kin.data, np.asarray(q, float))
        H = np.triu(H) + np.triu(H, 1).T
        return float(H[j, j])
    except Exception:
        return None


def report(rows, j, home, inertia=None):
    print('\n' + '=' * 76)
    print(f'  COULOMB SCALE SWEEP -- joint {j}')
    print('=' * 76)
    print(f'  pose (rad): {np.round(home, 6)}')
    print(f'  pendant (deg): {np.round(np.degrees(home), 3)}\n')
    if inertia is not None:
        print(f'  joint {j} effective inertia {inertia:.2f} kg m^2\n')
    print('  coulomb   breakaway + / -         coast +        coast -        '
          'residual kinetic +/- [Nm]')
    limit = None
    resid = {}
    for r in rows:
        p_, m_ = r.get('+'), r.get('-')
        bp = f'{p_[0]:6.2f}' if p_ else '   n/a'
        bm = f'{m_[0]:6.2f}' if m_ else '   n/a'
        vp = p_[5] if p_ else 'n/a'
        vm = m_[5] if m_ else 'n/a'
        unstable = [v for v in (vp, vm)
                    if v in ('sustains', 'accelerates', 'RUNAWAY')]
        if unstable and limit is None:
            limit = r['coulomb']
        mark = '   <-- first unstable' if unstable and limit == r['coulomb'] else ''
        rk = []
        for v in (p_, m_):
            rk.append(np.nan if v is None else
                      residual_kinetic(v[1], v[2], v[4], inertia))
        resid[r['coulomb']] = rk
        rs = '  '.join('  n/a ' if not np.isfinite(x) else f'{x:+6.2f}'
                       for x in rk)
        print(f'   {r["coulomb"]:5.2f}   {bp} / {bm} Nm      '
              f'{vp:12s}  {vm:12s}   {rs}{mark}')

    print()
    if not rows:
        print('  no data')
        return
    if limit is None:
        print('  every scale tested decelerates after breakaway. No '
              'over-compensation in\n  this range AT THIS POSE -- the highest '
              'scale tested is usable, and the\n  threshold is somewhere '
              'above it.')
    safe = [r['coulomb'] for r in rows
            if limit is None or r['coulomb'] < limit]
    if limit is not None:
        print(f'  over-compensation starts at coulomb {limit:.2f}.')
    if safe and limit is not None:
        print(f'  highest scale that still decelerates: {max(safe):.2f}')
        print(f'\n  Run below that, and re-measure f_c there. A breakaway read '
              f'at or above\n  {limit:.2f} is not a friction number -- part of '
              f'it is the scale driving the joint.')
    elif safe:
        print(f'  highest scale tested: {max(safe):.2f}, still stable. Sweep '
              f'higher to find\n  the threshold, or use this one.')
    elif limit is not None and limit > 0.0:
        print('  even the lowest scale tested is unstable; extend the sweep '
              'downward.')
    else:
        # Nothing to extend: 0.0 is the floor. But VISCOUS is still live on
        # this joint, and viscous compensation cancels damping, which is
        # destabilising in its own right.
        print('  unstable at coulomb 0.00 -- so the COULOMB scale is not what '
              'drives it.\n  Viscous is still at '
              f'{VISCOUS[j]:.2f} on this joint and cancelling damping is '
              'destabilising\n  too. Re-run with --viscous 0 to take the '
              'joint fully uncompensated; if it\n  is still unstable there, '
              'the cause is outside the friction scales entirely.')
    if limit is not None and limit <= 0.8:
        print(f'\n  NOTE: 0.8 is the default in test-friction-recal.py, '
              f'test-gravity-residual.py\n  and test-friction-repeat.py. The '
              f'limit is at or below it, so every f_c\n  measured at 0.8 -- '
              f'including the 5.19-8.77 Nm spread in\n  RECALIBRATION-PLAN.md '
              f'-- needs redoing.')
    if inertia is not None and resid:
        print('\n' + '-' * 76)
        print('  f_k FOR friction.toml')
        print('-' * 76)
        usable = {c: v for c, v in resid.items()
                  if (limit is None or c < limit)
                  and any(np.isfinite(x) and x > 0 for x in v)}
        if not usable:
            print('  no stable scale produced a positive residual -- nothing to '
                  'put in f_k.')
        else:
            best = max(usable)
            rp, rm = usable[best]
            print(f'  At the highest STABLE scale tested ({best:.2f}), the coast '
                  f'still opposes\n  motion by {rp:+.2f} / {rm:+.2f} Nm. That '
                  f'residual is what the firmware did not\n  cancel, and it is '
                  f'what belongs in f_k_pos[{j}] / f_k_neg[{j}].')
            print(f'\n  Run the arm at coulomb {best:.2f} on joint {j} and set:')
            print(f'    f_k_pos[{j}] = {max(rp, 0.0):.2f}   '
                  f'f_k_neg[{j}] = {max(rm, 0.0):.2f}')
            print('\n  Under-compensate if in doubt: too large an f_k drives the '
                  'joint, which is\n  the same runaway the scale sweep above is '
                  'looking for.')
    print(f'\n  This is for the pose above. The threshold is load dependent; '
          f're-run it\n  somewhere else before generalising.')


def rows_from_npz(path):
    """
    Rebuild the report's `rows` from a saved run.

    Exists because the report is the last thing main() does, after the npz is
    written -- so a bug there (as in the q_sweep NameError of 2026-09-21) loses
    the printout but never the data. --replay regenerates it without touching
    the robot, and gives the report path a way to be exercised offline, which
    is how that bug reached hardware in the first place.
    """
    d = np.load(path)
    cells, verdicts = d['cells'], d['verdict']
    rows = []
    for c in d['values']:
        row = {'coulomb': float(c)}
        for k in range(len(cells)):
            if not np.isclose(cells[k, 0], c):
                continue
            name = '+' if cells[k, 1] > 0 else '-'
            b = cells[k, 2]
            row[name] = (None if not np.isfinite(b) else
                         (b, cells[k, 3], cells[k, 4], cells[k, 5],
                          cells[k, 6], str(verdicts[k])))
        rows.append(row)
    return rows, int(d['joint']), d['pose']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', default='192.168.0.100')
    ap.add_argument('--script', default='rtde_control-1.6.5-frictionfix.script')
    ap.add_argument('--joint', type=int, default=1)
    ap.add_argument('--pose', default=None,
                    help='6 joint angles [rad]. Default: wherever the arm is '
                         'when you start, which is the point -- put it where '
                         'you want it measured.')
    ap.add_argument('--axis', choices=('coulomb', 'viscous'), default='coulomb',
                    help="which scale to sweep. 'viscous' holds coulomb at "
                         "--hold-coulomb and sweeps viscous instead -- use it "
                         "to ask whether damping bounds a coulomb runaway.")
    ap.add_argument('--hold-coulomb', type=float, default=0.8,
                    help='coulomb scale held fixed when --axis viscous '
                         '(default 0.8, the value that feels light)')
    ap.add_argument('--values', default=None,
                    help='scales to sweep (default 0.0,0.2,0.4,0.6,0.8 for '
                         'coulomb; 0.9,0.7,0.5,0.3,0.0 for viscous)')
    ap.add_argument('--viscous', type=float, default=None,
                    help='viscous scale on the test joint, held fixed across '
                         'the sweep (default: the set value, 0.9 on joint 1). '
                         '0 takes the joint fully uncompensated at coulomb 0.')
    ap.add_argument('--rate', type=float, default=None, help='Nm/s')
    ap.add_argument('--cap', type=float, default=None, help='Nm')
    ap.add_argument('--dwell', type=float, default=2.0,
                    help='s at rest in torque mode before the ramp')
    ap.add_argument('--predwell', type=float, default=1.0,
                    help='s under POSITION control after moveJ, before torque '
                         'mode. moveJ returns on trajectory completion, not on '
                         'the servo settling.')
    ap.add_argument('--qd-quiet', type=float, default=QD_QUIET)
    ap.add_argument('--out', default=None,
                    help='default: scale-sweep-<YYYYmmdd-HHMMSS>.npz')
    ap.add_argument('--replay', default=None,
                    help='re-print the report from a saved .npz without '
                         'touching the robot')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if args.replay:
        rows, j_, pose_ = rows_from_npz(args.replay)
        report(rows, j_, pose_, joint_inertia(pose_, j_))
        return

    j = args.joint
    if args.viscous is not None:
        VISCOUS[j] = args.viscous
    if args.values:
        values = [float(x) for x in args.values.split(',')]
    else:
        values = ([0.0, 0.2, 0.4, 0.6, 0.8] if args.axis == 'coulomb'
                  else [0.9, 0.7, 0.5, 0.3, 0.0])
    rate = RATE[j] if args.rate is None else args.rate
    cap = TAU_CAP[j] if args.cap is None else args.cap
    out = args.out or time.strftime('scale-sweep-%Y%m%d-%H%M%S.npz')

    print(f'{args.axis.upper()} SCALE SWEEP, joint {j}')
    print(f'  values {values}')
    print(f'  scales from friction.toml: viscous {VISCOUS}')
    if args.axis == 'coulomb':
        print(f'  viscous {VISCOUS[j]} held fixed on joint {j}; every other '
              f'joint at scale 0')
    else:
        print(f'  coulomb {args.hold_coulomb} held fixed on joint {j}; every '
              f'other joint at scale 0')
        print(f'  Coulomb over-compensation is a roughly CONSTANT excess '
              f'torque, so with no\n  damping it integrates into acceleration. '
              f'Viscous compensation cancels the\n  damping that would '
              f'otherwise bound it at a terminal velocity. This asks how\n  '
              f'much has to come back before the runaway becomes a bounded '
              f'drift.')
    print(f'  rate {rate} Nm/s   cap {cap} Nm   dwell {args.dwell} s   '
          f'predwell {args.predwell} s')
    print(f'  at breakaway the torque is ZEROED and the joint watched for '
          f'{COAST_WINDOW:.2f} s')
    print(f'    budget {np.degrees(COAST_BUDGET):.1f} deg, hard abort on that '
          f'or {QD_ABORT} rad/s')
    est = len(values) * 2 * (cap / rate * 0.5 + args.dwell + args.predwell + 4)
    print(f'\n  ~{est / 60:.0f} min if nothing aborts.')
    if args.dry_run:
        return

    import rtde_control
    import rtde_receive

    print('\nThis one deliberately lets the joint move after breakaway.')
    print('Put the arm where you want it measured. Hand on the e-stop.')
    input('enter to start, ctrl-C to abort: ')

    ctrl = rtde_control.RTDEControlInterface(args.ip, 500.0)
    recv = rtde_receive.RTDEReceiveInterface(args.ip)
    ctrl.setCustomScriptFile(args.script)
    wait_for_control_script(ctrl)

    home = (np.array([float(x) for x in args.pose.split(',')]) if args.pose
            else np.array(recv.getActualQ()))
    print(f'\nsweeping at the CURRENT configuration'
          if not args.pose else '\nsweeping at the given pose')
    print(f'  pose (rad): {np.round(home, 6)}')
    print(f'  pendant (deg): {np.round(np.degrees(home), 3)}')
    try:
        from kinematics import URKin
        import pinocchio as pin
        kin = URKin([0.0] * 6)
        g = pin.computeGeneralizedGravity(kin.model, kin.data, home)
        print(f'  joint {j} gravity torque (links only, no payload): '
              f'{g[j]:.1f} Nm')
    except Exception:
        pass
    print()

    rows, log = [], []
    try:
        for val in values:
            # `cj` is always the coulomb scale actually applied; when sweeping
            # viscous it is pinned and `val` lands on VISCOUS[j] instead.
            if args.axis == 'coulomb':
                cj = val
            else:
                cj = args.hold_coulomb
                VISCOUS[j] = val
            print(f'  {args.axis} {val:.2f}:')
            row = {'coulomb': val}
            for sgn, name in ((1.0, '+'), (-1.0, '-')):
                approach(ctrl, home, j)
                if args.predwell > 0:
                    time.sleep(args.predwell)
                ctrl.setWatchdog(0.05)
                try:
                    q_start, _, ok = settle(ctrl, recv, j, cj, args.dwell,
                                            qd_quiet=args.qd_quiet)
                    if not ok:
                        print(f'    {name}: would not settle -- skipped')
                        row[name] = None
                        continue
                    b = ramp(ctrl, recv, j, sgn, q_start, rate, cap, cj, log)
                    if b is None:
                        print(f'    {name}: no breakaway below {cap:.1f} Nm')
                        row[name] = None
                        continue
                    q_break = np.array(recv.getActualQ())
                    qd0, qd1, peak, dq, verdict = coast(ctrl, recv, j, cj,
                                                        q_break, log, sgn)
                    print(f'    {name}: breakaway {b:6.3f} Nm   coast '
                          f'{qd0:.4f} -> {qd1:.4f} rad/s '
                          f'(peak {peak:.4f}, {np.degrees(dq):+.2f} deg)   '
                          f'{verdict}')
                    row[name] = (b, qd0, qd1, peak, dq, verdict)
                finally:
                    safe_stop(ctrl)
                time.sleep(0.6)
            rows.append(row)
    except KeyboardInterrupt:
        print('\naborted by user')
    except RuntimeError as e:
        print(f'\nABORT: {e}')
    finally:
        safe_stop(ctrl, 'end of sweep')
        try:
            ctrl.stopScript()
        except Exception:
            pass

    cells, verdicts = [], []
    for r in rows:
        for name, sgn in (('+', 1.0), ('-', -1.0)):
            v = r.get(name)
            if v is None:
                cells.append([r['coulomb'], sgn, np.nan, np.nan, np.nan,
                              np.nan, np.nan])
                verdicts.append('skipped')
            else:
                b, qd0, qd1, pk, dq, vd = v
                cells.append([r['coulomb'], sgn, b, qd0, qd1, pk, dq])
                verdicts.append(vd)
    if log or cells:
        L = np.array(log) if log else np.zeros((0, 15))
        np.savez(out, t=L[:, 0], coulomb=L[:, 1], tau=L[:, 2], phase=L[:, 3],
                 q=L[:, 4:10], qd=L[:, 10:16],
                 cells=np.array(cells, float),
                 cell_cols=np.array(['coulomb', 'sgn', 'breakaway', 'qd0',
                                     'qd_end', 'qd_peak', 'travel']),
                 verdict=np.array(verdicts),
                 values=np.array(values), pose=home, joint=j,
                 viscous=VISCOUS, coulomb_set=COULOMB,
                 coast_window=COAST_WINDOW, coast_budget=COAST_BUDGET)
        print(f'\nwrote {out}')

    report(rows, j, home, joint_inertia(home, j))


if __name__ == '__main__':
    main()
