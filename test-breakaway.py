"""
Per-direction breakaway torque, measured so the numbers can be trusted.

PURPOSE
-------
Produce f_c+ and f_c- per joint for impedance.friction_feedforward. A single
symmetric f_c cannot serve joint 1, which breaks away at roughly 11 Nm one way
and 24 Nm the other: 80% of the weaker direction under-compensates the stronger
one by half, which is why +x teleop converges and -x does not
(offaxis-20260919-165630.npz).

WHY NOT test-friction-repeat.py
-------------------------------
That script's run at 2026-09-19 17:04 produced joints 2 and 3 to under 1%
scatter and joint 1 as garbage: 10.6, then 20.4, then three ramps straight into
the 26 Nm cap. The forensics said why, and all of it is fixed here.

  1. DETECTION WAS A SINGLE SAMPLE. Every trip in that run was exactly one tick
     over QD_DETECT -- 0.0202, 0.0202, 0.0267, 0.0205, 0.0200 rad/s, longest
     run of 1 in every case. A noise spike or a micro-slip registered as
     breakaway.

  2. NO DISPLACEMENT REQUIREMENT. Joint 1's suspect trips moved 0.016-0.018
     deg. Its one clean reading moved 0.197, and joints 2 and 3 moved
     0.32-0.48. Travel separates real breakaway from a spike by a factor of
     ten, and it was not being checked. Here it is the PRIMARY criterion: a
     spike cannot accumulate displacement.

  3. NO CONTROLLED APPROACH. Friction state depends on how the joint was last
     moved. test-scale-sweep.py backs off and comes in identically every time;
     that script did not.

  4. NO RELAXATION. Joint 1's only clean reading was the FIRST measurement of
     the run, before it had been loaded. Every later one followed a ramp that
     wound it to 20-26 Nm for 20-26 s with 0.6 s to recover, and they climbed
     monotonically: 10.6 -> 20.4 -> cap. Temperature was flat (28.8-29.8 C the
     whole run) so it is not thermal. Relaxation here scales with the torque
     actually applied.

  5. THE CAP WAS TOO LOW. 26 Nm is 1.2x the old UNCOMPENSATED 21.87 Nm, and
     joint 1 ran past it three times with 0.004-0.008 deg of travel. Raised,
     and a cap hit is now reported as a cap hit rather than a NaN.

Scales are ZERO on every joint. The coulomb scale does not reduce static
breakaway at all -- 0.7% over 0.0-0.6, see test-scale-sweep.py -- and 0.8 is
past the over-compensation threshold, so there is nothing to gain and stability
to lose.

SAFETY
------
Guarded ramp, same doctrine as the rest: detection is an abort; every stop
zeroes the SCALES first, then the torque, then stopJ; per-joint caps and hard
travel limits. Unlike test-scale-sweep.py this never watches the joint after
breakaway. Runs at whatever pose the arm is in -- put it where you teleoperate,
because breakaway is load dependent.

Hand on the e-stop.
"""
import argparse
import time

import numpy as np

OFF = [0.0] * 6

# Raised from test-friction-repeat.py's [17, 26, 14, 5, 6, 5]. Joint 1 ran past
# 26 Nm three times with no motion, so that limit was measuring the cap, not the
# joint. 40 Nm is ~12% of its 330 Nm rating and still well inside tau_sat.
TAU_CAP = np.array([24.0, 40.0, 20.0, 8.0, 8.0, 8.0])
RATE = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])          # Nm/s

# Breakaway is DISPLACEMENT, not a velocity sample. 0.05 deg sits an order of
# magnitude above the 0.016-0.018 deg of the spurious trips and an order below
# the 0.2-0.48 deg of the real ones.
DQ_CONFIRM = 0.00087     # rad (0.050 deg) -- breakaway
DQ_ONSET = 0.00017       # rad (0.010 deg) -- presliding onset, logged only
QD_DETECT = 0.02         # rad/s, and it must HOLD for DETECT_HOT ticks
DETECT_HOT = 10          # 20 ms at 500 Hz

QD_ABORT = 0.15          # rad/s on ANY joint
DQ_ABORT = 0.026         # rad (1.5 deg) on the joint under test
DQ_OTHER = 0.026         # rad (1.5 deg) on any other joint

QD_QUIET = 0.006         # rad/s -- "stopped" (brr-log-old.npz noise floor)
DQ_SETTLE = 0.035        # rad (2.0 deg) allowed on torque-mode entry
SETTLE_GRACE = 0.15      # s before the settle's velocity abort arms
SETTLE_HOT = 5           # consecutive ticks over QD_ABORT = runaway

APPROACH_BACKOFF = 0.035     # rad (2 deg) detour so every arrival is identical
RELAX_PER_NM = 0.15          # s of unloaded rest per Nm of peak torque applied
RELAX_MIN, RELAX_MAX = 1.0, 6.0


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


def approach(ctrl, q, j):
    """Back off 2 deg on joint j, then come in. Identical history every time."""
    back = np.asarray(q, float).copy()
    back[j] += APPROACH_BACKOFF
    ctrl.moveJ(back.tolist(), 0.3, 0.3)
    ctrl.moveJ(np.asarray(q, float).tolist(), 0.3, 0.3)


def settle(ctrl, recv, dwell, qd_quiet=QD_QUIET, timeout=8.0):
    """
    Enter torque mode at zero torque with every scale at 0, and wait for rest.

    Must happen before q_start is read: moveJ leaves the joint under position
    control and the first directTorque releases it, so the sag lands in the
    measurement otherwise. Returns (q_start, settled).
    """
    q0 = np.array(recv.getActualQ())
    t0 = time.perf_counter()
    quiet_since, hot = None, 0
    while True:
        ts = ctrl.initPeriod()
        ctrl.directTorque(OFF, OFF, OFF)
        qd = np.abs(np.array(recv.getActualQd()))
        dq = np.array(recv.getActualQ()) - q0
        now = time.perf_counter()
        if now - t0 > SETTLE_GRACE:
            hot = hot + 1 if np.max(qd) > QD_ABORT else 0
            if hot >= SETTLE_HOT:
                k = int(np.argmax(qd))
                raise RuntimeError(f'settle: joint {k} at {qd[k]:.3f} rad/s '
                                   f'for {hot} ticks')
        if np.max(np.abs(dq)) > DQ_SETTLE:
            k = int(np.argmax(np.abs(dq)))
            print(f'      will not hold: joint {k} crept '
                  f'{np.degrees(dq[k]):+.2f} deg on entry')
            return np.array(recv.getActualQ()), False
        if np.max(qd) < qd_quiet:
            quiet_since = quiet_since if quiet_since is not None else now
            if now - quiet_since >= dwell:
                return np.array(recv.getActualQ()), True
        else:
            quiet_since = None
        if now - t0 > timeout:
            print(f'      settle timed out')
            return np.array(recv.getActualQ()), False
        ctrl.waitPeriod(ts)


def ramp(ctrl, recv, j, sgn, q_start, rate, cap, log, rep):
    """
    Ramp joint j until it genuinely moves.

    Returns (breakaway, onset, travel, how) where `how` is 'travel', 'speed'
    or 'cap'. Breakaway is DISPLACEMENT-gated: a single velocity sample cannot
    trip it, which is what corrupted joint 1 in the previous script.
    """
    tau = np.zeros(6)
    t0 = time.perf_counter()
    hot = 0
    onset = None
    while True:
        ts = ctrl.initPeriod()
        mag = rate * (time.perf_counter() - t0)
        if mag > cap:
            return None, onset, 0.0, 'cap'
        tau[j] = sgn * mag
        ctrl.directTorque(tau.tolist(), OFF, OFF)

        q = np.array(recv.getActualQ())
        qd = np.array(recv.getActualQd())
        log.append(np.r_[time.perf_counter() - t0, rep, j, sgn * mag, q, qd])
        dq = q - q_start

        if np.max(np.abs(qd)) > QD_ABORT:
            raise RuntimeError(f'joint speed {np.max(np.abs(qd)):.3f} rad/s')
        if abs(dq[j]) > DQ_ABORT:
            raise RuntimeError(f'joint {j} travelled {np.degrees(dq[j]):.2f} deg')
        oth = np.abs(dq).copy()
        oth[j] = 0.0
        k = int(np.argmax(oth))
        if oth[k] > DQ_OTHER:
            raise RuntimeError(f'joint {k} moved {np.degrees(dq[k]):+.2f} deg '
                               f'while joint {j} was under test')

        if onset is None and abs(dq[j]) > DQ_ONSET:
            onset = mag
        if abs(dq[j]) > DQ_CONFIRM:
            return mag, onset, float(dq[j]), 'travel'
        # Sustained speed is a backstop for a breakaway too fast to accumulate
        # displacement first. DETECT_HOT ticks, never one.
        hot = hot + 1 if abs(qd[j]) > QD_DETECT else 0
        if hot >= DETECT_HOT:
            return mag, onset, float(dq[j]), 'speed'
        ctrl.waitPeriod(ts)


def relax(ctrl, recv, peak):
    """
    Unloaded rest scaled to how hard the joint was just driven.

    Joint 1's readings climbed 10.6 -> 20.4 -> cap across the previous run, and
    its only clean value was the first measurement, before anything had loaded
    it. 0.6 s was not enough to let a joint wound to 26 Nm relax.
    """
    t = float(np.clip(RELAX_PER_NM * peak, RELAX_MIN, RELAX_MAX))
    safe_stop(ctrl)
    time.sleep(t)
    return t


def report(res, joints, pose):
    """res[j] = list of (sgn, breakaway|None, onset, travel, how)."""
    print('\n' + '=' * 78)
    print('  BREAKAWAY BY DIRECTION')
    print('=' * 78)
    print(f'  pose (rad): {np.round(pose, 6)}')
    print(f'  pendant (deg): {np.round(np.degrees(pose), 3)}\n')
    fc = {}
    for j in joints:
        rows = res.get(j, [])
        print(f'  --- joint {j} ---')
        print('     dir   breakaway   onset   travel[deg]   how')
        for sgn, b, on_, tr, how in rows:
            bs = f'{b:8.3f}' if b is not None else '  CAP   '
            os_ = f'{on_:7.3f}' if on_ is not None else '   n/a '
            flag = ''
            if how == 'speed':
                flag = '   <- speed-gated, check travel'
            if b is not None and abs(tr) < 2 * DQ_CONFIRM:
                flag = '   <- barely moved, SUSPECT'
            print(f'      {"+" if sgn > 0 else "-"}   {bs}  {os_}   '
                  f'{np.degrees(abs(tr)):8.3f}   {how}{flag}')
        got = {}
        for s in (1.0, -1.0):
            v = [b for sg, b, *_ in rows if sg == s and b is not None]
            got[s] = (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else np.nan,
                      len(v)) if v else None
        if got[1.0] and got[-1.0]:
            p, sp, np_ = got[1.0]
            m, sm, nm = got[-1.0]
            print(f'    +  {p:7.3f} +/- {0.0 if np.isnan(sp) else sp:.3f} '
                  f'(n={np_})     -  {m:7.3f} +/- '
                  f'{0.0 if np.isnan(sm) else sm:.3f} (n={nm})     '
                  f'ratio {p/m:.2f}')
            fc[j] = (0.8 * p, 0.8 * m)
        else:
            missing = [('+' if s > 0 else '-') for s in (1.0, -1.0) if not got[s]]
            print(f'    incomplete: no usable {"/".join(missing)} reading')
        print()

    if not fc:
        print('  nothing usable. Raise --cap, or move to a lower-load pose.')
        return
    print('-' * 78)
    print('  f_c AT 80% OF EACH DIRECTION')
    print('-' * 78)
    for j in sorted(fc):
        p, m = fc[j]
        asym = max(p, m) / min(p, m)
        note = ('symmetric -- one f_c is fine here' if asym < 1.15 else
                f'ASYMMETRIC {asym:.2f}x -- needs f_c+ and f_c- separately')
        print(f'    joint {j}:  f_c+ {p:6.2f}   f_c- {m:6.2f} Nm    {note}')
    print('\n  friction_feedforward currently takes ONE f_c per joint and flips '
          'its sign\n  with tanh(tau_cmd/t_eps); the magnitude does not change '
          'with direction. Any\n  joint flagged ASYMMETRIC above cannot be '
          'served by it as written.')
    print('\n  Remember fc_assist: at standstill the term is f_c * assist * '
          'tanh(...), so\n  the default 0.5 delivers half of whatever goes in.')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', default='192.168.0.100')
    ap.add_argument('--script', default='rtde_control-1.6.5-frictionfix.script')
    ap.add_argument('--joints', default='1,2,3',
                    help='default 1,2,3 -- the only joints that move under a '
                         'teleop command (0/4/5 moved <0.02 deg, see '
                         'test-offaxis.py)')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--pose', default=None, help='6 angles [rad]; default = here')
    ap.add_argument('--rate', type=float, default=None, help='Nm/s')
    ap.add_argument('--cap', type=float, default=None, help='Nm, overrides TAU_CAP')
    ap.add_argument('--dwell', type=float, default=2.0,
                    help='s at rest in torque mode before the ramp')
    ap.add_argument('--predwell', type=float, default=1.0,
                    help='s under POSITION control after moveJ, before torque '
                         'mode; moveJ returns on trajectory completion, not on '
                         'the servo settling')
    ap.add_argument('--out', default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    joints = [int(x) for x in args.joints.split(',')]
    rates = {j: (RATE[j] if args.rate is None else args.rate) for j in joints}
    caps = {j: (TAU_CAP[j] if args.cap is None else args.cap) for j in joints}
    out = args.out or time.strftime('breakaway-%Y%m%d-%H%M%S.npz')

    print(f'BREAKAWAY BY DIRECTION, joints {joints}, {args.reps} reps')
    print('  all friction scales ZERO (coulomb does nothing for breakaway and '
          '0.8 is\n  past the over-compensation threshold)')
    for j in joints:
        print(f'  joint {j}: rate {rates[j]} Nm/s   cap {caps[j]} Nm')
    print(f'  breakaway = {np.degrees(DQ_CONFIRM):.3f} deg of travel, or '
          f'{QD_DETECT} rad/s held for {DETECT_HOT} ticks')
    print(f'  dwell {args.dwell} s in torque mode, predwell {args.predwell} s '
          f'in position mode')
    print(f'  relax {RELAX_PER_NM} s per Nm applied, {RELAX_MIN}-{RELAX_MAX} s')
    n = len(joints) * 2 * args.reps
    est = sum(caps[j] / rates[j] * 0.6 + args.dwell + args.predwell + 6
              for j in joints) * 2 * args.reps / 60
    print(f'\n  {n} ramps, ~{est:.0f} min if nothing aborts.')
    if args.dry_run:
        return

    import rtde_control
    import rtde_receive
    print('\nEvery joint twitches; joint 1 swings the arm. Clear space.')
    input('enter to start, ctrl-C to abort: ')

    ctrl = rtde_control.RTDEControlInterface(args.ip, 500.0)
    recv = rtde_receive.RTDEReceiveInterface(args.ip)
    ctrl.setCustomScriptFile(args.script)
    wait_for_control_script(ctrl)

    home = (np.array([float(x) for x in args.pose.split(',')]) if args.pose
            else np.array(recv.getActualQ()))
    print(f'\npose (rad): {np.round(home, 6)}')
    print(f'    pendant (deg): {np.round(np.degrees(home), 3)}\n')

    res = {j: [] for j in joints}
    log = []
    try:
        for rep in range(args.reps):
            print(f'--- rep {rep} ---')
            for j in joints:
                # Alternate which direction leads so any drift over the run is
                # common-mode between them rather than biasing one.
                order = ((1.0, -1.0) if (rep + j) % 2 == 0 else (-1.0, 1.0))
                for sgn in order:
                    approach(ctrl, home, j)
                    if args.predwell > 0:
                        time.sleep(args.predwell)
                    ctrl.setWatchdog(0.05)
                    peak = caps[j]
                    try:
                        q_start, ok = settle(ctrl, recv, args.dwell)
                        if not ok:
                            print(f'  j{j} {"+-"[sgn < 0]}: would not settle '
                                  f'-- skipped')
                            continue
                        b, on_, tr, how = ramp(ctrl, recv, j, sgn, q_start,
                                               rates[j], caps[j], log, rep)
                        peak = caps[j] if b is None else b
                        res[j].append((sgn, b, on_, tr, how))
                        if b is None:
                            print(f'  j{j} {"+-"[sgn < 0]}: NO breakaway below '
                                  f'{caps[j]:.1f} Nm')
                        else:
                            print(f'  j{j} {"+-"[sgn < 0]}: {b:7.3f} Nm  '
                                  f'(onset {on_ if on_ else float("nan"):.2f}, '
                                  f'travel {np.degrees(abs(tr)):.3f} deg, {how})')
                    finally:
                        t_r = relax(ctrl, recv, peak)
                    print(f'        relaxed {t_r:.1f} s')
    except KeyboardInterrupt:
        print('\naborted by user')
    except RuntimeError as e:
        print(f'\nABORT: {e}')
    finally:
        safe_stop(ctrl, 'end of run')
        try:
            ctrl.stopScript()
        except Exception:
            pass

    if log:
        L = np.array(log)
        flat = [(j, s, (np.nan if b is None else b),
                 (np.nan if o is None else o), t,
                 {'travel': 0, 'speed': 1, 'cap': 2}[h])
                for j in joints for s, b, o, t, h in res[j]]
        np.savez(out, t=L[:, 0], rep=L[:, 1], joint=L[:, 2], tau=L[:, 3],
                 q=L[:, 4:10], qd=L[:, 10:16],
                 results=np.array(flat, float),
                 result_cols=np.array(['joint', 'sgn', 'breakaway', 'onset',
                                       'travel', 'how_0trav_1speed_2cap']),
                 pose=home, joints=np.array(joints),
                 dq_confirm=DQ_CONFIRM, detect_hot=DETECT_HOT)
        print(f'\nwrote {out}')
    report(res, joints, home)


if __name__ == '__main__':
    main()
