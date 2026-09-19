"""
Prerequisite for RECALIBRATION-PLAN.md step 1: how repeatable is a breakaway
measurement at ONE pose?

WHY THIS EXISTS
---------------
Step 1 reports joint 1's residual breakaway spanning 5.19-8.77 Nm over three
poses and calls it load dependence. It might be. But those three poses were
measured once each, with no repeats, so the spread has never been compared
against the scatter of the measurement that produced it. Refitting the
gravity-residual trials against their own model (constant f, constant dm)
leaves 1.70 Nm rms -- 23% of f -- which is the same order as the spread being
attributed to pose.

Everything downstream forks on this number:

  * scatter LARGE   -> the pose spread is mostly noise. Per-pose f_c is fitting
                       noise; the plan's min-across-poses constant is already
                       the right answer and no model, GP or otherwise, can beat
                       it. Spend the robot time on more repeats, not more poses.
  * scatter SMALL   -> there is real pose structure worth modelling, and a
                       campaign over the task poses will resolve it.

Ten ramps at one pose, ~10 minutes. Cheapest decision in the plan.

ONE JOINT AT A TIME, joint 1 by default -- it is the only joint with across-pose
data to compare against, so it is the only one that can answer the question.
`--joints 0,1,2,3,4,5` sweeps the rest, which gives no verdict but does give the
noise floor each joint's step-1 numbers will sit on: a load effect smaller than
roughly 2 sigma there will not be visible however many poses get sampled.

WHAT IS HELD FIXED
------------------
Breakaway is not a constant of the joint; it depends on how it is approached.
Both known confounds are parameters here rather than accidents of timing:

  * `--dwell`  time at rest before the ramp starts. Stiction grows with dwell
               (rate-and-state); the existing scripts use an incidental 0.4-1.0 s.
  * `--rate`   ramp rate. Slower ramps read higher and quantise finer. Default
               is 1.0 Nm/s, matching test-gravity-residual, NOT the 2.0 Nm/s of
               test-friction-recal -- we are trying to resolve a sigma that may
               be a few tenths of a Nm.

Change either and the numbers are not comparable to the ones in the plan.

Directions alternate in order (+- , -+ , +- ...) so that any drift over the run
biases the two directions equally and cancels out of (tau+ - tau-)/2.

Joint temperatures are logged before every ramp. Harmonic-drive friction falls
as the joint warms, and nothing in this repo has ever recorded it; if the
scatter is really a warm-up transient it will show as a trend against
temperature, which is a missing model input rather than noise.

COMPENSATION IS APPLIED TO THE TEST JOINT ONLY
----------------------------------------------
The firmware takes per-joint scale vectors, so only the joint being ramped gets
compensated; the rest stay at scale 0 and keep full natural stiction. Without
this, joint 1 creeps during a ramp on any other joint -- at coulomb 0.8 its
residual stiction is 5-9 Nm, which does not reliably hold it against the gravity
model error. That creep moves the pose, and with it the gravity load on the
joint being measured, which is the one thing a per-pose campaign must hold
fixed. See scales_for(). --compensate-all restores the old behaviour.

SAFETY
------
Same doctrine as test-friction-recal.py and test-gravity-residual.py:

  * Zero commanded torque is NOT a stop while compensation is on. Every abort
    zeroes the SCALES first, restoring natural stiction, then the torque, then
    stopJ to leave torque mode.
  * Detection is an abort, not a checkpoint: the ramp stops on first motion.
  * An abort ends the RUN, not just the repeat -- if the joint got away from us
    once, the answer is not to immediately do it nine more times. Whatever was
    collected is still saved.

TRAVEL LIMITS ARE IN MILLIMETRES, NOT DEGREES
---------------------------------------------
The default pose is near-vertical and close to the wall on the +q1 side. Degrees
are the wrong unit for that clearance: at this pose joint 1 moves the TCP
16.6 mm/deg and joint 5 moves it 0.0. The abort limit is therefore 1.5 deg here
rather than the 3 deg the sibling scripts use -- 25 mm at joint 1 instead of
50 mm, still 3x the 0.5 deg detection threshold, with ~5 mm more from stopJ.
The startup preflight prints the excursion in mm for every joint being probed;
read it before pressing enter, and raise --max-travel only if the mm number says
you have the room.

Hand on the e-stop. Joint 1 swings the arm.
"""
import argparse
import time

import numpy as np

# Scales to MEASURE AGAINST. Must match whatever the resulting f_c will be used
# with: residual breakaway is a property of the joint AND the compensation.
VISCOUS = [0.9, 0.9, 0.8, 0.9, 0.9, 0.9]
COULOMB = [0.8, 0.8, 0.7, 0.8, 0.8, 0.8]
OFF = [0.0] * 6

# Default pose: the bracket-D configuration at q1 = -1.5668 rad, from
# test-gravity-residual.py. Chosen deliberately -- tau_hat = 0 there, so the
# gravity-model error drops out of the +/- pair and the scatter measured is the
# friction's own. It is also one of the three poses already measured
# (f = 8.742 Nm), so the repeats land on top of an existing single sample.
# --here overrides it with wherever the arm already is; see the note there.
#         rad (pendant deg: 114.60, -89.77, -10.625, -79.98, -170.005, -76.83)
BASE_POSE = np.array([2.00014732, -1.56678207, -0.18544123,
                      -1.39591434, -2.96714699, -1.34093646])

# The three single-shot per-pose f values this probe exists to explain, from
# gravity-residual.npz, at q1 = -1.8326 / -1.5668 / -1.3439 rad. The DECISION
# block compares against these, so it only means something at (or near) one of
# them -- hence the default pose, and the warning when --here moves off it.
ACROSS_POSE_F = np.array([8.767, 8.742, 5.192])
ACROSS_POSE_Q1 = np.array([-1.83259571, -1.56678207, -1.34390352])   # rad

TAU_CAP = np.array([17.0, 16.0, 14.0, 5.0, 6.0, 5.0])
RATE_DEFAULT = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])   # Nm/s

QD_DETECT = 0.02     # rad/s  -- breakaway
DQ_DETECT = 0.0087   # rad (0.5 deg) -- breakaway by displacement
QD_ABORT = 0.15      # rad/s on ANY joint
# Tightened from the 0.052 (3 deg) the sibling scripts use. At the default
# pose the arm is near-vertical and joint 1 moves the TCP 16.6 mm/deg, so 3 deg
# is a 50 mm excursion -- too much next to the wall on the +q1 side. 1.5 deg is
# 25 mm, still 3x the 0.5 deg detection threshold, and stopJ(2.0) from the
# 0.15 rad/s abort adds only ~5 mm. --max-travel raises it if you have room.
DQ_ABORT = 0.026     # rad (1.5 deg) on the joint under test
DQ_OTHER = 0.026     # rad (1.5 deg) on any other joint


def preflight(home, joints, dq_abort):
    """
    Print the travel limits as TCP millimetres at this pose.

    Degrees are not the units the wall is in. Joint 1 near-vertical moves the
    TCP 16.6 mm/deg; joint 5 moves it 0. The same 1.5 deg limit is a 25 mm
    excursion on one joint and 0 on another, and that is the number that decides
    whether a pose is safe to probe.
    """
    try:
        from kinematics import URKin
    except Exception as e:
        print(f'  (no kinematics preflight: {e!r})')
        return
    kin = URKin([0.0] * 6)
    J = kin.jacobian(home)
    print(f'  TCP at this pose (m): {np.round(kin.fk(home)[:3], 4)}')
    print(f'  worst-case TCP excursion before abort ({np.degrees(dq_abort):.1f} '
          f'deg + ~5 mm stopJ):')
    for j in joints:
        mm = float(np.linalg.norm(J[:3, j])) * dq_abort * 1000.0
        print(f'    joint {j}: {mm:6.1f} mm   (detect at '
              f'{mm * DQ_DETECT / dq_abort:5.1f} mm)')


def preflight_campaign(poses, joints, dq_abort):
    """One line per pose: the worst TCP excursion an abort can allow there."""
    try:
        from kinematics import URKin
    except Exception as e:
        print(f'  (no kinematics preflight: {e!r})')
        return
    kin = URKin([0.0] * 6)
    print(f'\n  per-pose abort excursion at {np.degrees(dq_abort):.1f} deg '
          f'(worst joint of {joints}):')
    worst = 0.0
    for pi, q in enumerate(poses):
        J = kin.jacobian(q)
        mm = {j: float(np.linalg.norm(J[:3, j])) * dq_abort * 1000 for j in joints}
        jw = max(mm, key=mm.get)
        worst = max(worst, mm[jw])
        print(f'    pose {pi:2d}: TCP {np.round(kin.fk(q)[:3], 3)}   '
              f'{mm[jw]:5.1f} mm (joint {jw})')
    print(f'  worst across the campaign: {worst:.1f} mm')


def scales_for(j, isolate=True):
    """
    (viscous, coulomb) 6-vectors to send during a ramp on joint j.

    The firmware applies these PER JOINT (see rtde_control-1.6.5-frictionfix
    .script line 1102), so compensation can be enabled on the joint under test
    and left OFF everywhere else. That is the default, and it matters:

    Torque mode is gravity-compensated float, so a joint at rest is only
    fighting the gravity MODEL error -- but at coulomb 0.8 joint 1's residual
    stiction is just 5-9 Nm, which is not always enough to hold it. It creeps,
    which is the "drifts to an attractor" behaviour in test-gravity-residual.py.
    During a ramp on some other joint that creep is not a nuisance, it is
    corruption: the pose moves, so the gravity load on the joint being measured
    moves with it, and the campaign's whole premise is that load is fixed per
    pose. Zeroing the other joints' scales restores their full natural stiction
    (11.9-21.9 Nm on joint 1) and pins the pose.

    This does not change what is measured. UR's compensation is a per-joint
    feedforward on that joint's own velocity, so joint j's breakaway does not
    depend on joint k's scale except through the pose -- which is exactly what
    this is protecting. --compensate-all restores the old all-joints behaviour
    if you want to A/B it.
    """
    if not isolate:
        return list(VISCOUS), list(COULOMB)
    v, c = [0.0] * 6, [0.0] * 6
    v[j], c[j] = VISCOUS[j], COULOMB[j]
    return v, c


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


def temperatures(recv):
    """Six joint temperatures [C], or NaNs if this controller will not report."""
    try:
        return np.array(recv.getJointTemperatures(), float)
    except Exception:
        return np.full(6, np.nan)


def ramp(ctrl, recv, j, sgn, q_start, rate, tau_cap, log, rep, dq_abort,
         isolate=True):
    """Ramp joint j until it moves. Returns breakaway [Nm], or None at the cap."""
    tau = np.zeros(6)
    vis, cou = scales_for(j, isolate)
    t0 = time.perf_counter()
    while True:
        ts = ctrl.initPeriod()
        mag = rate * (time.perf_counter() - t0)
        if mag > tau_cap:
            return None
        tau[j] = sgn * mag
        ctrl.directTorque(tau.tolist(), vis, cou)

        q = np.array(recv.getActualQ())
        qd = np.array(recv.getActualQd())
        log.append(np.r_[time.perf_counter() - t0, rep, sgn, sgn * mag, q, qd])

        dq = q - q_start
        if np.max(np.abs(qd)) > QD_ABORT:
            raise RuntimeError(f'joint speed {np.max(np.abs(qd)):.3f} rad/s')
        if abs(dq[j]) > dq_abort:
            raise RuntimeError(f'joint {j} travelled {np.degrees(dq[j]):.1f} deg')
        oth = np.abs(dq).copy()
        oth[j] = 0.0
        k = int(np.argmax(oth))
        if oth[k] > DQ_OTHER:
            raise RuntimeError(
                f'joint {k} moved {np.degrees(dq[k]):+.2f} deg while joint {j} '
                f'was under test (limit {np.degrees(DQ_OTHER):.1f}); '
                f'all dq = {np.round(np.degrees(dq), 2)}')

        if abs(qd[j]) > QD_DETECT or abs(dq[j]) > DQ_DETECT:
            return mag
        ctrl.waitPeriod(ts)


def detrend_sigma(y, x):
    """(sigma after removing a linear trend in x, slope). NaNs dropped."""
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 3 or np.ptp(x[m]) < 1e-9:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x[m], y[m], 1)
    resid = y[m] - (slope * x[m] + intercept)
    return float(np.std(resid, ddof=2)), float(slope)


def sigma_ci(s, n, conf=0.95):
    """
    Two-sided CI on a standard deviation from n samples (chi-square).

    This matters more than it looks. At n = 10 the 95% CI on sigma spans roughly
    0.69x to 1.83x the estimate, so a hard threshold on a single sigma estimate
    is overconfident by a factor of two in each direction. The verdict below
    uses the bounds, not the point estimate.
    """
    from scipy.stats import chi2
    if not np.isfinite(s) or n < 2:
        return np.nan, np.nan
    a = (1.0 - conf) / 2.0
    dof = n - 1
    return (s * np.sqrt(dof / chi2.ppf(1 - a, dof)),
            s * np.sqrt(dof / chi2.ppf(a, dof)))


def report(res, temps, t_start, joint, on_ref=True):
    """res: (n, 2) breakaway magnitudes [+, -]; temps: (n, 6) before each repeat."""
    n = len(res)
    if n == 0:
        print('\nno complete repeats')
        return

    plus, minus = res[:, 0], res[:, 1]
    # Both are magnitudes, so this is the MEAN of the two directions -- the same
    # quantity the plan calls f. See the note at the end about why it is not the
    # weaker direction.
    f = (plus + minus) / 2.0
    asym = (plus - minus) / 2.0

    print('\n' + '=' * 62)
    print(f'  {n} repeats at one pose, joint {joint}')
    print('=' * 62)
    print('\n  rep   tau+      tau-      f=mean    asym     T_joint')
    for i in range(n):
        tj = temps[i, joint]
        tstr = f'{tj:9.1f}' if np.isfinite(tj) else '      n/a'
        print(f'  {i:3d} {plus[i]:8.3f} {minus[i]:9.3f} {f[i]:9.3f} '
              f'{asym[i]:8.3f} {tstr}')

    def line(name, v):
        ok = v[np.isfinite(v)]
        if len(ok) < 2:
            print(f'  {name:12s} n < 2')
            return np.nan
        s = float(np.std(ok, ddof=1))
        print(f'  {name:12s} mean {ok.mean():7.3f}   sigma {s:6.3f} Nm   '
              f'({100 * s / abs(ok.mean()):4.1f}%)   range {np.ptp(ok):6.3f}')
        return s

    print()
    line('tau+', plus)
    line('tau-', minus)
    sigma_f = line('f (mean dir)', f)
    line('asymmetry', asym)

    # --- drift vs random scatter -------------------------------------------
    # A trend against temperature is a MISSING INPUT, not noise: it can be
    # modelled. A trend against time with no temperature signal is something
    # else, and needs finding before any f_c is trusted.
    s_t, k_t = detrend_sigma(f, t_start)
    s_T, k_T = detrend_sigma(f, temps[:, joint])
    print()
    if np.isfinite(s_t):
        print(f'  vs elapsed time : slope {k_t * 60:+.3f} Nm/min   '
              f'sigma after detrend {s_t:.3f} Nm')
    if np.isfinite(s_T):
        print(f'  vs T_joint      : slope {k_T:+.3f} Nm/degC   '
              f'sigma after detrend {s_T:.3f} Nm')
        if np.isfinite(sigma_f) and s_T < 0.6 * sigma_f:
            m = np.isfinite(temps[:, joint])
            rho = (np.corrcoef(t_start[m], temps[m, joint])[0, 1]
                   if m.sum() > 2 else np.nan)
            print('    -> a trend explains most of the scatter.')
            if np.isfinite(rho) and abs(rho) > 0.95:
                print(f'       BUT time and T are collinear here (r = {rho:+.3f}):'
                      f' this run\n       cannot tell warm-up from anything else '
                      f'that drifts with it.\n       Re-run from cold, or '
                      f'randomise the pose order, to separate them.')
            else:
                print('       Temperature is a missing model input: log it in '
                      'test-friction-recal.py\n       and carry it as a '
                      'regressor.')
    elif not np.any(np.isfinite(temps[:, joint])):
        print('  vs T_joint      : no temperature reported by this controller')
    else:
        print(f'  vs T_joint      : held at '
              f'{np.nanmean(temps[:, joint]):.1f} C over the run -- no spread '
              f'to regress\n                    against, so warm-up is '
              f'untested, not ruled out')

    # --- the decision -------------------------------------------------------
    if not np.isfinite(sigma_f):
        return
    lo, hi = sigma_ci(sigma_f, n)

    # ACROSS_POSE_F is joint 1 only -- it came from the gravity-residual run,
    # which swept q1 alone. For every other joint there is nothing to compare
    # against yet, so sigma is reported as a noise floor and no verdict is
    # given: it is the resolution the step-1 campaign will have on that joint.
    if joint != 1:
        print('\n' + '-' * 62)
        print(f'  NOISE FLOOR, joint {joint}')
        print('-' * 62)
        print(f'  within-pose sigma : {sigma_f:.3f} Nm   '
              f'(95% CI {lo:.3f} - {hi:.3f}, n = {n})')
        print(f'  no across-pose data for this joint, so no verdict. This is '
              f'the\n  resolution step 1 will have here: a load effect smaller '
              f'than about\n  {2 * hi:.2f} Nm will not be visible above it.')
        return

    s_across = float(np.std(ACROSS_POSE_F, ddof=1))

    print('\n' + '-' * 62)
    print('  DECISION')
    print('-' * 62)
    print(f'  within-pose sigma : {sigma_f:.3f} Nm   '
          f'(95% CI {lo:.3f} - {hi:.3f}, n = {n})')
    print(f'  across-pose sigma : {s_across:.3f} Nm   '
          f'(n = 3 poses, ONE shot each, gravity-residual.npz)')
    if not on_ref:
        print('  [run was not at one of those three poses -- the comparison '
              'assumes the\n   scatter measured here also holds there]')

    # The null is NOT "ratio near zero". If pose did not matter at all, the
    # three single-shot poses would still scatter by sigma_f, so the expected
    # ratio under the null is 1. The question is whether the across-pose
    # variance is larger than the within-pose variance can account for: an
    # F test with (2, n-1) dof.
    from scipy.stats import f as f_dist
    F = (s_across ** 2) / (sigma_f ** 2)
    p = float(f_dist.sf(F, 2, n - 1))
    F_crit = float(f_dist.ppf(0.95, 2, n - 1))
    sigma_crit = s_across / np.sqrt(F_crit)
    print(f'  F = {F:.2f} on (2, {n - 1}) dof   p = {p:.3f}   '
          f'(F_crit = {F_crit:.2f})')
    print(f'  -> structure is only detectable at this n if within-pose sigma '
          f'< {sigma_crit:.2f} Nm')

    # Same question, better posed: is the odd pose out actually out?
    delta = float(np.median(ACROSS_POSE_F) - ACROSS_POSE_F.min())
    z, z_cons = delta / sigma_f, delta / hi
    print(f'\n  the {ACROSS_POSE_F.min():.2f} Nm pose sits {delta:.2f} Nm below '
          f'the other two:\n    {z:.1f} sigma at the point estimate, '
          f'{z_cons:.1f} sigma at the conservative end of the CI')

    print()
    if p < 0.05 and z_cons > 3.0:
        print('  REAL STRUCTURE. The pose spread survives the conservative end '
              'of the\n  sigma CI. A load model is worth the campaign: sample '
              'the task poses and\n  fit against load torque (kinematics.URKin '
              'gives it free), not raw q.')
    elif p > 0.20 and z < 2.0:
        print('  NOT DISTINGUISHABLE from measurement scatter. Per-pose f_c '
              'would be\n  fitting noise, and no model -- GP, linear or table '
              '-- can beat the\n  min-across-poses constant the plan already '
              'uses. Spend robot time on\n  repeats, not on more poses.')
    else:
        print('  INCONCLUSIVE. The two are the same order, or the sigma CI is '
              'too wide\n  to commit. n is the cheap axis here: repeats cost '
              'minutes, a pose\n  campaign costs hours.')
        need = int(np.ceil(1 + 2 * (1.83 * sigma_f / sigma_crit) ** 2)) if \
            sigma_crit > 0 else 0
        if sigma_f < sigma_crit < 1.83 * sigma_f:
            print(f'  The point estimate is under the threshold but the CI is '
                  f'not -- roughly\n  n = {max(need, n + 5)} repeats would '
                  f'tighten it enough to decide.')

    print('\n  NOTE: (tau+ + tau-)/2 is the MEAN of the two directions, not the'
          '\n  weaker one. f+ and f- are not separately identifiable from a +/-'
          '\n  pair alone -- the pair gives (f+ + f-)/2 and (f+ - f-)/2 - g,'
          '\n  where g is the gravity-model error. The "80% of the weaker'
          '\n  direction" rule at impedance.py:63 needs g from somewhere else'
          '\n  (the friction-free equilibrium fit) before it can be applied.'
          '\n  At this default pose tau_hat = 0, so g is small here by design.')


def load_poses(path):
    """Radians, one pose per line, '#' comments. Returns (N, 6)."""
    P = []
    for ln, raw in enumerate(open(path), 1):
        txt = raw.split('#')[0].strip().rstrip(',')
        if not txt:
            continue
        v = [float(x) for x in txt.replace('[', '').replace(']', '').split(',')]
        if len(v) != 6:
            raise ValueError(f'{path}:{ln}: expected 6 angles, got {len(v)}')
        P.append(v)
    if not P:
        raise ValueError(f'{path}: no poses')
    return np.array(P, float)


def gravity_torque(q, payload=1.551, com=(0.006, 0.006, 0.048)):
    """Joint gravity torque [Nm], links + payload. NaNs if pinocchio is absent."""
    try:
        import pinocchio as pin
        from kinematics import URKin
    except Exception:
        return np.full(6, np.nan)
    kin = URKin([0.0] * 6)
    M, D = kin.model, kin.data
    pin.computeJointJacobians(M, D, q)
    pin.updateFramePlacements(M, D)
    Rt = D.oMf[kin.f_tool].rotation.copy()
    Jf = pin.getFrameJacobian(M, D, kin.f_tool,
                              pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
    Jc = Jf[:3] - np.cross(Rt @ np.asarray(com), Jf[3:], axisa=0, axisb=0).T
    g = np.array([0.0, 0.0, -1.0])
    return pin.computeGeneralizedGravity(M, D, q) + Jc.T @ (payload * 9.81 * g)


def report_campaign(res, pose_idx, jcol, poses, joints):
    """
    Nested variance decomposition, per joint: is the between-pose spread larger
    than the within-pose scatter accounts for?

    This replaces the single-pose DECISION block, which borrowed its across-pose
    number from three single shots in gravity-residual.npz. Here both halves are
    measured in the same campaign, which is the whole point of running poses.
    """
    from scipy.stats import f as f_dist
    f = (res[:, 0] + res[:, 1]) / 2.0
    tg = np.array([gravity_torque(q) for q in poses])

    for j in joints:
        sel = jcol == j
        if not sel.any():
            continue
        print('\n' + '=' * 70)
        print(f'  JOINT {j}')
        print('=' * 70)
        print('  pose   tau_g [Nm]    n    mean f [Nm]   sd')
        means, ns, ss, loads = [], [], [], []
        for pi in range(len(poses)):
            m = sel & (pose_idx == pi)
            y = f[m][np.isfinite(f[m])]
            if len(y) == 0:
                continue
            sd = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
            print(f'  {pi:4d} {tg[pi, j]:10.1f} {len(y):6d}  {y.mean():10.3f}'
                  + (f' {sd:8.3f}' if np.isfinite(sd) else '      n/a'))
            means.append(y.mean()); ns.append(len(y)); loads.append(tg[pi, j])
            if len(y) > 1:
                ss.append((len(y) - 1) * np.var(y, ddof=1))
        means = np.array(means); ns = np.array(ns); loads = np.array(loads)
        N = len(means)
        dof_w = int(ns.sum() - N)
        if N < 2 or dof_w < 1:
            print('  not enough data for a decomposition')
            continue

        ms_w = float(np.sum(ss) / dof_w)
        gm = float(np.average(means, weights=ns))
        ms_b = float(np.sum(ns * (means - gm) ** 2) / (N - 1))
        F = ms_b / ms_w if ms_w > 0 else np.inf
        pv = float(f_dist.sf(F, N - 1, dof_w))
        R_bar = float(ns.mean())
        s_w = np.sqrt(ms_w)
        s_b = np.sqrt(max(0.0, (ms_b - ms_w) / R_bar))

        print(f'\n  within-pose sigma  {s_w:.3f} Nm  ({dof_w} dof)')
        print(f'  between-pose sigma {s_b:.3f} Nm  ({N - 1} dof)')
        print(f'  F = {F:.2f} on ({N - 1}, {dof_w})   p = {pv:.4f}')

        # Is the between-pose part actually explained by load?
        if N >= 3 and np.all(np.isfinite(loads)) and np.ptp(loads) > 1e-6:
            k, b = np.polyfit(loads, means, 1)
            pred = k * loads + b
            ss_tot = float(np.sum((means - means.mean()) ** 2))
            r2 = 1.0 - float(np.sum((means - pred) ** 2)) / ss_tot if ss_tot > 0 else np.nan
            print(f'  vs gravity load: {k * 10:+.3f} Nm friction per 10 Nm load, '
                  f'R^2 = {r2:.2f}')
            if np.ptp(loads) < 1e-3:
                print('    (no load variation at this joint -- pose cannot '
                      'excite it; see joints 0 and 5)')
        else:
            r2 = np.nan
            print('  vs gravity load: no load variation across these poses')

        print()
        if pv < 0.05 and s_b > 0.5 * s_w:
            if np.isfinite(r2) and r2 > 0.5:
                print(f'  REAL, AND LOAD EXPLAINS IT (R^2 = {r2:.2f}). Fit f_c '
                      f'against load\n  torque. This is the case a GP earns its '
                      f'keep in: use the posterior\n  lower bound, not the mean.')
            else:
                print('  REAL BUT NOT LOAD-SHAPED. Something varies with pose '
                      'that is not\n  gravity torque -- check temperature drift '
                      'and pose ORDER before\n  believing it (see the campaign '
                      'ordering note).')
        else:
            print('  NOT DISTINGUISHABLE from within-pose scatter. A per-pose '
                  'f_c here would\n  be fitting noise; keep the '
                  'min-across-poses constant for this joint.')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', default='192.168.0.100')
    ap.add_argument('--script', default='rtde_control-1.6.5-frictionfix.script')
    ap.add_argument('--joints', default='1',
                    help='comma-separated. Default 1 -- the only joint with '
                         'across-pose data to compare against.')
    ap.add_argument('--pose', default=None,
                    help='6 joint angles [rad]; default = the bracket-D pose')
    ap.add_argument('--here', action='store_true',
                    help='use the current configuration instead of --pose. '
                         'Avoids the transit moveJ, but moves off the poses '
                         'ACROSS_POSE_F was measured at.')
    ap.add_argument('--poses', default=None,
                    help='file of poses [rad], one per line. Switches to '
                         'campaign mode: --repeats passes over ALL poses, each '
                         'pass in a fresh random order, so pose order is not '
                         'time order.')
    ap.add_argument('--seed', type=int, default=0,
                    help='pose-shuffle seed; the realised order is saved')
    ap.add_argument('--repeats', type=int, default=10,
                    help='repeats per pose (campaign mode: passes over the set)')
    ap.add_argument('--rate', type=float, default=None, help='Nm/s')
    ap.add_argument('--cap', type=float, default=None, help='Nm')
    ap.add_argument('--dwell', type=float, default=2.0,
                    help='seconds at rest before each ramp; stiction grows with it')
    ap.add_argument('--max-travel', type=float, default=DQ_ABORT,
                    help='rad on the joint under test')
    ap.add_argument('--coulomb', default=','.join(str(c) for c in COULOMB))
    ap.add_argument('--viscous', default=','.join(str(v) for v in VISCOUS))
    # Timestamped by default. A campaign is many invocations and a fixed name
    # silently overwrites the previous one; the pooled analysis wants them all.
    ap.add_argument('--out', default=None,
                    help='default: friction-repeat-<YYYYmmdd-HHMMSS>.npz')
    ap.add_argument('--compensate-all', action='store_true',
                    help='apply the friction scales to ALL joints during a '
                         'ramp, not just the one under test. The old behaviour; '
                         'lets the other joints creep and moves the pose.')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    out = args.out or time.strftime('friction-repeat-%Y%m%d-%H%M%S.npz')
    if args.poses and (args.here or args.pose):
        ap.error('--poses is exclusive with --here / --pose')
    joints = [int(x) for x in args.joints.split(',')]
    dq_abort = args.max_travel
    COULOMB[:] = [float(x) for x in args.coulomb.split(',')]
    VISCOUS[:] = [float(x) for x in args.viscous.split(',')]
    rates = {j: (RATE_DEFAULT[j] if args.rate is None else args.rate)
             for j in joints}
    caps = {j: (TAU_CAP[j] if args.cap is None else args.cap) for j in joints}

    print(f'breakaway REPEATABILITY: joints {joints}, {args.repeats} '
          + ('passes over the pose set' if args.poses else 'repeats at one pose'))
    if args.compensate_all:
        print(f'  viscous {VISCOUS}\n  coulomb {COULOMB}   (ALL joints)')
        print('  !! other joints will creep and move the pose -- see scales_for()')
    else:
        print('  scales applied to the joint under test ONLY; every other joint '
              'is left\n  at scale 0 so its natural stiction pins the pose:')
        for j in joints:
            print(f'    joint {j}: viscous {VISCOUS[j]}  coulomb {COULOMB[j]}')
    for j in joints:
        print(f'  joint {j}: rate {rates[j]} Nm/s   cap {caps[j]} Nm')
    print(f'  dwell {args.dwell} s')
    print(f'  abort: {QD_ABORT} rad/s any joint, {dq_abort:.4f} rad '
          f'({np.degrees(dq_abort):.1f} deg) travel')
    POSES = load_poses(args.poses) if args.poses else None
    npose = 1 if POSES is None else len(POSES)
    est = npose * sum(args.repeats * 2 * (caps[j] / rates[j] * 0.5 + args.dwell + 3)
                      for j in joints) / 60.0
    if POSES is not None:
        print(f'  CAMPAIGN: {npose} poses x {args.repeats} passes, '
              f'pose order reshuffled each pass (seed {args.seed})')
    print(f'\n  ~{est:.0f} min ({est / 60:.1f} h) if nothing aborts.')

    rng = np.random.default_rng(args.seed)
    if POSES is not None:
        plan = [(pas, int(pi)) for pas in range(args.repeats)
                for pi in rng.permutation(len(POSES))]
        preflight_campaign(POSES, joints, dq_abort)
        print('\n  pose order (pass:pose): ' +
              ' '.join(f'{a}:{b}' for a, b in plan))
    else:
        plan = [(r, None) for r in range(args.repeats)]
        if args.here and args.dry_run:
            print('\n  (--here needs the robot; preflight skipped in --dry-run)')
        elif not args.here:
            home_preview = (np.array([float(x) for x in args.pose.split(',')])
                            if args.pose else BASE_POSE.copy())
            print()
            preflight(home_preview, joints, dq_abort)

    # These guards have never been exercised on hardware: test-friction-recal.py
    # has not been run, and this script inherits its abort paths. The first real
    # run is also their first test.
    if args.repeats > 1 or POSES is not None:
        print('\n  NOTE: the abort paths here are inherited from '
              'test-friction-recal.py,\n  which has never been run. The first '
              'real run is also their first test.')
        if POSES is not None:
            print('  A campaign is the wrong place to find that out -- shake '
                  'them out first with\n    --repeats 1 --joints 1   (one pose, '
                  'two ramps, ~30 s)')
        else:
            print(f'  Consider --repeats 1 first, before committing to '
                  f'{args.repeats}.')
    if args.dry_run:
        return

    import rtde_control
    import rtde_receive

    print('\nThe joint twitches on every ramp; joints 1 and 2 swing the arm.')
    print('Clear space, hand on the e-stop.')
    input('enter to start, ctrl-C to abort: ')

    ctrl = rtde_control.RTDEControlInterface(args.ip, 500.0)
    recv = rtde_receive.RTDEReceiveInterface(args.ip)
    ctrl.setCustomScriptFile(args.script)
    wait_for_control_script(ctrl)

    on_ref = True
    if POSES is not None:
        home = POSES[0].copy()          # logged only; the plan drives the run
    else:
        if args.here:
            home = np.array(recv.getActualQ())
        elif args.pose:
            home = np.array([float(x) for x in args.pose.split(',')])
        else:
            home = BASE_POSE.copy()
        # Degrees for the teach pendant only; everything stored is radians.
        print(f'\npose (rad): {np.round(home, 6)}')
        print(f'    pendant (deg): {np.round(np.degrees(home), 3)}')
        if args.here:
            print()
            preflight(home, joints, dq_abort)
        on_ref = float(np.abs(home[1] - ACROSS_POSE_Q1).min()) < np.radians(2.0)
        if not on_ref:
            print('\n  NOTE: not at one of the poses ACROSS_POSE_F was measured '
                  'at.\n  The within-pose sigma is still valid, but the DECISION '
                  'block then assumes\n  the scatter here also holds there.')
    print()

    res, temps, t_start, jcol, pcol, log = [], [], [], [], [], []
    t_run = time.perf_counter()

    # Campaign mode: PASSES over the whole pose set, reshuffled each pass, so
    # that pose is not confounded with elapsed time (and therefore with joint
    # warm-up). Visiting each pose once per pass and coming back is the whole
    # reason this loop is nested this way -- do not "optimise" it into
    # all-repeats-at-one-pose, which is exactly the confound it avoids.
    try:
        for pas, pi in plan:
            here = home if pi is None else POSES[pi]
            label = f'pass {pas}' + (f'  pose {pi}' if pi is not None else '')
            T = temperatures(recv)
            tj = ' '.join(f'{T[j]:.0f}' for j in joints if np.isfinite(T[j]))
            print(f'--- {label}' + (f'   T {tj} C' if tj else '') + ' ---')
            for j in joints:
                # Alternate which direction leads, so drift is common-mode in
                # (tau+ - tau-)/2 rather than biasing it.
                order = (((1.0, 0), (-1.0, 1)) if (pas + (pi or 0)) % 2 == 0
                         else ((-1.0, 1), (1.0, 0)))
                pair = np.full(2, np.nan)
                temps.append(temperatures(recv))
                t_start.append(time.perf_counter() - t_run)
                jcol.append(j)
                pcol.append(-1 if pi is None else pi)
                for sgn, k in order:
                    ctrl.moveJ(here.tolist(), 0.3, 0.3)
                    time.sleep(args.dwell)
                    q_start = np.array(recv.getActualQ())
                    ctrl.setWatchdog(0.05)
                    try:
                        b_ = ramp(ctrl, recv, j, sgn, q_start, rates[j], caps[j],
                                  log, pas, dq_abort,
                                  isolate=not args.compensate_all)
                    finally:
                        safe_stop(ctrl)
                    if b_ is None:
                        print(f'    j{j} {"+-"[k]} : NO breakaway below '
                              f'{caps[j]:.1f} Nm')
                    else:
                        pair[k] = b_
                        print(f'    j{j} {"+-"[k]} : {b_:6.3f} Nm')
                    time.sleep(0.6)
                res.append(pair)
    except KeyboardInterrupt:
        print('\naborted by user')
    except RuntimeError as e:
        print(f'\nABORT: {e}')
        print('run stopped -- a joint got away from us once; not repeating it')
    finally:
        safe_stop(ctrl, 'end of run')
        try:
            ctrl.stopScript()
        except Exception:
            pass

    res = np.array(res, float).reshape(-1, 2)
    temps = np.array(temps, float).reshape(-1, 6)
    t_start = np.array(t_start, float)
    jcol = np.array(jcol, int)
    pcol = np.array(pcol, int)

    if log:
        L = np.array(log)
        np.savez(out, t=L[:, 0], rep=L[:, 1], sgn=L[:, 2], tau=L[:, 3],
                 q=L[:, 4:10], qd=L[:, 10:16], breakaway=res, temps=temps,
                 t_start=t_start, res_joint=jcol, res_pose=pcol,
                 poses=(np.zeros((0, 6)) if POSES is None else POSES),
                 plan=np.array([[a_, b_] for a_, b_ in plan
                                if b_ is not None], dtype=int).reshape(-1, 2),
                 seed=args.seed, pose=home, joints=np.array(joints),
                 dwell=args.dwell,
                 rate=np.array([rates.get(j, np.nan) for j in range(6)]),
                 cap=np.array([caps.get(j, np.nan) for j in range(6)]),
                 viscous=VISCOUS, coulomb=COULOMB)
        print(f'\nwrote {out}')

    ok = np.all(np.isfinite(res), axis=1)
    if POSES is not None:
        report_campaign(res[ok], pcol[ok], jcol[ok], POSES, joints)
    else:
        for j in joints:
            m = ok & (jcol == j)
            if m.any():
                report(res[m], temps[m], t_start[m], j, on_ref and j == 1)


if __name__ == '__main__':
    main()
