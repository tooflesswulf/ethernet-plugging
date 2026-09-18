"""
Measure the gravity-model error on joint 1: the phantom mass dm and the residual
stiction f, from breakaway torques at known angles.

WHY THIS EXISTS
---------------
With the patched control script (firmware friction compensation live), joint 1
drifts to an attractor and sits there. That attractor is where the gravity-model
error torque vanishes. Bracketing it located the error mass at

    p = [-0.0377, -0.0537, -0.0951] m   in tool0

but the equilibrium condition is magnitude-independent -- dm cancels out of it.
This routine supplies the missing equation.

At a fixed pose, joint 1 is held by stiction while

    | dm*tau_hat(q) + tau_cmd | < f

so ramping tau_cmd to breakaway in each direction gives

    tau+ =  f - dm*tau_hat        ->  dm = -(tau+ + tau-) / (2*tau_hat)
    tau- = -f - dm*tau_hat            f  =  (tau+ - tau-) / 2

Both unknowns, one pose. Three poses over-determines them.

SAFETY -- READ THIS
-------------------
`identify` in test-impedance.py accelerated joint 1 into a wall under exactly
these conditions. Two things caused it and both are handled here:

  * Zero commanded torque is NOT a stop when friction compensation is on. The
    firmware keeps pushing a moving joint. Every abort path here zeroes the
    friction SCALES as well, restoring full natural stiction, and then calls
    stopJ -- which is non-realtime, so it moves the command register off cmd 66
    and hands the joint back to the controller's own position control.
  * After breakaway the gravity residual keeps driving the joint. Detection is
    therefore an abort, not a checkpoint: the ramp stops on first motion.

Hand on the e-stop. Clear at least 30 cm around the arm in the shoulder's plane.
"""
import argparse
import time

import numpy as np
import pinocchio as pin

from kinematics import URKin

# Located by bracketing the drift attractor at three configurations (see
# brr-log*.npz). 16-84% spread is +/-1-2 cm per axis.
P_ERR = np.array([-0.0377, -0.0537, -0.0951])

# Test pose. q1 is overridden per trial; the rest is the bracket-D configuration,
# chosen because its Jacobian row is 0.93-aligned with the direction the earlier
# brackets could not constrain.
BASE_POSE_DEG = np.array([114.60, -89.77, -10.625, -79.98, -170.005, -76.83])

# tau_hat = 0 at -89.77 deg, so that angle measures f alone; the off-equilibrium
# angles are what separate dm.
TRIAL_DEG = (-105.0, -89.77, -77.0)

# MUST match the scales used for the bracket measurements -- f is the residual
# stiction under a particular compensation setting, not a property of the joint.
VISCOUS = [0.9, 0.9, 0.8, 0.9, 0.9, 0.9]
COULOMB = [0.8, 0.8, 0.7, 0.8, 0.8, 0.8]
OFF = [0.0] * 6

RATE = 1.0          # Nm/s ramp. Slow: breakaway torque is read off the ramp, so
                    # rate sets the quantisation of the answer.
TAU_CAP = 8.0       # Nm. Predicted breakaways are ~3 Nm; this is 2.5x headroom
                    # and far below joint 1's rating.
QD_DETECT = 0.02    # rad/s -- breakaway
DQ_DETECT = 0.0087  # rad (0.5 deg) -- breakaway by displacement
QD_ABORT = 0.15     # rad/s on ANY joint -- hard abort
DQ_ABORT = 0.052    # rad (3 deg) on joint 1 -- hard abort. Must stay well under
                    # the clearance on the +q1 side: the `+` ramp pushes toward
                    # the wall, and this is the backstop if detection is missed.
                    # Detection fires at 0.5 deg, so 3 deg is pure margin.
DQ_OTHER = 0.052    # rad (3 deg) on any other joint -- pose no longer valid


def tau_hat(kin, q):
    """Joint-1 gravity torque per kg of error mass at P_ERR [Nm/kg]."""
    gh = np.array([0.0, 0.0, -1.0])
    q = np.asarray(q, float)
    pin.computeJointJacobians(kin.model, kin.data, q)
    pin.updateFramePlacements(kin.model, kin.data)
    J = pin.getFrameJacobian(kin.model, kin.data, kin.f_tool,
                             pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    R = kin.data.oMf[kin.f_tool].rotation
    return (J[:3][:, 1] @ gh + (R.T @ np.cross(gh, J[3:][:, 1])) @ P_ERR) * 9.81


def safe_stop(ctrl, why=''):
    """
    Zero torque is not enough. Kill the friction compensation first so natural
    stiction comes back and helps arrest the joint, then leave torque mode.
    """
    if why:
        print(f'    STOP: {why}')
    try:
        for _ in range(5):
            ctrl.directTorque(OFF, OFF, OFF)
        ctrl.stopJ(2.0)
    except Exception as e:
        print(f'    !! safe_stop failed: {e!r}')


def wait_for_control_script(ctrl, timeout=5.0, poll=0.01):
    """reuploadScript() does not call waitForProgramRunning(); see env.py."""
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


def ramp(ctrl, recv, sgn, q_start, dt, log, dq_abort):
    """
    Ramp joint-1 torque at RATE until the joint moves. Returns breakaway torque
    or None if it reached TAU_CAP. Any abort condition raises.
    """
    tau = np.zeros(6)
    t0 = time.perf_counter()
    while True:
        ts = ctrl.initPeriod()
        mag = RATE * (time.perf_counter() - t0)
        if mag > TAU_CAP:
            return None
        tau[1] = sgn * mag
        ctrl.directTorque(tau.tolist(), VISCOUS, COULOMB)

        q = np.array(recv.getActualQ())
        qd = np.array(recv.getActualQd())
        log.append(np.r_[time.perf_counter() - t0, mag * sgn, q, qd])

        dq = q - q_start
        if np.max(np.abs(qd)) > QD_ABORT:
            raise RuntimeError(f'joint speed {np.max(np.abs(qd)):.3f} > {QD_ABORT}')
        if abs(dq[1]) > dq_abort:
            raise RuntimeError(f'joint 1 travelled {np.degrees(dq[1]):.1f} deg')
        other = np.max(np.abs(np.delete(dq, 1)))
        if other > DQ_OTHER:
            raise RuntimeError(f'another joint moved {np.degrees(other):.1f} deg')

        if abs(qd[1]) > QD_DETECT or abs(dq[1]) > DQ_DETECT:
            return mag
        ctrl.waitPeriod(ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', default='192.168.0.100')
    ap.add_argument('--script', default='rtde_control-1.6.5-frictionfix.script')
    ap.add_argument('--angles', default=','.join(str(a) for a in TRIAL_DEG),
                    help='joint-1 test angles [deg]')
    ap.add_argument('--out', default='gravity-residual.npz')
    ap.add_argument('--max-travel', type=float, default=np.degrees(DQ_ABORT),
                    help='joint-1 travel [deg] that aborts a ramp')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the plan and predicted torques, move nothing')
    args = ap.parse_args()
    angles = [float(a) for a in args.angles.split(',')]
    dq_abort = np.radians(args.max_travel)

    kin = URKin(np.zeros(6))
    print(f'error mass position p = {P_ERR} m (tool0)\n')
    print('  q1 [deg]   tau_hat [Nm/kg]   expected tau+ / tau- at dm=1kg, f=0.6')
    for a in angles:
        q = np.radians(BASE_POSE_DEG.copy())
        q[1] = np.radians(a)
        th = tau_hat(kin, q)
        print(f'  {a:8.2f}   {th:+9.4f}        {0.6 - th:+7.3f} / {-0.6 - th:+7.3f}')
    if args.dry_run:
        return

    # Imported here so --dry-run works on a machine without ur_rtde.
    import rtde_control
    import rtde_receive

    print(f'\nramp {RATE} Nm/s, cap {TAU_CAP} Nm, abort at {QD_ABORT} rad/s '
          f'or {np.degrees(dq_abort):.1f} deg of joint-1 travel')
    print('The arm WILL move at each breakaway and the gravity residual keeps')
    print('pushing afterwards. Clear space, hand on the e-stop.')
    input('enter to start, ctrl-C to abort: ')

    ctrl = rtde_control.RTDEControlInterface(args.ip, 500.0)
    recv = rtde_receive.RTDEReceiveInterface(args.ip)
    ctrl.setCustomScriptFile(args.script)
    wait_for_control_script(ctrl)
    dt = ctrl.getStepTime() or 0.002

    res, log = [], []
    try:
        for a in angles:
            q_tgt = np.radians(BASE_POSE_DEG.copy())
            q_tgt[1] = np.radians(a)
            th = tau_hat(kin, q_tgt)
            print(f'\n=== q1 = {a:.2f} deg,  tau_hat = {th:+.4f} Nm/kg')
            pair = {}
            for sgn, name in ((+1.0, '+'), (-1.0, '-')):
                ctrl.moveJ(q_tgt.tolist(), 0.3, 0.3)
                time.sleep(0.5)
                q_start = np.array(recv.getActualQ())
                ctrl.setWatchdog(0.05)
                try:
                    b = ramp(ctrl, recv, sgn, q_start, dt, log, dq_abort)
                finally:
                    safe_stop(ctrl)
                if b is None:
                    print(f'  {name}: no breakaway below {TAU_CAP} Nm')
                else:
                    pair[name] = sgn * b
                    print(f'  {name}: breakaway {sgn * b:+7.3f} Nm')
                time.sleep(1.0)
            if '+' in pair and '-' in pair:
                f = (pair['+'] - pair['-']) / 2.0
                dm = (-(pair['+'] + pair['-']) / (2.0 * th)) if abs(th) > 0.05 else np.nan
                res.append((a, th, pair['+'], pair['-'], f, dm))
                print(f'  -> f = {f:.3f} Nm', end='')
                print(f',  dm = {dm:+.3f} kg' if np.isfinite(dm)
                      else ',  dm not identifiable here (tau_hat ~ 0, by design)')
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
        np.savez(args.out, t=L[:, 0], tau=L[:, 1], q=L[:, 2:8], qd=L[:, 8:14],
                 trials=np.array(res, float) if res else np.zeros((0, 6)),
                 p_err=P_ERR, base_pose=BASE_POSE_DEG,
                 viscous=VISCOUS, coulomb=COULOMB)
        print(f'\nwrote {args.out}  ({len(L)} samples, {len(res)} complete pairs)')

    if res:
        R = np.array(res, float)
        print('\n   q1[deg]  tau_hat    tau+      tau-       f[Nm]   dm[kg]')
        for r in R:
            print(f'  {r[0]:8.2f} {r[1]:+8.3f} {r[2]:+8.3f} {r[3]:+8.3f} '
                  f'{r[4]:8.3f} {r[5]:+8.3f}')
        ok = R[np.isfinite(R[:, 5])]
        if len(ok):
            print(f'\n  dm = {ok[:, 5].mean():+.3f} +/- {ok[:, 5].std():.3f} kg'
                  f'   (declared payload 1.67 kg)')
        print(f'  f  = {R[:, 4].mean():.3f} +/- {R[:, 4].std():.3f} Nm'
              f'   (bracket D implied f/dm = 0.59 Nm/kg)')


if __name__ == '__main__':
    main()
