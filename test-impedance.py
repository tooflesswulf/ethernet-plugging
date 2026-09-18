#!/usr/bin/env python3
"""
Quick standalone impedance-control probe. Not wired into Env -- this is the
minimal thing that answers "is the control law right and are the frames right?"
before any of it goes into env.py.

Run in this order:

    python test-impedance.py frames     # no torque at all. hand-move the arm.
    python test-impedance.py hold       # impedance hold at the current pose
    python test-impedance.py step --axis z --dist 0.02

`frames` MUST pass before `hold` is run. It validates the base-frame convention,
joint ordering, [v; w] row ordering and the TCP offset in one shot -- if it fails,
the commanded force points the wrong way.

Deliberately minimal: no watchdog state machine, no runaway detection, no fault
latch. The only things kept are a hard per-joint torque clip and the robot-side
watchdog, because sending unbounded torque to a real arm is not a thing to omit
for convenience. Keep a hand on the e-stop.
"""
from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse
import time

import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description
import rtde_control
import rtde_receive

ROBOT_IP = "192.168.0.100"
ROBOT_DESC = "ur16e_description"   # UR16e: effort [330,330,150,54,54,54] Nm


def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def pose_to_se3(p):
    return pin.SE3(R.from_rotvec(p[3:]).as_matrix(), np.asarray(p[:3], float))


class URKin:
    """
    pinocchio UR wrapped so that everything it returns is in the frames the UR
    controller actually reports in: the `base` frame (NOT `base_link`, which is
    the ROS REP-103 convention and is rotated 180 deg about Z), and the TCP
    (NOT `flange`, which shares tool0's origin but not its orientation).

    The robot description must match the actual arm. A UR5e model on a UR16e is
    76 mm of TCP error on average (131 mm worst case), which looks like a frame
    problem but is not one.
    """

    _cache = {}

    def __init__(self, tcp_offset, desc=ROBOT_DESC, base="base", tool="tool0"):
        if desc not in URKin._cache:
            URKin._cache[desc] = load_robot_description(desc).model
        self.model = URKin._cache[desc]
        self.data = self.model.createData()
        self.desc = desc
        self.base_name, self.tool_name = base, tool
        self.f_base = self.model.getFrameId(base)
        self.f_tool = self.model.getFrameId(tool)
        self.tcp = pose_to_se3(tcp_offset)
        # Rated joint torques straight from the URDF -- no hardcoded table.
        self.tau_rated = np.asarray(self.model.effortLimit, float)

    def _update(self, q):
        pin.forwardKinematics(self.model, self.data, np.asarray(q, float))
        pin.updateFramePlacements(self.model, self.data)

    def fk(self, q):
        """base -> TCP, as a UR pose vector [x, y, z, rx, ry, rz]."""
        self._update(q)
        bMt = self.data.oMf[self.f_base].inverse() * self.data.oMf[self.f_tool] * self.tcp
        return np.r_[bMt.translation, R.from_matrix(bMt.rotation).as_rotvec()]

    def jacobian(self, q):
        """
        6x6 geometric Jacobian at the TCP, expressed in the `base` frame.
        Rows are [vx vy vz wx wy wz], columns are joints 1..6.
        """
        q = np.asarray(q, float)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        J = pin.getFrameJacobian(
            self.model, self.data, self.f_tool, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ).copy()

        # Shift the reference point from tool0 to the TCP: v_tcp = v_tool + w x r
        r = self.data.oMf[self.f_tool].rotation @ self.tcp.translation
        J[:3, :] -= skew(r) @ J[3:, :]

        # Rotate from the URDF root into the UR `base` frame.
        Rb = self.data.oMf[self.f_base].rotation
        return np.block([[Rb.T, np.zeros((3, 3))], [np.zeros((3, 3)), Rb.T]]) @ J


def friction_feedforward(qd, tau_cmd, f_c, v_eps, t_eps):
    """
    Per-joint Coulomb friction compensation, needed because ur_rtde 1.6.5's
    control script never applies the viscous/coulomb scales it reads:

        viscous_scaling = [0,0,0,0,0,0]          # never assigned
        viscous_scale   = q_from_input_float_registers(6)
        direct_torque(torque, viscous_scale=viscous_scaling, ...)

    so friction compensation is off no matter what directTorque() is passed.

    A pure tanh(qd) Coulomb term cannot break away from rest -- at qd = 0 it is
    identically 0, which is exactly the stiction case we care about. So blend:
    use velocity direction while moving, and fall back to the direction of the
    commanded torque while stationary.

    Under-compensate (f_c below the true breakaway torque). Over-compensation
    turns stiction into a limit cycle, which is worse than a deadband.
    """
    s_v = np.tanh(qd / v_eps)
    s_t = np.tanh(tau_cmd / t_eps)
    return f_c * (s_v + (1.0 - np.abs(s_v)) * s_t)


def pose_error(actual, desired):
    """
    6-vector [dp; drotvec] in the base frame, pointing from actual to desired.

    Orientation must be R_des @ R_act.T (left-multiplied) so the error rotvec
    lives in the base frame and pairs with a base-frame Jacobian. Never subtract
    rotvec components: home_pose's (2.44, 2.44, 0.653) has norm 3.512 > pi, so
    componentwise subtraction against a canonicalised pose gives a ~2pi error.
    """
    actual, desired = np.asarray(actual, float), np.asarray(desired, float)
    R_act = R.from_rotvec(actual[3:])
    R_des = R.from_rotvec(desired[3:])
    return np.r_[desired[:3] - actual[:3], (R_des * R_act.inv()).as_rotvec()]


def make_torque_fn(ctrl, viscous=None, coulomb=None):
    """
    directTorque's extra args are version-dependent:
      1.6.3  directTorque(torque, friction_comp: bool)               <- works
      1.6.5  directTorque(torque, viscous_scale[], coulomb_scale[])  <- script drops them
    Bind the right call once instead of branching in the 500 Hz loop.
    """
    doc = ctrl.directTorque.__doc__ or ''
    if 'viscous' in doc:
        if viscous is None and coulomb is None:
            print('directTorque : 1.6.5 scale-vector form -- NOTE its control script '
                  'never applies the scales, so friction comp is off')
            return ctrl.directTorque
        vs = [viscous if viscous is not None else v for v in (.9, .9, .8, .9, .9, .9)]
        cs = [coulomb if coulomb is not None else c for c in (.8, .8, .7, .8, .8, .8)]
        print(f'directTorque : scale-vector form, viscous {vs} coulomb {cs}')
        return lambda t: ctrl.directTorque(t, vs, cs)
    print('directTorque : friction_comp=True (1.6.3 form)')
    return lambda t: ctrl.directTorque(t, True)


def parse_fc(args, tau_rated):
    """Per-joint Coulomb torques [Nm]. Explicit --fc-nm wins over scalar --fc."""
    if getattr(args, 'fc_nm', None):
        v = np.array([float(x) for x in args.fc_nm.split(',')], float)
        if v.size != 6:
            raise SystemExit('--fc-nm needs 6 comma-separated values')
        return v
    return args.fc * tau_rated


def connect(kin=True):
    # NOTE: do NOT pass FLAG_UPPER_RANGE_REGISTERS -- it hangs construction on
    # this controller. It is only needed for getJacobian()/getMassMatrix(), and
    # we compute the Jacobian locally anyway.
    F = rtde_control.RTDEControlInterface.Flags
    ctrl = rtde_control.RTDEControlInterface(ROBOT_IP, flags=F.FLAG_UPLOAD_SCRIPT)
    ctrl.setCustomScriptFile('rtde_control-1.6.5-frictionfix.script')

    recv = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    tcp_offset = ctrl.getTCPOffset()
    step = ctrl.getStepTime()
    print(f'TCP offset : {np.round(tcp_offset, 5)}')
    print(f'step time  : {step} s')
    print(f'payload    : {recv.getPayload()} kg   cog {np.round(recv.getPayloadCog(), 4)}')
    print(f'prog running: {ctrl.isProgramRunning()}   connected: {ctrl.isConnected()}')

    if step <= 0:
        print('\n!! getStepTime() returned 0. The control script is probably not')
        print('!! running, which means directTorque() will also fail silently.')
        print('!! Assuming 0.002 s, but fix this before commanding any torque.\n')
        step = 0.002

    return ctrl, recv, tcp_offset, step, (URKin(tcp_offset) if kin else None)


# ======================================================================
# 1. frames -- no torque commanded, hand-move the arm
# ======================================================================
def cmd_frames(args):
    """
    Collect (q, pose, qd, twist) samples, then work out empirically which frame
    convention the controller is actually using instead of assuming one.
    """
    ctrl, recv, tcp_offset, _, _ = connect(kin=False)

    print('\nEntering freedrive. Move the arm slowly through several DIFFERENT poses')
    print('-- vary all 6 joints, not just one. Sampling for %.0f s...\n' % args.duration)

    S = []
    try:
        ctrl.teachMode()
        t_end = time.time() + args.duration
        last = 0.0
        while time.time() < t_end:
            q = np.array(recv.getActualQ())
            qd = np.array(recv.getActualQd())
            S.append((q, qd,
                      np.array(recv.getActualTCPPose()),
                      np.array(recv.getActualTCPSpeed())))
            if time.time() - last > 2.0:
                last = time.time()
                print(f'  {len(S):4d} samples, {t_end - time.time():4.0f}s left', end='\r')
            time.sleep(0.02)
    finally:
        ctrl.endTeachMode()

    moving = [s for s in S if np.linalg.norm(s[1]) > 0.05]
    print(f'\n{len(S)} samples, {len(moving)} while moving\n')
    if len(S) < 20:
        print('Too few samples.')
        return

    # ---- 1. try every frame convention, score FK and J independently --------
    print(f'{"base":<10} {"tool":<8} {"FK pos [mm]":>12} {"J@qd vs v [mm/s]":>18} {"J@qd vs w [mrad/s]":>20}')
    print('-' * 72)
    results = {}
    for base in ('base', 'base_link'):
        for tool in ('tool0', 'flange'):
            try:
                k = URKin(tcp_offset, desc=args.robot, base=base, tool=tool)
            except Exception as e:
                print(f'{base:<10} {tool:<8}  unavailable ({e})')
                continue
            fk_e = [np.abs(k.fk(q)[:3] - p[:3]).max() for q, _, p, _ in S]
            if moving:
                ve, we = zip(*[(np.abs((k.jacobian(q) @ qd)[:3] - t[:3]).max(),
                                np.abs((k.jacobian(q) @ qd)[3:] - t[3:]).max())
                               for q, qd, _, t in moving])
            else:
                ve, we = (np.nan,), (np.nan,)
            results[(base, tool)] = (np.mean(fk_e), np.mean(ve), np.mean(we))
            print(f'{base:<10} {tool:<8} {np.mean(fk_e)*1e3:12.3f} '
                  f'{np.mean(ve)*1e3:18.3f} {np.mean(we)*1e3:20.3f}')

    best = min(results, key=lambda k: results[k][0])
    print(f'\nbest FK match: base={best[0]!r} tool={best[1]!r}  '
          f'({results[best][0]*1000:.3f} mm)')

    # ---- 2. if nothing matches, fit the residual rotation (Kabsch) ----------
    # If the only error is a wrong base frame, actual = R_fit @ model exactly,
    # and R_fit tells us precisely which rotation is missing.
    k = URKin(tcp_offset, desc=args.robot, base=best[0], tool=best[1])
    P = np.array([k.fk(q)[:3] for q, _, _, _ in S])       # model
    Q = np.array([p[:3] for _, _, p, _ in S])             # controller
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R_fit = Vt.T @ np.diag([1, 1, d]) @ U.T
    resid = np.abs((R_fit @ P.T).T - Q).max()
    rv = R.from_matrix(R_fit).as_rotvec()
    ang = np.linalg.norm(rv)

    passed = results[best][0] < 2e-3 and results[best][1] < 5e-3
    if not passed:
        # Only meaningful when something is wrong: a large angle with a small
        # residual means the whole error is one rigid rotation, i.e. a frame
        # convention. On a passing run this just absorbs sub-mm calibration
        # error into a meaningless tiny angle, so don't show it.
        print(f'\nbest-fit rotation model->controller:')
        print(f'  angle {np.degrees(ang):7.2f} deg  about {np.round(rv/ang, 4) if ang > 1e-9 else "n/a"}')
        print(f'  residual after applying it: {resid*1000:.3f} mm')

    if passed:
        print(f'\nPASS -- use base={best[0]!r} tool={best[1]!r}. Safe to try `hold`.')
        print(f'  Residual {results[best][0]*1000:.2f} mm is per-robot delta-DH calibration,')
        print(f'  which the controller applies and the nominal URDF does not carry.')
        print(f'  Expected at ~1 mm; it is ~0.1% of reach and irrelevant to J^T F.')
        if best != ('base', 'tool0'):
            print(f'\n  !! Non-default frames -- pass --base {best[0]} --tool {best[1]} to hold/step.')
    elif resid < 2e-3 and ang > 1e-3:
        print(f'\nMISMATCH IS A PURE ROTATION of {np.degrees(ang):.2f} deg.')
        print('  ~180 deg about z  -> base frame convention; switch base_link <-> base.')
        print('  ~90 deg about z   -> the arm is mounted rotated; add it as a fixed offset.')
        print('  anything else     -> non-standard mounting; use this rotvec as the offset.')
    else:
        print('\nFAIL and the residual is NOT a pure rotation, so it is not just a')
        print('frame convention. Check, in order:')
        print(f'  - is {args.robot!r} the right arm? a model mismatch dominates')
        print(f'    everything else -- e.g. a UR5e model on a UR16e is ~76 mm mean,')
        print(f'    131 mm max, and looks exactly like this. Try --robot')
        print(f'  - TCP offset {np.round(tcp_offset, 4)} -- does it match the pendant?')
        print(f'  - joint ordering/signs from getActualQ()')
        print(f'\n  per-axis mean signed FK error [mm]: {np.round((Q - P).mean(0)*1000, 2)}')
        print(f'  model |p| mean {np.linalg.norm(P,axis=1).mean():.4f} m vs '
              f'controller {np.linalg.norm(Q,axis=1).mean():.4f} m  '
              f'(ratio {np.linalg.norm(Q,axis=1).mean()/np.linalg.norm(P,axis=1).mean():.4f})')


# ======================================================================
# 1b. identify -- measure per-joint breakaway (Coulomb) torque
# ======================================================================
def cmd_identify(args):
    """
    Ramp one joint's torque until it breaks free, every other joint at zero
    torque (gravity-compensated float, so the arm holds pose). The torque at
    first motion IS that joint's static friction -- measured, not inferred from
    rated torque, which is what a single --fc scalar gets wrong: over-compensating
    some joints into a limit cycle while others still have a deadband.

    Both directions, because Coulomb friction is usually asymmetric.
    """
    ctrl, recv, _, dt, _ = connect(kin=False)
    torque_cmd = make_torque_fn(ctrl)

    print(f'\nramp {args.rate} Nm/s, cap {args.cap} Nm, breakaway at '
          f'|qd| > {args.qd_thresh} rad/s')
    print('The arm WILL twitch at each breakaway. Clear space, hand on the e-stop.')
    # Prompt BEFORE arming the watchdog: once armed, any pause longer than
    # 1/min_frequency stops the robot with "fieldbus interrupted".
    input('enter to start, ctrl-C to abort: ')
    ctrl.setWatchdog(args.watchdog)

    res = np.full((6, 2), np.nan)
    try:
        for j in range(6):
            for k, sgn in enumerate((1.0, -1.0)):
                tau = np.zeros(6)
                q0 = recv.getActualQ()[j]
                t0 = time.perf_counter()
                while True:
                    ts = ctrl.initPeriod()
                    mag = args.rate * (time.perf_counter() - t0)
                    if mag > args.cap:
                        print(f'  joint {j} {"+-"[k]} : no breakaway below {args.cap} Nm')
                        break
                    tau[j] = sgn * mag
                    torque_cmd(tau.tolist())
                    if (abs(recv.getActualQd()[j]) > args.qd_thresh
                            or abs(recv.getActualQ()[j] - q0) > 0.02):
                        res[j, k] = mag
                        print(f'  joint {j} {"+-"[k]} : breakaway {mag:6.2f} Nm')
                        break
                    ctrl.waitPeriod(ts)
                # Keep streaming while settling -- any gap trips the watchdog.
                for _ in range(100):
                    ts = ctrl.initPeriod()
                    torque_cmd([0.0] * 6)
                    ctrl.waitPeriod(ts)
    except KeyboardInterrupt:
        print('\naborted')
    finally:
        for _ in range(5):
            torque_cmd([0.0] * 6)
        ctrl.stopJ(2.0)
        ctrl.stopScript()

    print('\n       joint :  ' + '  '.join(f'{j:6d}' for j in range(6)))
    print('  breakaway + :  ' + '  '.join(f'{v:6.2f}' for v in res[:, 0]))
    print('  breakaway - :  ' + '  '.join(f'{v:6.2f}' for v in res[:, 1]))
    mean = np.nanmean(res, axis=1)
    lo = np.nanmin(res, axis=1)
    asym = np.abs(res[:, 0] - res[:, 1]) / np.fmax(mean, 1e-9)
    if np.all(np.isnan(mean)):
        return
    print('  mean        :  ' + '  '.join(f'{v:6.2f}' for v in mean))
    print('  asymmetry   :  ' + '  '.join(f'{v:5.0%} ' for v in asym))

    # Compensation must stay under the SMALLER of the two directions. Using the
    # mean over-compensates the weak direction on an asymmetric joint, which is
    # exactly what produces a limit cycle in one direction while the opposite
    # direction still has a deadband.
    use = np.nan_to_num(lo, nan=0.0) * args.frac
    print(f'\nCompensating {args.frac:.0%} of the weaker direction (not the mean):')
    print('  --fc-nm ' + ','.join(f'{v:.2f}' for v in use))

    bad = np.where(asym > 0.3)[0]
    if bad.size:
        print(f'\n  NOTE joints {list(bad)} are >30% asymmetric. Half the +/- gap is a')
        print('  residual gravity-compensation error, not friction:')
        for j in bad:
            print(f'    joint {j}: ~{abs(res[j,0]-res[j,1])/2:.2f} Nm bias')
        print('  Gravity-loaded joints (shoulder/elbow) reading asymmetric usually')
        print('  means the payload mass or CoG is wrong. Worth checking -- that bias')
        print('  is a constant torque the impedance controller has to fight.')


# ======================================================================
# 2/3. impedance hold, and a setpoint step
# ======================================================================
def impedance_loop(args, step_delta=None):
    ctrl, recv, tcp_offset, dt, _ = connect(kin=False)
    kin = URKin(tcp_offset, desc=args.robot, base=args.base, tool=args.tool)
    print(f'frames     : base={args.base!r} tool={args.tool!r}')

    # Refuse to command torque if the kinematics do not match the controller --
    # a wrong frame here points the commanded force the wrong way.
    pose = np.array(recv.getActualTCPPose())
    fk_err = np.abs(kin.fk(recv.getActualQ())[:3] - pose[:3]).max()
    print(f'FK check   : {fk_err*1000:.3f} mm')
    if fk_err > 2e-3:
        print('\nABORT: kinematics disagree with the controller. Run `frames` first.')
        return

    K = np.r_[np.full(3, args.kp), np.full(3, args.kr)]
    D = np.r_[np.full(3, args.dp), np.full(3, args.dr)]
    tau_max = args.tau_frac * kin.tau_rated
    f_c = parse_fc(args, kin.tau_rated)

    print(f'\nK        : {K}')
    print(f'D        : {D}')
    print(f'tau clip : {np.round(tau_max, 1)} Nm  ({args.tau_frac:.0%} of rated)')

    torque_cmd = make_torque_fn(ctrl, args.viscous, args.coulomb)
    print(f'coulomb ff : {np.round(f_c, 2)} Nm')
    print(f'watchdog   : {args.watchdog} Hz '
          f'({1000.0/args.watchdog:.0f} ms max gap before "fieldbus interrupted")')

    eq = pose.copy()                            # equilibrium latched here, never moves
    print(f'equilibrium: {np.round(eq, 4)}')
    if step_delta is not None:
        print(f'stepping after {args.settle:.1f} s by {step_delta}')
    print('\nCtrl-C to stop.\n')

    # Arm last, immediately before streaming starts: nothing may block after this.
    ctrl.setWatchdog(args.watchdog)

    xd_f = np.zeros(6)
    t0 = time.perf_counter()
    ticks, late = 0, 0
    t_prev = t0
    stepped = False

    try:
        while True:
            t_start = ctrl.initPeriod()
            now = time.perf_counter() - t0

            if step_delta is not None and not stepped and now > args.settle:
                eq = eq + step_delta
                stepped = True
                print(f'\n[{now:6.2f}s] stepped equilibrium -> {np.round(eq, 4)}')

            q = np.array(recv.getActualQ())
            qd = np.array(recv.getActualQd())
            pose = np.array(recv.getActualTCPPose())
            twist = np.array(recv.getActualTCPSpeed())

            # Damping must come from velocity, NOT from the force sensor, and it
            # must be filtered fast. force_alpha=0.03 (~2.4 Hz) here would
            # reintroduce exactly the phase lag that makes admittance chatter.
            xd_f = args.vel_alpha * twist + (1 - args.vel_alpha) * xd_f

            # Ramp gains in over the first ramp seconds so entry is bumpless.
            ramp = min(1.0, now / args.ramp) if args.ramp > 0 else 1.0

            e = pose_error(pose, eq)
            e[3:] = np.clip(e[3:], -0.2, 0.2)          # rotvec axis is ill-conditioned near pi

            F = ramp * (K * e - D * xd_f)
            J = kin.jacobian(q)
            tau = J.T @ F - args.dq * qd
            if np.any(f_c > 0):
                tau = tau + ramp * friction_feedforward(
                    qd, tau, f_c, args.fc_veps, args.fc_teps)

            if not np.isfinite(tau).all():
                print('\nnon-finite torque, aborting')
                break
            tau = np.clip(tau, -tau_max, tau_max)

            ok = torque_cmd(tau.tolist())
            if ok is False:
                print('\ndirectTorque() returned False -- command rejected.')
                break

            ticks += 1
            if (time.perf_counter() - t_prev) > 3 * dt:
                late += 1
            t_prev = time.perf_counter()

            if ticks % 250 == 0:
                print(f'  t={now:6.2f}s  |e_p|={np.linalg.norm(e[:3])*1000:6.2f}mm  '
                      f'|F|={np.linalg.norm(F[:3]):6.2f}N  |tau|={np.abs(tau).max():6.2f}Nm  '
                      f'clip={"Y" if np.any(np.abs(tau) >= tau_max - 1e-9) else "n"}  '
                      f'rate={ticks/now:5.0f}Hz  late={late}', end='\r')

            ctrl.waitPeriod(t_start)
    except KeyboardInterrupt:
        print('\n\ninterrupted')
    finally:
        # directTorque is re-applied by the controller every cycle while the
        # command register still holds it, so a stale torque does NOT decay on
        # its own. Zero it, then stopJ to leave torque mode entirely.
        try:
            for _ in range(5):
                ctrl.directTorque([0.0] * 6)
            ctrl.stopJ(2.0)
        finally:
            ctrl.stopScript()
        el = time.perf_counter() - t0
        print(f'stopped. {ticks} ticks in {el:.1f}s = {ticks/max(el,1e-9):.0f} Hz, {late} late')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    f = sub.add_parser('frames', help='validate frame conventions, no torque')
    f.add_argument('--duration', type=float, default=20.0)
    f.add_argument('--robot', default=ROBOT_DESC,
                   help='robot_descriptions name; must match the actual arm')

    def add_gains(sp):
        sp.add_argument('--kp', type=float, default=200., help='translational stiffness [N/m]')
        sp.add_argument('--kr', type=float, default=10., help='rotational stiffness [Nm/rad]')
        sp.add_argument('--dp', type=float, default=90., help='translational damping [Ns/m]')
        sp.add_argument('--dr', type=float, default=2., help='rotational damping [Nms/rad]')
        sp.add_argument('--dq', type=float, default=0.5, help='joint damping floor [Nms/rad]')
        sp.add_argument('--vel-alpha', type=float, default=0.4)
        sp.add_argument('--tau-frac', type=float, default=0.20, help='fraction of rated joint torque')
        sp.add_argument('--ramp', type=float, default=0.5)
        sp.add_argument('--fc', type=float, default=0.0,
                        help='Coulomb friction feedforward, as a fraction of each '
                             'rated joint torque. 0 = off. Try 0.005 and raise until '
                             'the deadband closes; back off if it buzzes or creeps.')
        sp.add_argument('--watchdog', type=float, default=10.0,
                        help='robot-side watchdog [Hz]. Stops the robot if RTDE '
                             'updates stall for longer than 1/this. Higher is '
                             'safer but 50 Hz leaves only 20 ms, which Python GC '
                             'or slow terminal I/O can exceed.')
        sp.add_argument('--fc-nm', default=None,
                        help='per-joint Coulomb torques [Nm], 6 comma-separated, '
                             'from `identify`. Overrides --fc. A single scalar '
                             'cannot work -- real breakaway friction does not '
                             'scale with rated torque.')
        sp.add_argument('--fc-veps', type=float, default=0.02,
                        help='joint speed [rad/s] at which Coulomb comp saturates')
        sp.add_argument('--fc-teps', type=float, default=2.0,
                        help='torque [Nm] at which the stationary breakaway assist saturates')
        sp.add_argument('--viscous', type=float, default=None,
                        help='viscous friction-compensation scale, all joints '
                             '(default: library values ~0.9). 0 disables.')
        sp.add_argument('--coulomb', type=float, default=None,
                        help='coulomb friction-compensation scale, all joints '
                             '(default: library values ~0.8). 0 disables.')
        sp.add_argument('--robot', default=ROBOT_DESC)
        sp.add_argument('--base', default='base', choices=('base', 'base_link'),
                        help='whichever frame `frames` reported as best')
        sp.add_argument('--tool', default='tool0', choices=('tool0', 'flange'))

    i = sub.add_parser('identify', help='measure per-joint breakaway torque')
    i.add_argument('--rate', type=float, default=1.0, help='torque ramp rate [Nm/s]')
    i.add_argument('--cap', type=float, default=25.0, help='give up above this [Nm]')
    i.add_argument('--qd-thresh', type=float, default=0.02, help='breakaway speed [rad/s]')
    i.add_argument('--frac', type=float, default=0.8,
                   help='fraction of measured friction to compensate')
    i.add_argument('--watchdog', type=float, default=10.0, help='robot-side watchdog [Hz]')

    h = sub.add_parser('hold', help='impedance hold at the current pose')
    add_gains(h)

    s = sub.add_parser('step', help='hold, then step the equilibrium')
    add_gains(s)
    s.add_argument('--axis', choices='xyz', default='z')
    s.add_argument('--dist', type=float, default=0.02, help='step distance [m]')
    s.add_argument('--settle', type=float, default=2.0)

    args = p.parse_args()
    if args.cmd == 'frames':
        cmd_frames(args)
    elif args.cmd == 'identify':
        cmd_identify(args)
    elif args.cmd == 'hold':
        args.settle = 0.0
        impedance_loop(args)
    else:
        d = np.zeros(6)
        d['xyz'.index(args.axis)] = args.dist
        impedance_loop(args, step_delta=d)


if __name__ == '__main__':
    main()
