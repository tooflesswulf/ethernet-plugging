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

# UR5e rated joint torques [Nm]. The clip below is a fraction of these.
TAU_RATED = np.array([150., 150., 150., 28., 28., 28.])


def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def pose_to_se3(p):
    return pin.SE3(R.from_rotvec(p[3:]).as_matrix(), np.asarray(p[:3], float))


class UR5eKin:
    """
    pinocchio UR5e wrapped so that everything it returns is in the frames the UR
    controller actually reports in: the `base` frame (NOT `base_link`, which is
    the ROS REP-103 convention and is rotated 180 deg about Z), and the TCP
    (NOT `flange`, which shares tool0's origin but not its orientation).
    """

    def __init__(self, tcp_offset):
        robot = load_robot_description("ur5e_description")
        self.model = robot.model
        self.data = self.model.createData()
        self.f_base = self.model.getFrameId("base")
        self.f_tool = self.model.getFrameId("tool0")
        self.tcp = pose_to_se3(tcp_offset)

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


def connect():
    flags = (rtde_control.RTDEControlInterface.FLAGS_DEFAULT
             | rtde_control.RTDEControlInterface.FLAG_UPPER_RANGE_REGISTERS)
    ctrl = rtde_control.RTDEControlInterface(ROBOT_IP, flags=flags)
    recv = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    tcp_offset = ctrl.getTCPOffset()
    print(f'TCP offset : {np.round(tcp_offset, 5)}')
    print(f'step time  : {ctrl.getStepTime()} s')
    print(f'payload    : {recv.getPayload() if hasattr(recv, "getPayload") else "?"} kg')
    return ctrl, recv, UR5eKin(tcp_offset)


# ======================================================================
# 1. frames -- no torque commanded, hand-move the arm
# ======================================================================
def cmd_frames(args):
    ctrl, recv, kin = connect()
    print('\nEntering freedrive. Move the arm slowly through a few poses.')
    print('Sampling for %.0f s...\n' % args.duration)

    v_err, w_err, fk_err, n, worst = [], [], [], 0, 0.0
    try:
        ctrl.teachMode()
        t_end = time.time() + args.duration
        while time.time() < t_end:
            q = np.array(recv.getActualQ())
            qd = np.array(recv.getActualQd())
            twist = np.array(recv.getActualTCPSpeed())
            pose = np.array(recv.getActualTCPPose())

            if np.linalg.norm(qd) < 0.05:      # only score while actually moving
                time.sleep(0.02)
                continue

            pred = kin.jacobian(q) @ qd
            v_err.append(np.abs(pred[:3] - twist[:3]).max())
            w_err.append(np.abs(pred[3:] - twist[3:]).max())
            fk_err.append(np.abs(kin.fk(q)[:3] - pose[:3]).max())
            worst = max(worst, v_err[-1])
            n += 1
            time.sleep(0.02)
    finally:
        ctrl.endTeachMode()

    if n == 0:
        print('No motion sampled -- nothing was validated. Move the arm and retry.')
        return

    print(f'samples: {n}')
    print(f'  FK position     : mean {np.mean(fk_err)*1000:7.3f} mm   max {np.max(fk_err)*1000:7.3f} mm')
    print(f'  J@qd vs v_tcp   : mean {np.mean(v_err)*1000:7.3f} mm/s  max {np.max(v_err)*1000:7.3f} mm/s')
    print(f'  J@qd vs w_tcp   : mean {np.mean(w_err)*1000:7.3f} mrad/s max {np.max(w_err)*1000:7.3f} mrad/s')

    ok = np.mean(fk_err) < 2e-3 and np.mean(v_err) < 5e-3
    print('\n' + ('PASS -- frames agree, safe to try `hold`.' if ok else
                  'FAIL -- do NOT command torque.\n'
                  '  Large x/y sign errors => wrong base frame (base vs base_link).\n'
                  '  Orientation errors     => wrong tool frame (tool0 vs flange).\n'
                  '  Uniform scale errors   => wrong TCP offset.'))


# ======================================================================
# 2/3. impedance hold, and a setpoint step
# ======================================================================
def impedance_loop(args, step_delta=None):
    ctrl, recv, kin = connect()

    K = np.r_[np.full(3, args.kp), np.full(3, args.kr)]
    D = np.r_[np.full(3, args.dp), np.full(3, args.dr)]
    tau_max = args.tau_frac * TAU_RATED
    dt = ctrl.getStepTime()

    print(f'\nK        : {K}')
    print(f'D        : {D}')
    print(f'tau clip : {np.round(tau_max, 1)} Nm  ({args.tau_frac:.0%} of rated)')

    ctrl.setWatchdog(50.0)   # robot-side: stops control if this process dies

    eq = np.array(recv.getActualTCPPose())      # equilibrium latched here, never moves
    print(f'equilibrium: {np.round(eq, 4)}')
    if step_delta is not None:
        print(f'stepping after {args.settle:.1f} s by {step_delta}')
    print('\nCtrl-C to stop.\n')

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

            if not np.isfinite(tau).all():
                print('\nnon-finite torque, aborting')
                break
            tau = np.clip(tau, -tau_max, tau_max)

            ctrl.directTorque(tau.tolist(), args.friction_comp)

            ticks += 1
            if (time.perf_counter() - t_prev) > 3 * dt:
                late += 1
            t_prev = time.perf_counter()

            if ticks % 250 == 0:
                print(f'  t={now:6.2f}s  |e_p|={np.linalg.norm(e[:3])*1000:6.2f}mm  '
                      f'|F|={np.linalg.norm(F[:3]):6.2f}N  |tau|={np.abs(tau).max():6.2f}Nm  '
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
                ctrl.directTorque([0.0] * 6, False)
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

    def add_gains(sp):
        sp.add_argument('--kp', type=float, default=200., help='translational stiffness [N/m]')
        sp.add_argument('--kr', type=float, default=10., help='rotational stiffness [Nm/rad]')
        sp.add_argument('--dp', type=float, default=40., help='translational damping [Ns/m]')
        sp.add_argument('--dr', type=float, default=2., help='rotational damping [Nms/rad]')
        sp.add_argument('--dq', type=float, default=0.5, help='joint damping floor [Nms/rad]')
        sp.add_argument('--vel-alpha', type=float, default=0.4)
        sp.add_argument('--tau-frac', type=float, default=0.20, help='fraction of rated joint torque')
        sp.add_argument('--ramp', type=float, default=0.5)
        sp.add_argument('--friction-comp', action='store_true', default=True)
        sp.add_argument('--no-friction-comp', dest='friction_comp', action='store_false')

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
    elif args.cmd == 'hold':
        args.settle = 0.0
        impedance_loop(args)
    else:
        d = np.zeros(6)
        d['xyz'.index(args.axis)] = args.dist
        impedance_loop(args, step_delta=d)


if __name__ == '__main__':
    main()
