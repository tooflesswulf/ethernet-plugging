"""
Feel-test for 6-axis forceMode teleop: a host-side Cartesian spring rendered
through UR's on-controller force loop.

    wrench = K * (x_target - x) - D * xdot          (base frame, all 6 axes compliant)
    forceMode(base, [1]*6, clamp(wrench), type=2, limits=speed caps)

The DualSense moves x_target exactly like teleoperation.py (same interface.py
mapping). The contact reaction is closed on the controller at 500 Hz; only the
spring goes over RTDE, and the spring is soft compared to the contact.

Controls
    sticks / L2 R2 (+L1 R1 flips)   move target (same as teleoperation.py)
    Dpad Up / Down                  stiffness x1.5 / /1.5 (damping scales with sqrt)
    Cross                           re-anchor target to the actual pose
    Dpad Left                       zero the F/T sensor -- ONLY when not in contact
    Ctrl-C                          stop (forceModeStop, then stopScript)

Built-in F/T caveat: the wrist sensor shows a ~10 N orientation-dependent error
that the declared payload (1.551 kg, weighed) does not explain. forceMode will
push against whatever that error is at the current wrist orientation, so re-zero
(Dpad Left) after large rotations, out of contact. The log includes
getActualTCPForce and getFtRawWrench so the error can be characterised.
"""
from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse
import time

import rtde_control
import rtde_receive

TASK_FRAME = [0.0] * 6          # base frame
SELECTION = [1] * 6             # compliant in every axis
FRAME_TYPE = 2                  # task frame not transformed


def csv6(s):
    v = [float(x) for x in s.split(',')]
    if len(v) == 1:
        v = v * 6
    if len(v) == 2:             # translational, rotational
        v = [v[0]] * 3 + [v[1]] * 3
    assert len(v) == 6, s
    return np.array(v)


def pose_error(target, actual):
    """target - actual in the base frame: [dp, rotvec(R_t R_a^T)]."""
    dp = np.asarray(target[:3]) - np.asarray(actual[:3])
    dr = (R.from_rotvec(target[3:]) * R.from_rotvec(actual[3:]).inv()).as_rotvec()
    return np.r_[dp, dr]


def leash(target, actual, max_dp, max_dr):
    """
    Pull the target back so it is at most max_dp / max_dr from the actual pose.
    Without this, pushing the stick into a wall winds the target up and the arm
    lunges when contact breaks.
    """
    e = pose_error(target, actual)
    n = np.linalg.norm(e[:3])
    if n > max_dp:
        target[:3] = np.asarray(actual[:3]) + e[:3] * (max_dp / n)
    n = np.linalg.norm(e[3:])
    if n > max_dr:
        dr = e[3:] * (max_dr / n)
        target[3:] = (R.from_rotvec(dr) * R.from_rotvec(actual[3:])).as_rotvec()
    return target


def wrench_to_flange(w, pose, tcp_offset):
    """
    Re-reference a wrench from the TCP to the flange.

    forceMode's compliance is centred on the FLANGE, not the TCP: measured on
    hardware (forcemode-center-{base,tcp}-20260922-*.npz), a push at the flange
    is pure translation while a push at the TCP pivots about the flange, and
    task_frame does not move it. A wrench meant to act at the TCP therefore has
    to carry the moment of its force about the flange, or you are silently
    commanding up to |r| * F of extra torque -- 0.154 m * 25 N ~ 3.9 Nm here.

        F_flange = F_tcp        tau_flange = tau_tcp + r x F_tcp
    """
    r = R.from_rotvec(pose[3:]).apply(np.asarray(tcp_offset, float)[:3])
    return np.r_[w[:3], w[3:] + np.cross(r, w[:3])]


def clamp_wrench(w, fmax, tmax):
    w = np.array(w, float)
    n = np.linalg.norm(w[:3])
    if n > fmax:
        w[:3] *= fmax / n
    n = np.linalg.norm(w[3:])
    if n > tmax:
        w[3:] *= tmax / n
    return w


class Edge:
    """Rising-edge detector for DualSense buttons."""

    def __init__(self):
        self.last = {}

    def __call__(self, state, name):
        now = bool(getattr(state, name))
        rose = now and not self.last.get(name, False)
        self.last[name] = now
        return rose


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--ip', default='192.168.0.100')
    ap.add_argument('--hz', type=float, default=500.0)
    ap.add_argument('--teleop-hz', type=float, default=100.0,
                    help='DualSense / target update rate')
    ap.add_argument('--k', type=csv6, default=csv6('400,15'),
                    help='stiffness N/m,Nm/rad: one value, "trans,rot", or 6 values')
    ap.add_argument('--d', type=csv6, default=csv6('40,0.8'),
                    help='damping N s/m,Nm s/rad (same format as --k)')
    ap.add_argument('--fmax', type=float, default=25.0, help='force clamp [N]')
    ap.add_argument('--tmax', type=float, default=2.0, help='torque clamp [Nm]')
    ap.add_argument('--vmax', type=csv6, default=csv6('0.10,0.5'),
                    help='forceMode speed limits m/s,rad/s')
    ap.add_argument('--leash', type=csv6, default=csv6('0.03,0.25'),
                    help='max target-actual offset m,rad (only first and last used)')
    ap.add_argument('--fm-damping', type=float, default=None,
                    help='forceModeSetDamping 0..1 (controller default 0.005); set once before entry')
    ap.add_argument('--fm-gain', type=float, default=None,
                    help='forceModeSetGainScaling 0..2 (default 1); set once before entry')
    ap.add_argument('--payload-kg', type=float, default=None,
                    help='override payload mass for this run (keeps current CoG)')
    ap.add_argument('--wrench-at', choices=('flange', 'tcp'), default='flange',
                    help='where the commanded wrench is meant to act; forceMode centres its '
                         'compliance on the FLANGE, so "flange" re-references the TCP wrench '
                         'before sending. "tcp" reproduces the old (wrong) behaviour.')
    ap.add_argument('--no-zero', action='store_true', help='skip zeroFtSensor at start')
    ap.add_argument('--out', default=None, help='log file (default forcemode-teleop-<time>.npz)')
    args = ap.parse_args()

    k_scale = 1.0
    dt = 1.0 / args.hz
    teleop_every = max(1, round(args.hz / args.teleop_hz))

    ctrl = rtde_control.RTDEControlInterface(args.ip)
    recv = rtde_receive.RTDEReceiveInterface(args.ip)

    mass, cog = recv.getPayload(), list(recv.getPayloadCog())
    tcp = np.array(ctrl.getTCPOffset())
    print(f'payload     : {mass:.3f} kg  cog {np.round(cog, 4)}  tcp {np.round(tcp, 4)}')
    print(f'wrench sent : referenced at the {args.wrench_at}')
    if args.payload_kg is not None:
        ctrl.setPayload(args.payload_kg, cog)
        print(f'payload set : {args.payload_kg:.3f} kg for this run')

    # Lazy import: interface.py builds a robosuite env for the DualSense driver.
    from interface import DualSenseInterface
    iface = DualSenseInterface(recv.getActualTCPPose(), enable_zadaptive=False)
    edge = Edge()

    if not args.no_zero:
        print('zeroing F/T -- arm must be free of contact')
        ctrl.zeroFtSensor()
        time.sleep(0.2)

    if args.fm_damping is not None:
        ctrl.forceModeSetDamping(args.fm_damping)
    if args.fm_gain is not None:
        ctrl.forceModeSetGainScaling(args.fm_gain)

    limits = args.vmax.tolist()
    log = []
    print(f'K {args.k}  D {args.d}  fmax {args.fmax} N  tmax {args.tmax} Nm  vmax {args.vmax}')
    print('running -- Ctrl-C to stop')
    t0 = time.perf_counter()
    i = 0
    try:
        while True:
            t_start = ctrl.initPeriod()
            if recv.isProtectiveStopped() or recv.isEmergencyStopped():
                print('\nprotective/emergency stop -- exiting')
                break

            pose = np.array(recv.getActualTCPPose())
            twist = np.array(recv.getActualTCPSpeed())

            if i % teleop_every == 0:
                iface.update(teleop_every * dt)
                st = iface.dualsense.state
                if edge(st, 'DpadUp'):
                    k_scale *= 1.5
                    print(f'\nstiffness x{k_scale:.2f}  K {np.round(args.k * k_scale, 2)}')
                if edge(st, 'DpadDown'):
                    k_scale /= 1.5
                    print(f'\nstiffness x{k_scale:.2f}  K {np.round(args.k * k_scale, 2)}')
                if edge(st, 'Cross'):
                    iface.targ_pose = pose.copy()
                    print('\ntarget re-anchored')
                if edge(st, 'DpadLeft'):
                    ctrl.zeroFtSensor()
                    print('\nF/T zeroed')
                leash(iface.targ_pose, pose, args.leash[0], args.leash[5])

            target = iface.targ_pose
            K = args.k * k_scale
            D = args.d * np.sqrt(k_scale)      # keep the damping ratio roughly fixed
            e = pose_error(target, pose)
            wrench = clamp_wrench(K * e - D * twist, args.fmax, args.tmax)
            sent = wrench if args.wrench_at == 'tcp' else wrench_to_flange(wrench, pose, tcp)
            ctrl.forceMode(TASK_FRAME, SELECTION, sent.tolist(), FRAME_TYPE, limits)

            force = recv.getActualTCPForce()
            try:
                raw = recv.getFtRawWrench()
            except Exception:
                raw = [np.nan] * 6
            log.append(np.r_[time.perf_counter() - t0, pose, target, twist, wrench,
                             force, raw, recv.getActualQ(), k_scale])
            if i % int(args.hz / 5) == 0:
                print(f'|e| {1e3 * np.linalg.norm(e[:3]):5.1f} mm {np.degrees(np.linalg.norm(e[3:])):5.2f} deg'
                      f'  cmd F {np.round(wrench[:3], 1)}  meas F {np.round(force[:3], 1)}   ', end='\r')
            i += 1
            ctrl.waitPeriod(t_start)
    except KeyboardInterrupt:
        print('\nstopping')
    finally:
        try:
            ctrl.forceModeStop()
        finally:
            if args.payload_kg is not None:
                ctrl.setPayload(mass, cog)
                print(f'payload restored to {mass:.3f} kg')
            ctrl.stopScript()

    if log:
        L = np.array(log)
        out = args.out or time.strftime('forcemode-teleop-%Y%m%d-%H%M%S.npz')
        np.savez(out, t=L[:, 0], pose=L[:, 1:7], target=L[:, 7:13], twist=L[:, 13:19],
                 wrench_cmd=L[:, 19:25], tcp_force=L[:, 25:31], ft_raw=L[:, 31:37],
                 q=L[:, 37:43], k_scale=L[:, 43], k=args.k, d=args.d, limits=args.vmax,
                 payload=mass, payload_cog=cog, tcp_offset=tcp, wrench_at=args.wrench_at,
                 fmax=args.fmax, tmax=args.tmax, leash=args.leash,
                 fm_damping=np.nan if args.fm_damping is None else args.fm_damping,
                 fm_gain=np.nan if args.fm_gain is None else args.fm_gain)
        print(f'wrote {out}  ({len(L)} samples, {len(L) / L[-1, 0]:.0f} Hz achieved)')


if __name__ == '__main__':
    main()
