"""
Find the point that forceMode's compliance is actually centred on.

No spring, no teleop: forceMode is entered with a ZERO target wrench and all six
axes compliant, so the arm simply yields to whatever you push with. You push at
two (ideally three) marked points on the tool, one segment each.

Geometry, for a compliance centred at c with isotropic gains (offsets are
distances along the tool z axis measured from the TCP, tip positive):

    v_c = a_t * F        omega = a_r * (d - c) * F        k = a_t / a_r

    what each segment measures:   s(d) = |omega| / |v| at the push point
                                       = |d - c| / (k + (d - c)^2)

s is the right observable because it stays finite everywhere: a push exactly at
the centre is pure translation, s = 0, whereas the pivot would run off to
infinity there. Each segment gives one (d, s), so two pin down c and k. A third
breaks the |d - c| sign symmetry, which can otherwise admit a mirror solution.

Run it twice -- once with --task-frame base, once with --task-frame tcp. If c
moves with the task frame, the reference point is the task frame and you can put
the compliance wherever you like. If c does not move, it is tied to the TCP or
the flange and only the TCP definition can move it.

    python test-forcemode-center.py --offsets tip=0.03,base=-0.154

Safety: zero target wrench means the arm is nearly free-floating. Speed limits are
low by default and the run aborts if the TCP drifts more than --max-drift from
where the segment started. Keep the e-stop within reach.
"""
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import numpy as np
import argparse
import time

import rtde_control
import rtde_receive

SELECTION = [1] * 6
FRAME_TYPE = 2
ZERO_WRENCH = [0.0] * 6


def offsets_arg(s):
    """"tip=0.03,base=-0.154" -> [("tip", 0.03), ("base", -0.154)]"""
    out = []
    for part in s.split(','):
        name, _, val = part.partition('=')
        out.append((name.strip(), float(val)))
    if len(out) < 2:
        raise argparse.ArgumentTypeError('need at least two push points')
    return out


def spin_ratio(pose, twist, d, v_min):
    """
    s = |omega| / |v| at the PUSH POINT, for one segment.

    This is the observable to fit, not the pivot location: pushing exactly at the
    compliance centre gives pure translation, where the pivot runs off to infinity
    but s simply goes to zero. Model:

        s(d) = |d - c| / (k + (d - c)^2),     k = a_t / a_r

    Returns (s, n_used, pivot_z_or_nan).
    """
    v, w = twist[:, :3], twist[:, 3:]
    Rb = R.from_rotvec(pose[:, 3:]).as_matrix()
    arm = np.einsum('nij,j->ni', Rb, np.array([0.0, 0.0, d]))
    v_push = v + np.cross(w, arm)                      # velocity of the pushed point
    vn, wn = np.linalg.norm(v_push, axis=1), np.linalg.norm(w, axis=1)
    m = vn > v_min
    if m.sum() < 50:
        return np.nan, m.sum(), np.nan
    s = float(np.median(wn[m] / vn[m]))
    # Descriptive only: where the motion looked like it pivoted (NaN if barely rotating).
    mr = m & (wn > 0.02)
    if mr.sum() < 50:
        return s, m.sum(), np.nan
    r = np.cross(w[mr], v[mr]) / (wn[mr] ** 2)[:, None]
    pz = float(np.median(np.einsum('nij,nj->ni', Rb[mr].transpose(0, 2, 1), r)[:, 2]))
    return s, m.sum(), pz


def solve_centre(d, s):
    """
    Fit s_i = |d_i - c| / (k + (d_i - c)^2) over segments. Returns list of
    distinct (c, k, cost), best first -- |d - c| is symmetric about c, so two
    push points can admit two solutions and the caller should say so.
    """
    def resid(x):
        c, k = x
        return s - np.abs(d - c) / (max(k, 1e-6) + (d - c) ** 2)

    sols = []
    for c0 in np.linspace(d.min() - 0.25, d.max() + 0.25, 40):
        for k0 in (0.002, 0.02, 0.2):
            try:
                r = least_squares(resid, [c0, k0], bounds=([-1.0, 1e-6], [1.0, 10.0]))
            except Exception:
                continue
            if not np.isfinite(r.cost):
                continue
            if not any(abs(r.x[0] - c) < 5e-3 and abs(r.x[1] - k) < 1e-3 for c, k, _ in sols):
                sols.append((float(r.x[0]), float(r.x[1]), float(r.cost)))
    sols.sort(key=lambda t: t[2])
    return sols[:3]


def run_segment(ctrl, recv, args, label, log, seg):
    print(f'\n--- segment "{label}" --- keep hands OFF the tool while the F/T zeroes')
    time.sleep(1.0)
    ctrl.zeroFtSensor()
    time.sleep(0.3)

    task_frame = list(recv.getActualTCPPose()) if args.task_frame == 'tcp' else [0.0] * 6
    start = np.array(recv.getActualTCPPose())
    print(f'    push at "{label}" and keep pushing -- {args.seconds:.0f} s, '
          f'task_frame={args.task_frame}')
    t0 = time.perf_counter()
    aborted = ''
    while time.perf_counter() - t0 < args.seconds:
        t_start = ctrl.initPeriod()
        if recv.isProtectiveStopped() or recv.isEmergencyStopped():
            aborted = 'protective/emergency stop'
            break
        pose = np.array(recv.getActualTCPPose())
        drift = np.linalg.norm(pose[:3] - start[:3])
        if drift > args.max_drift:
            aborted = f'drifted {1e3 * drift:.0f} mm'
            break
        ctrl.forceMode(task_frame, SELECTION, ZERO_WRENCH, FRAME_TYPE, args.vmax.tolist())
        log.append(np.r_[time.perf_counter() - t0, seg, pose,
                         recv.getActualTCPSpeed(), recv.getActualTCPForce(),
                         recv.getActualQ()])
        ctrl.waitPeriod(t_start)
    ctrl.forceModeStop()
    if aborted:
        print(f'    ABORTED: {aborted}')
    return aborted


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--ip', default='192.168.0.100')
    ap.add_argument('--hz', type=float, default=500.0)
    ap.add_argument('--offsets', type=offsets_arg, default='tip=0.03,base=-0.154',
                    help='push points as name=offset_along_tool_z_from_TCP[m], tip positive')
    ap.add_argument('--task-frame', choices=('base', 'tcp'), default='base')
    ap.add_argument('--seconds', type=float, default=20.0, help='push time per segment')
    ap.add_argument('--vmax', type=lambda s: np.array([float(x) for x in s.split(',')] * 3
                                                      if len(s.split(',')) == 2 else
                                                      [float(x) for x in s.split(',')]),
                    default=np.array([0.03] * 3 + [0.3] * 3),
                    help='forceMode speed limits "m/s,rad/s"')
    ap.add_argument('--max-drift', type=float, default=0.10, help='abort if TCP moves this far [m]')
    ap.add_argument('--v-min', type=float, default=1e-3,
                    help='ignore samples whose push point moves slower than this [m/s]')
    ap.add_argument('--fm-damping', type=float, default=None)
    ap.add_argument('--fm-gain', type=float, default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    ctrl = rtde_control.RTDEControlInterface(args.ip)
    recv = rtde_receive.RTDEReceiveInterface(args.ip)
    tcp = np.array(ctrl.getTCPOffset())
    print(f'TCP offset  : {np.round(tcp, 4)}')
    print(f'payload     : {recv.getPayload():.3f} kg  cog {np.round(recv.getPayloadCog(), 4)}')
    print(f'push points : {args.offsets}')
    if args.fm_damping is not None:
        ctrl.forceModeSetDamping(args.fm_damping)
    if args.fm_gain is not None:
        ctrl.forceModeSetGainScaling(args.fm_gain)

    log, results = [], []
    try:
        for seg, (label, d) in enumerate(args.offsets):
            input(f'\n[{seg + 1}/{len(args.offsets)}] ready to push at "{label}" '
                  f'({1e3 * d:+.0f} mm from TCP along tool z)? press Enter')
            run_segment(ctrl, recv, args, label, log, seg)
            L = np.array([r for r in log if r[1] == seg])
            s, n, pz = spin_ratio(L[:, 2:8], L[:, 8:14], d, args.v_min)
            if not np.isfinite(s):
                print(f'    the tool barely moved ({n} usable samples) -- push harder')
                continue
            results.append((label, d, s))
            print(f'    spin ratio |w|/|v| at the push point: {s:6.2f} rad/m   (n={n})')
            print(f'    pivot along tool z: '
                  + (f'{1e3 * pz:+.0f} mm from TCP' if np.isfinite(pz)
                     else 'none -- essentially pure translation, so the centre is HERE'))
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        try:
            ctrl.forceModeStop()
        finally:
            ctrl.stopScript()

    if len(results) >= 2:
        d = np.array([r[1] for r in results]); s = np.array([r[2] for r in results])
        sols = solve_centre(d, s)
        print(f'\ncompliance centre (tool z, 0 = TCP, flange = {-1e3 * tcp[2]:+.0f} mm):')
        for i, (c, k, cost) in enumerate(sols):
            print(f'  {"best" if i == 0 else "also"}: {1e3 * c:+6.0f} mm   '
                  f'a_t/a_r = {k:.4f} m^2  (pushes within {1e3 * np.sqrt(k):.0f} mm of it '
                  f'translate more than they rotate)   cost {cost:.4g}')
        if len(sols) > 1 and abs(sols[0][0] - sols[1][0]) > 0.02:
            print('  NOTE: |d - c| is symmetric, so these fit equally well. Add a third '
                  'push point to break the tie.')

    if log:
        L = np.array(log)
        out = args.out or time.strftime(f'forcemode-center-{args.task_frame}-%Y%m%d-%H%M%S.npz')
        np.savez(out, t=L[:, 0], seg=L[:, 1], pose=L[:, 2:8], twist=L[:, 8:14],
                 tcp_force=L[:, 14:20], q=L[:, 20:26],
                 labels=np.array([r[0] for r in args.offsets]),
                 push_offsets=np.array([r[1] for r in args.offsets]),
                 task_frame=args.task_frame, tcp_offset=tcp, vmax=args.vmax,
                 payload=recv.getPayload() if recv.isConnected() else np.nan)
        print(f'wrote {out}  ({len(L)} samples)')


if __name__ == '__main__':
    main()
