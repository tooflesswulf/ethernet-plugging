"""
Find the point that forceMode's compliance is actually centred on.

No spring, no teleop: forceMode is entered with a ZERO target wrench and all six
axes compliant, so the arm simply yields to whatever you push with. You push at
two (ideally three) marked points on the tool, one segment each.

The arm runs away from your hand, so each segment is several SHORT shoves rather
than one long push, and the arm is driven back to the segment's starting pose
between them. You do not need to hold a steady force: s below is a ratio of two
velocities, so it does not care how hard you push, only where.

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

NOTE (2026-09-22): every --ft-input mode except "off" currently protective-stops
this robot ("fieldbus input disconnected"). The trigger is the ftRtdeInputEnable
call itself, not the streaming: enabling from inside an already-running 500 Hz
stream stops just as fast as enabling with a stale value, and 125 Hz and 400 Hz
variants behave identically. See probe-ft-input.py, which isolates it without
commanding any motion.

--ft-input replaces what force mode READS. "off" leaves the controller on its own
F/T. The others stream getActualTCPForce back through ftRtdeInputEnable, either
unchanged ("passthrough") or with the moment re-referenced ("tcp"/"flange").
Re-referencing the measurement is what actually moves the compliance centre: with
the moment taken about the TCP, a force through the TCP produces no torque, so
the controller has no reason to rotate. "passthrough" is the control -- it isolates
the streaming path's own effect (frame convention, extra latency) from the shift.
Expect: off -> centre at the flange; passthrough -> unchanged if the conventions
match; tcp -> centre moves to the TCP.

Safety: zero target wrench means the arm is nearly free-floating. With --ft-input
other than "off", force mode reads a wrench THIS SCRIPT supplies -- if the loop
stalls, the controller acts on a stale one. Streaming is enabled only inside a
push and handed back straight after. Speed limits are low by default and the run
aborts if the TCP drifts more than --max-drift from where the segment started.
Keep the e-stop within reach.

If a run dies while the streamed input is enabled, the CONTROLLER stays enabled
and the next run protective-stops ("fieldbus input disconnected") as soon as it
pauses -- typically while zeroing the F/T, before it has streamed anything. This
script therefore disables the input at startup and again before every zero, so
just starting it clears that state (clear the protective stop on the pendant
first).
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


def spin_ratio(pose, twist, force, d, v_min, f_min):
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
    # Only samples where you are actually pushing AND the tool is moving. s is
    # scale-invariant in the force, so short shoves are as good as a steady push.
    m = (vn > v_min) & (np.linalg.norm(force[:, :3], axis=1) > f_min)
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


def shift_wrench(w, pose, tcp_offset, mode):
    """
    Move a wrench's reference point along the tool axis.

    Moment about B, given a wrench (F, tau_A) referenced at A:
        tau_B = tau_A + (p_A - p_B) x F

    With r = R_tool * tcp_offset pointing flange -> TCP:
        'to_tcp'    tau' = tau - r x F      (reference moves outward to the TCP)
        'to_flange' tau' = tau + r x F      (reference moves inward to the flange)
        'none'      unchanged

    Which one you need depends on where getActualTCPForce is actually
    referenced, which UR's docs and its behaviour disagree about -- the doc says
    "in the TCP" while forceMode acts at the flange. Run the modes and let the
    measurement decide.
    """
    if mode == 'none':
        return np.array(w, float)
    r = R.from_rotvec(pose[3:]).apply(np.asarray(tcp_offset, float)[:3])
    sgn = -1.0 if mode == 'to_tcp' else 1.0
    return np.r_[w[:3], np.asarray(w[3:], float) + sgn * np.cross(r, w[:3])]


def to_frame(w, pose, frame):
    """Express a base-frame wrench in the tool frame, if asked."""
    if frame == 'base':
        return w
    Rt = R.from_rotvec(pose[3:]).as_matrix().T
    return np.r_[Rt @ w[:3], Rt @ w[3:]]


# --ft-input value -> (shift mode, human description)
FT_MODES = {
    'off': (None, 'controller uses its own F/T (no streaming)'),
    'passthrough': ('none', 'stream getActualTCPForce back unchanged'),
    'tcp': ('to_tcp', 'stream it with the moment re-referenced to the TCP'),
    'flange': ('to_flange', 'stream it with the moment re-referenced to the flange'),
}


def run_segment(ctrl, recv, args, label, log, seg, t_log):
    """
    One push point, done as several SHORT pushes. The arm yields and runs away
    from your hand, so holding a steady push is not possible and not needed --
    s is scale-invariant in the applied force. Between pushes the arm is driven
    back to where the segment started, so the workspace stays bounded.
    """
    print(f'\n--- segment "{label}" --- hands OFF while the F/T zeroes')
    # Never zero while the controller is waiting on a streamed wrench: nothing is
    # writing it here, and zeroing a source with no data is a fieldbus protective
    # stop. Cheap insurance -- the previous rep already disabled it.
    ctrl.ftRtdeInputEnable(False)
    time.sleep(1.0)
    ctrl.zeroFtSensor()
    time.sleep(0.3)
    home = np.array(recv.getActualTCPPose())

    shift = FT_MODES[args.ft_input][0]
    for rep in range(args.reps):
        task_frame = list(recv.getActualTCPPose()) if args.task_frame == 'tcp' else [0.0] * 6
        print(f'    push {rep + 1}/{args.reps}: shove at "{label}" NOW '
              f'({args.seconds:.0f} s)', flush=True)

        if shift is not None:
            # Prime the input with one sample BEFORE handing the controller the
            # reins, so force mode never starts on a stale or empty register.
            pose = np.array(recv.getActualTCPPose())
            w = np.array(recv.getActualTCPForce(), float)
            ctrl.setExternalForceTorque(
                to_frame(shift_wrench(w, pose, args.tcp, shift), pose, args.ft_frame).tolist())
            # sensor_mass 0: getActualTCPForce is already gravity-compensated, so
            # letting the controller compensate again would subtract the payload twice.
            ctrl.ftRtdeInputEnable(True, 0.0, [0.0] * 3, [0.0] * 3)

        t0 = t_prev = time.perf_counter()
        aborted = ''
        try:
            while time.perf_counter() - t0 < args.seconds:
                t_start = ctrl.initPeriod()
                if recv.isProtectiveStopped() or recv.isEmergencyStopped():
                    aborted = 'protective/emergency stop'
                    break
                pose = np.array(recv.getActualTCPPose())
                if np.linalg.norm(pose[:3] - home[:3]) > args.max_drift:
                    aborted = 'drift limit'
                    break
                w = np.array(recv.getActualTCPForce(), float)
                sent = w if shift is None else to_frame(
                    shift_wrench(w, pose, args.tcp, shift), pose, args.ft_frame)
                if shift is not None:
                    ctrl.setExternalForceTorque(sent.tolist())
                ctrl.forceMode(task_frame, SELECTION, ZERO_WRENCH, FRAME_TYPE, args.vmax.tolist())
                log.append(np.r_[time.perf_counter() - t_log, seg, rep, pose,
                                 recv.getActualTCPSpeed(), w, sent, recv.getActualQ()])
                ctrl.waitPeriod(t_start)
                now = time.perf_counter()
                # Bail out ourselves rather than let the controller notice: while the
                # streamed input is enabled, a gap in it is a "fieldbus input
                # disconnected" protective stop, which needs a pendant reset.
                if shift is not None and now - t_prev > args.max_stall:
                    aborted = f'loop stalled {1e3 * (now - t_prev):.0f} ms'
                    t_prev = now
                    break
                t_prev = now
        finally:
            # ORDER MATTERS. forceModeStop() runs stopl(10) in the control script,
            # which blocks while the arm decelerates; the script processes no further
            # commands meanwhile. Disabling the streamed input first means the
            # controller is back on its own F/T before that blocking stop, instead of
            # sitting on an input nobody is writing.
            if shift is not None:
                ctrl.ftRtdeInputEnable(False)
            ctrl.forceModeStop()
        if aborted == 'protective/emergency stop':
            print(f'    ABORTED: {aborted}')
            return aborted
        if aborted:
            print(f'      push {rep + 1} ended early: {aborted}')
        back = np.linalg.norm(np.array(recv.getActualTCPPose())[:3] - home[:3])
        if back > 2e-3:
            print(f'      hands off -- returning {1e3 * back:.0f} mm', flush=True)
            time.sleep(0.8)
            ctrl.moveL(home.tolist(), args.return_speed, 0.3)
    return ''


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--ip', default='192.168.0.100')
    ap.add_argument('--offsets', type=offsets_arg, default='tip=0.03,base=-0.154',
                    help='push points as name=offset_along_tool_z_from_TCP[m], tip positive')
    ap.add_argument('--task-frame', choices=('base', 'tcp'), default='base')
    ap.add_argument('--ft-input', choices=tuple(FT_MODES), default='off',
                    help='what force mode reads: "off" = the controller\'s own F/T; the others '
                         'stream getActualTCPForce back via ftRtdeInputEnable, either unchanged '
                         '("passthrough") or with the moment re-referenced ("tcp"/"flange")')
    ap.add_argument('--ft-frame', choices=('tool', 'base'), default='tool',
                    help='frame the streamed wrench is expressed in; UR does not document this, '
                         'so if "tool" behaves oddly try "base"')
    ap.add_argument('--seconds', type=float, default=3.0, help='length of one push [s]')
    ap.add_argument('--reps', type=int, default=6, help='pushes per point')
    ap.add_argument('--f-min', type=float, default=3.0,
                    help='only use samples pushed harder than this [N]')
    ap.add_argument('--return-speed', type=float, default=0.05,
                    help='speed for the move back between pushes [m/s]')
    ap.add_argument('--vmax', type=lambda s: np.array([float(x) for x in s.split(',')] * 3
                                                      if len(s.split(',')) == 2 else
                                                      [float(x) for x in s.split(',')]),
                    default=np.array([0.03] * 3 + [0.3] * 3),
                    help='forceMode speed limits "m/s,rad/s"')
    ap.add_argument('--max-drift', type=float, default=0.10, help='abort if TCP moves this far [m]')
    ap.add_argument('--max-stall', type=float, default=0.02,
                    help='with --ft-input on, end the push if one loop takes longer than this [s]; '
                         'a gap in the streamed wrench is a fieldbus protective stop')
    ap.add_argument('--v-min', type=float, default=1e-3,
                    help='ignore samples whose push point moves slower than this [m/s]')
    ap.add_argument('--fm-damping', type=float, default=None)
    ap.add_argument('--fm-gain', type=float, default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    ctrl = rtde_control.RTDEControlInterface(args.ip)
    recv = rtde_receive.RTDEReceiveInterface(args.ip)
    # ft_rtde_input_enable is CONTROLLER state and outlives the process that set
    # it: a run that died before its cleanup leaves the controller waiting on a
    # wrench nobody is sending, and the next run trips "fieldbus input
    # disconnected" while it sits there zeroing. Clear it before anything else.
    ctrl.ftRtdeInputEnable(False)

    tcp = args.tcp = np.array(ctrl.getTCPOffset())
    print(f'TCP offset  : {np.round(tcp, 4)}')
    print(f'F/T input   : {args.ft_input} -- {FT_MODES[args.ft_input][1]}'
          + (f', sent in the {args.ft_frame} frame' if args.ft_input != 'off' else ''))
    print(f'payload     : {recv.getPayload():.3f} kg  cog {np.round(recv.getPayloadCog(), 4)}')
    print(f'push points : {args.offsets}')
    if args.fm_damping is not None:
        ctrl.forceModeSetDamping(args.fm_damping)
    if args.fm_gain is not None:
        ctrl.forceModeSetGainScaling(args.fm_gain)

    log, results = [], []
    t_log = time.perf_counter()
    try:
        if args.ft_input != 'off':
            print('\nWARNING: --ft-input other than "off" protective-stops this robot '
                  '("fieldbus input\n         disconnected") as soon as the enable call goes '
                  'through -- cause not yet\n         found. See probe-ft-input.py. Expect a stop.')
        for seg, (label, d) in enumerate(args.offsets):
            input(f'\n[{seg + 1}/{len(args.offsets)}] ready to push at "{label}" '
                  f'({1e3 * d:+.0f} mm from TCP along tool z)? press Enter')
            run_segment(ctrl, recv, args, label, log, seg, t_log)
            L = np.array([r for r in log if r[1] == seg])
            s, n, pz = spin_ratio(L[:, 3:9], L[:, 9:15], L[:, 15:21], d, args.v_min, args.f_min)
            if not np.isfinite(s):
                print(f'    only {n} usable samples (need >50 with |F| > {args.f_min:.0f} N '
                      f'while moving) -- push harder, or lower --f-min')
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
            if args.ft_input != 'off':
                ctrl.ftRtdeInputEnable(False)
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
        out = args.out or time.strftime(
            f'forcemode-center-{args.task_frame}-ft{args.ft_input}-%Y%m%d-%H%M%S.npz')
        np.savez(out, t=L[:, 0], seg=L[:, 1], rep=L[:, 2], pose=L[:, 3:9], twist=L[:, 9:15],
                 tcp_force=L[:, 15:21], ft_sent=L[:, 21:27], q=L[:, 27:33],
                 f_min=args.f_min, v_min=args.v_min, seconds=args.seconds, reps=args.reps,
                 ft_input=args.ft_input, ft_frame=args.ft_frame, max_stall=args.max_stall,
                 labels=np.array([r[0] for r in args.offsets]),
                 push_offsets=np.array([r[1] for r in args.offsets]),
                 task_frame=args.task_frame, tcp_offset=tcp, vmax=args.vmax,
                 payload=recv.getPayload() if recv.isConnected() else np.nan)
        print(f'wrote {out}  ({len(L)} samples)')


if __name__ == '__main__':
    main()
