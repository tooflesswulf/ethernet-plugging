"""
Find which call triggers the "fieldbus input disconnected" protective stop.

NO MOTION AT ALL: this never enters force mode and never commands a move. It
walks through the external-F/T calls one at a time, waiting and checking the
protective-stop flag after each, so the offending step names itself.

What it established so far (2026-09-22): the ENABLE ITSELF trips the stop, and
timing is not the cause. Enabling with a 3 s old value stopped; enabling from
inside a 500 Hz stream stopped too (~70 updates in, worst loop gap 5.3 ms);
125 Hz with a matched RTDE connection and 400 Hz busy-waited both stopped the
same way. Rate and jitter change nothing, so this looks like configuration or
firmware, not starvation.

Remaining suspects, which --api and the version print below are for:
  * ft_rtde_input_enable vs the older enable_external_ft_sensor primitive
  * the installation not being configured to accept an external F/T source
  * external_force_torque missing from the RTDE input recipe despite the write
    returning True

    python probe-ft-input.py --api ft_rtde           # what failed so far
    python probe-ft-input.py --api external_ft       # the other primitive

Clear any protective stop on the pendant first. Read the step that reports
STOPPED, and if it is the very first check, the stop was already latched.
"""
import numpy as np
import argparse
import time

import rtde_control
import rtde_receive


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--ip', default='192.168.0.100')
    ap.add_argument('--no-prime', action='store_true',
                    help='skip writing external_force_torque before anything else')
    ap.add_argument('--settle', type=float, default=3.0,
                    help='seconds to wait after each step (the stop takes ~1-2 s to appear)')
    ap.add_argument('--order', choices=('stream-then-enable', 'enable-then-stream'),
                    default='stream-then-enable',
                    help='"enable-then-stream" reproduces the failure; the default is the fix')
    ap.add_argument('--warmup', type=int, default=20,
                    help='cycles to stream before flipping the enable on')
    ap.add_argument('--verbose', action='store_true',
                    help='construct with FLAG_VERBOSE so ur_rtde prints its RTDE recipe setup -- '
                         'look for external_force_torque being registered or rejected')
    ap.add_argument('--upper-registers', action='store_true',
                    help='construct with FLAG_UPPER_RANGE_REGISTERS (24-47), in case another '
                         'RTDE client already owns the low registers')
    ap.add_argument('--api', choices=('ft_rtde', 'external_ft'), default='ft_rtde',
                    help='which primitive enables the external wrench: ftRtdeInputEnable '
                         '(script cmd 56) or enableExternalFtSensor (cmd 57)')
    ap.add_argument('--rtde-freq', type=float, default=-1.0,
                    help='RTDE frequency for the control interface (-1 = default, 500 Hz on '
                         'e-series). Lower values may widen the gap the controller tolerates.')
    ap.add_argument('--stream-hz', type=float, default=0.0,
                    help='rate to write the register at (0 = every robot cycle via waitPeriod)')
    args = ap.parse_args()

    flags = int(rtde_control.RTDEControlInterface.FLAG_UPLOAD_SCRIPT)
    if args.verbose:
        flags |= int(rtde_control.RTDEControlInterface.FLAG_VERBOSE)
    if args.upper_registers:
        flags |= int(rtde_control.RTDEControlInterface.FLAG_UPPER_RANGE_REGISTERS)
    recv = rtde_receive.RTDEReceiveInterface(args.ip)
    ctrl = rtde_control.RTDEControlInterface(args.ip, args.rtde_freq, flags)
    print(f'control interface flags = {flags}')
    t0 = time.perf_counter()
    failed = []

    def enable(on):
        """Flip the chosen primitive. Both take the same arguments."""
        fn = ctrl.ftRtdeInputEnable if args.api == 'ft_rtde' else ctrl.enableExternalFtSensor
        return fn(on, 0.0, [0.0] * 3, [0.0] * 3) if on else fn(False)

    def check(step):
        """Wait, then report. Returns True while the robot is still healthy."""
        time.sleep(args.settle)
        ps, es = recv.isProtectiveStopped(), recv.isEmergencyStopped()
        state = 'STOPPED' if ps else ('E-STOP' if es else 'ok')
        print(f'  [{time.perf_counter() - t0:6.1f}s] {step:<44s} {state}')
        if ps or es:
            failed.append(step)
        return not (ps or es)

    try:
        import dashboard_client
        db = dashboard_client.DashboardClient(args.ip)
        db.connect()
        print(f'PolyScope {db.polyscopeVersion()}   safety {db.safetymode()}')
        db.disconnect()
    except Exception as e:
        print(f'(dashboard unavailable: {e})')
    print(f'connected. protective stop now: {recv.isProtectiveStopped()}')
    print(f'payload {recv.getPayload():.3f} kg   TCP {np.round(ctrl.getTCPOffset(), 4)}')
    print(f'robot mode {recv.getRobotMode()}  safety mode {recv.getSafetyMode()}')
    print(f'settle {args.settle:.0f} s after each step; prime={"no" if args.no_prime else "yes"}; '
          f'order={args.order}\n')

    try:
        if not check('0. just connected, nothing called'):
            print('\nAlready stopped before any call -- the stop was latched from before, '
                  'or fires purely on connect. Clear it on the pendant and rerun.')
            return

        if not args.no_prime:
            # Write the register BEFORE anything reads it. If the controller is
            # still subscribed from an earlier run, this is the first value it
            # has seen this session.
            ok = ctrl.setExternalForceTorque([0.0] * 6)
            print(f'  setExternalForceTorque([0]*6) returned {ok}')
            if not check('1. primed the register'):
                return

        ok = enable(False)
        print(f'  {args.api} disable returned {ok}')
        if not check('2. disabled streamed F/T input'):
            return

        ctrl.zeroFtSensor()
        if not check('3. zeroFtSensor (the step that failed before)'):
            return

        ok = ctrl.setExternalForceTorque(list(recv.getActualTCPForce()))
        print(f'  setExternalForceTorque(actual) returned {ok}')
        if not check('4. wrote a real wrench while disabled'):
            return

        if args.order == 'enable-then-stream':
            ok = enable(True)
            print(f'  {args.api} enable returned {ok}')
            if not check('5. enabled, streaming has NOT started'):
                print('\n-> Enabling alone trips it: a value written moments ago is already '
                      'stale. The register must be kept fresh from the instant it is '
                      'enabled -- rerun with --order stream-then-enable.')
                return

        # Stream continuously, and (in stream-then-enable order) flip the enable on
        # from INSIDE the loop, so the register is never once left unwritten.
        print(f'  streaming at 500 Hz for {2 * args.settle:.0f} s, '
              f'enable goes on after {args.warmup} cycles...')
        t_end = time.perf_counter() + 2 * args.settle
        n, enabled, worst = 0, args.order == 'enable-then-stream', 0.0
        period = 1.0 / args.stream_hz if args.stream_hz > 0 else 0.0
        t_prev = time.perf_counter()
        while time.perf_counter() < t_end:
            t_start = ctrl.initPeriod()
            t_cycle = time.perf_counter()
            ctrl.setExternalForceTorque(list(recv.getActualTCPForce()))
            n += 1
            if not enabled and n >= args.warmup:
                ok = enable(True)
                print(f'  {args.api} enable mid-stream returned {ok}')
                enabled = True
            if recv.isProtectiveStopped():
                print(f'  STOPPED after {n} updates, while streaming')
                break
            if period:
                while time.perf_counter() - t_cycle < period:
                    pass                      # busy-wait: sleep() granularity is the jitter
            else:
                ctrl.waitPeriod(t_start)
            now = time.perf_counter()
            worst = max(worst, now - t_prev)
            t_prev = now
        rate = n / (2 * args.settle) if n else 0
        print(f'  sent {n} updates (~{rate:.0f} Hz), worst gap {1e3 * worst:.1f} ms')
        if not check('6. streamed continuously, enabled mid-stream'):
            print('\n-> Even continuous streaming trips it. Check the worst gap above: '
                  'if it is tens of ms, Python jitter is the problem.')
            return

        # Keep the register fresh while the disable lands, then stop.
        enable(False)
        for _ in range(args.warmup):
            t_start = ctrl.initPeriod()
            ctrl.setExternalForceTorque(list(recv.getActualTCPForce()))
            ctrl.waitPeriod(t_start)
        check('7. disabled from inside the stream')
    finally:
        try:
            enable(False)
        finally:
            ctrl.stopScript()
        print('\nfirst failing step: ' + (failed[0] if failed else 'none -- all steps passed'))


if __name__ == '__main__':
    main()
