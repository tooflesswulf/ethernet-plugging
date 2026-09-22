"""
Find which call triggers the "fieldbus input disconnected" protective stop.

NO MOTION AT ALL: this never enters force mode and never commands a move. It
walks through the external-F/T calls one at a time, waiting and checking the
protective-stop flag after each, so the offending step names itself.

What it established so far (2026-09-22): writing the register once is NOT enough.
Step 4 wrote a real wrench and passed; step 5 then called ftRtdeInputEnable(True)
with that value ~3 s old and the controller protective-stopped immediately. So
the register must be kept FRESH from the instant it is enabled, not merely
non-empty -- ft_rtde_input_enable subscribes the controller to a value it expects
every cycle.

The default order therefore streams first and flips the enable on from inside
the streaming loop, and disables from inside it too.

    python probe-ft-input.py                              # the fix
    python probe-ft-input.py --order enable-then-stream   # reproduces the stop

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
    args = ap.parse_args()

    recv = rtde_receive.RTDEReceiveInterface(args.ip)
    ctrl = rtde_control.RTDEControlInterface(args.ip)
    t0 = time.perf_counter()
    failed = []

    def check(step):
        """Wait, then report. Returns True while the robot is still healthy."""
        time.sleep(args.settle)
        ps, es = recv.isProtectiveStopped(), recv.isEmergencyStopped()
        state = 'STOPPED' if ps else ('E-STOP' if es else 'ok')
        print(f'  [{time.perf_counter() - t0:6.1f}s] {step:<44s} {state}')
        if ps or es:
            failed.append(step)
        return not (ps or es)

    print(f'connected. protective stop now: {recv.isProtectiveStopped()}')
    print(f'payload {recv.getPayload():.3f} kg   TCP {np.round(ctrl.getTCPOffset(), 4)}')
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

        ok = ctrl.ftRtdeInputEnable(False)
        print(f'  ftRtdeInputEnable(False) returned {ok}')
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
            ok = ctrl.ftRtdeInputEnable(True, 0.0, [0.0] * 3, [0.0] * 3)
            print(f'  ftRtdeInputEnable(True) returned {ok}')
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
        t_prev = time.perf_counter()
        while time.perf_counter() < t_end:
            t_start = ctrl.initPeriod()
            ctrl.setExternalForceTorque(list(recv.getActualTCPForce()))
            n += 1
            if not enabled and n >= args.warmup:
                ok = ctrl.ftRtdeInputEnable(True, 0.0, [0.0] * 3, [0.0] * 3)
                print(f'  ftRtdeInputEnable(True) mid-stream returned {ok}')
                enabled = True
            if recv.isProtectiveStopped():
                print(f'  STOPPED after {n} updates, while streaming')
                break
            ctrl.waitPeriod(t_start)
            now = time.perf_counter()
            worst = max(worst, now - t_prev)
            t_prev = now
        print(f'  sent {n} updates, worst gap {1e3 * worst:.1f} ms')
        if not check('6. streamed continuously, enabled mid-stream'):
            print('\n-> Even continuous streaming trips it. Check the worst gap above: '
                  'if it is tens of ms, Python jitter is the problem.')
            return

        # Keep the register fresh while the disable lands, then stop.
        ctrl.ftRtdeInputEnable(False)
        for _ in range(args.warmup):
            t_start = ctrl.initPeriod()
            ctrl.setExternalForceTorque(list(recv.getActualTCPForce()))
            ctrl.waitPeriod(t_start)
        check('7. disabled from inside the stream')
    finally:
        try:
            ctrl.ftRtdeInputEnable(False)
        finally:
            ctrl.stopScript()
        print('\nfirst failing step: ' + (failed[0] if failed else 'none -- all steps passed'))


if __name__ == '__main__':
    main()
