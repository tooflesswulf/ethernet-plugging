"""
Find which call triggers the "fieldbus input disconnected" protective stop.

NO MOTION AT ALL: this never enters force mode and never commands a move. It
walks through the external-F/T calls one at a time, waiting and checking the
protective-stop flag after each, so the offending step names itself.

The hypothesis it tests: an RTDE input field that has never been written in the
CURRENT session counts as disconnected. ft_rtde_input_enable is controller state
and survives the process that set it, so after a crashed run the controller is
reading external_force_torque while a fresh client has never written it -- and
it protective-stops a second or two after connecting, which lands on whatever
the script happens to be doing then (for us, zeroing the F/T).

If that is right, PRIMING the register (writing it once, immediately after
connecting) prevents the stop, and --no-prime reproduces it.

    python probe-ft-input.py              # prime first, then walk the steps
    python probe-ft-input.py --no-prime   # skip the prime: expected to fail

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
    print(f'settle {args.settle:.0f} s after each step; prime={"no" if args.no_prime else "yes"}\n')

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

        ok = ctrl.ftRtdeInputEnable(True, 0.0, [0.0] * 3, [0.0] * 3)
        print(f'  ftRtdeInputEnable(True) returned {ok}')
        if not check('5. enabled, streaming has NOT started'):
            print('\n-> Enabling alone trips it: the controller needs the register kept '
                  'fresh from the moment it is enabled. Stream before enabling.')
            return

        print(f'  streaming getActualTCPForce for {args.settle:.0f} s at 500 Hz...')
        t_end = time.perf_counter() + args.settle
        n = 0
        while time.perf_counter() < t_end:
            t_start = ctrl.initPeriod()
            ctrl.setExternalForceTorque(list(recv.getActualTCPForce()))
            n += 1
            ctrl.waitPeriod(t_start)
        print(f'  sent {n} updates ({n / args.settle:.0f} Hz)')
        check('6. streamed continuously')

        ctrl.ftRtdeInputEnable(False)
        check('7. disabled again')
    finally:
        try:
            ctrl.ftRtdeInputEnable(False)
        finally:
            ctrl.stopScript()
        print('\nfirst failing step: ' + (failed[0] if failed else 'none -- all steps passed'))


if __name__ == '__main__':
    main()
