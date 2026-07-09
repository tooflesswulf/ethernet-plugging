from agent.eval.eval_realtime import EvalRealtimeChunking
from agent.utils.robot_utils import interrupt
import numpy as np
import argparse
import os

from util import URPose
from env import GRIP_OPEN, GRIP_CLOSED


class TeleoperationReset(EvalRealtimeChunking):
    # cable_drop_pos = URPose(-.0562, .6679, .0456, 2.508, 2.524, .936)
    cable_drop_pos = URPose(-0.04938359, 0.64969687, 0.07542422, -1.77502314, -1.78634705, -0.66244883)

    # Failed plugins wrench the cable out of the grippers: force z ramps past ~42N
    # before the cable slips, while successful ones stay under ~33N until release.
    # Only applies while gripping the cable (~9mm); the unplug phase grips the plug
    # head (~16mm) and legitimately reaches 55-75N, and an empty gripper closes to ~5mm.
    FZ_THRESH_N = 38.0            # force z above episode-start baseline
    FZ_TICKS = 3                  # consecutive control ticks above threshold
    CABLE_WIDTH_MM = (6.5, 12.0)  # gripper width range when holding the cable

    _fz_baseline = None
    _fz_count = 0
    _armed = True

    def get_action(self):
        if self.iface.dualsense.state.DpadLeft:
            last_pose, last_grip, _, _ = self.last_action
            if last_grip == GRIP_CLOSED:
                return self.reset_cable()

            print('Dpad-Left pressed, but gripper is open. Ignoring.')
        if self.detect_wrench_risk():
            print('Force z spike while gripping cable: failed plugin, resetting before wrench-out.')
            return self.reset_cable()
        return super().get_action()

    def detect_wrench_risk(self):
        obs = getattr(self, 'last_obs', None)
        if obs is None:
            return False
        fz = obs['state']['actual_force'][2]
        width = obs['state']['gripper_width']
        if self._fz_baseline is None:
            self._fz_baseline = fz
            return False

        if not self._armed:
            # Re-arm once the cable has left the gripper (reset seq / release opens it).
            self._armed = not (self.CABLE_WIDTH_MM[0] < width < self.CABLE_WIDTH_MM[1])
            return False

        _, last_grip, _, _ = self.last_action
        holding_cable = (last_grip == GRIP_CLOSED
                         and self.CABLE_WIDTH_MM[0] < width < self.CABLE_WIDTH_MM[1])
        if holding_cable and fz - self._fz_baseline > self.FZ_THRESH_N:
            self._fz_count += 1
        else:
            self._fz_count = 0

        if self._fz_count >= self.FZ_TICKS:
            self._armed = False
            self._fz_count = 0
            return True
        return False

    def reset_cable(self):
        # Start interrupt sequence. The seq methods queue up instructions behind the scenes,
        #   so custom logic needs to be 1. run through promise.then() and 2. be non-blocking.
        print('Starting cable reset sequence.')
        seq = interrupt(self)
        seq.move_relative([0, 0, .02, 0, 0, 0], speed=0.05)
        seq.move_to(self.cable_drop_pos)
        seq.gripper(GRIP_OPEN, settle_time=1.0)
        seq.move_to(self.home_pose) \
           .then(lambda _: self.buffer.clear())
        return self.get_action()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Teleoperation script for Ethernet Plugging task')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--control_freq', '--hz', type=float, default=20,
                        help='control/command frequency (Hz) for the real-time loop')
    parser.add_argument('--weight_decay', type=float, default=0.5,
                        help='recency-weighting rate (1/s) for ensembling overlapping chunks')
    args = parser.parse_args()

    teleop = TeleoperationReset(
        ckpt=args.ckpt, device=args.device,
        log_dir=None,
        control_freq=args.control_freq, weight_decay=args.weight_decay
    )
    teleop.run()
