from agent.eval.eval_realtime import EvalRealtimeChunking
from agent.utils.robot_utils import interrupt
import numpy as np
import argparse
import os

from util import URPose
from env import GRIP_OPEN, GRIP_CLOSED


class TeleoperationReset(EvalRealtimeChunking):
    # cable_drop_pos = URPose(-.0562, .6679, .0456, 2.508, 2.524, .936)
    cable_drop_pos = URPose(-0.03938359, 0.64969687, 0.07542422, -1.77502314, -1.78634705, -0.66244883)

    # Failed plugins wrench the cable out of the grippers: the plug catches on the
    # socket rim (first contact at z=68.6-71.1mm vs 55.5-64.0mm when it enters the
    # socket cleanly) and force z then ramps past ~42N before the cable slips, while
    # successful plugins stay under ~33N until release. The 38N trigger is therefore
    # gated on a rim-height first contact, so a clean insertion can push harder than
    # 38N without triggering; a 55N backstop covers wrench-out from a low contact.
    # Only applies while gripping the cable (~9mm); the unplug phase grips the plug
    # head (~16mm) and legitimately reaches 55-75N, and an empty gripper closes to ~5mm.
    FZ_THRESH_N = 38.0            # force z above episode-start baseline, rim contact
    FZ_BACKSTOP_N = 55.0          # trigger regardless of contact height
    FZ_TICKS = 3                  # consecutive control ticks above threshold
    CONTACT_RISE_N = 20.0         # force z rise marking first contact (2 ticks)
    CONTACT_Z_M = 0.0665          # first contact above this = rim hit, below = in socket
    CABLE_WIDTH_MM = (6.5, 12.0)  # gripper width range when holding the cable

    _fz_baseline = None
    _contact_z = None
    _contact_count = 0
    _fz_count = 0
    _armed = True

    def get_action(self):
        if self.iface.dualsense.state.DpadLeft:
            last_pose, last_grip, _, _ = self.last_action
            if last_grip == GRIP_CLOSED:
                return self.reset_cable()

            print('Dpad-Left pressed, but gripper is open. Ignoring.')
        # if self.detect_wrench_risk():
        #     print('Force z spike while gripping cable: failed plugin, resetting before wrench-out.')
        #     return self.reset_cable()
        return super().get_action()

    def runtime_info(self):
        obs = self.last_obs
        print(f"force_z: {obs['state']['filtered_force'][2]:7.2f}", end='\r')

    def reset_cable(self):
        # Start interrupt sequence. The seq methods queue up instructions behind the scenes,
        #   so custom logic needs to be 1. run through promise.then() and 2. be non-blocking.
        print('Starting cable reset sequence.')
        seq = interrupt(self)
        seq.move_relative([0, 0, .02, 0, 0, 0], speed=0.05)
        seq.move_to(self.cable_drop_pos)
        seq.gripper(GRIP_OPEN, settle_time=1.0)
        seq.move_to(self.home_pose) \
            .then(lambda _: self.buffer.clear()) \
            .then(lambda _: self.env.ctrl.zeroFtSensor())
        return self.get_action()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Teleoperation script for Ethernet Plugging task')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--control_freq', '--hz', type=float, default=20,
                        help='control/command frequency (Hz) for the real-time loop')
    parser.add_argument('--weight_decay', type=float, default=0.5,
                        help='recency-weighting rate (1/s) for ensembling overlapping chunks')
    parser.add_argument('--log', type=str, default=None, help='log directory')
    args = parser.parse_args()

    if args.log is not None:
        os.makedirs(args.log, exist_ok=True)
    teleop = TeleoperationReset(
        ckpt=args.ckpt, device=args.device,
        log_dir=args.log,
        control_freq=args.control_freq, weight_decay=args.weight_decay
    )
    teleop.run()
