from agent.eval.eval_realtime import EvalRealtimeChunking
from agent.utils.robot_utils import reset_gripper, reset_to_position, reset_relative, reset_wait, reset_teleop
import numpy as np
import argparse
import os

from util import URPose


class TeleoperationReset(EvalRealtimeChunking):
    # cable_drop_pos = URPose(-.0562, .6679, .0456, 2.508, 2.524, .936)
    cable_drop_pos = URPose(-0.04938359, 0.64969687 , 0.07542422 ,-1.77502314 ,-1.78634705, -0.66244883)

    def get_action(self):
        if self.iface.dualsense.state.DpadLeft:
            # Start reset sequence. The `reset_*` calls queue up instructions behind the scenes,
            #  so custom logic needs to be 1. run through promise.then() and 2. be non-blocking.
            reset_relative(self, [0, 0, .02, 0, 0, 0], speed=0.05)
            reset_to_position(self, self.cable_drop_pos)
            reset_gripper(self, 0, settle_time=1.0)
            reset_to_position(self, self.home_pose) \
               .then(lambda _: self.buffer.clear())
            return self.get_action()
        return super().get_action()


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
