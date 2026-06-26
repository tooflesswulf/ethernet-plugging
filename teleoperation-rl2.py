from agent.eval.eval_realtime import EvalRealtimeChunking
import numpy as np
import argparse
import os

class TeleoperationRL(EvalRealtimeChunking):
    # No init-- inherited from EvalRealtimeChunking.
    def pre_reset(self):
        print('============ EVAL+TELEOPERATION ============')
        print('Control inputs (above) add offsets to base policy.')
        print('Dpad-Left rewinds actions.')
        print('- press = rewind 1s')
        print('- hold = rewind until released')
        print('============================================')
        print()

    undo_action_buffer = []
    action_hist = []
    def get_action(self):
        if self.iface.dualsense.state.DpadLeft:
            # Interruption signal - Undo last 1s, hold for longer.
            if len(self.undo_action_buffer) == 0:
                undo_count = int(self.control_freq)
                self.undo_action_buffer = self.action_hist[-undo_count:]
                self.action_hist = self.action_hist[:-undo_count]
            elif len(self.undo_action_buffer) == 1 and len(self.action_hist) > 0:
                # If undo about to finish but we want more, add 1 at a time.
                next_act = self.action_hist.pop()
                self.undo_action_buffer.insert(0, next_act)

        if len(self.undo_action_buffer) > 0:
            action = self.undo_action_buffer.pop()
            if len(self.undo_action_buffer) == 0:
                # Finished undo-ing actions, empty prediction buffer
                self.buffer.clear()
            return action

        nn_action = super().get_action()
        des_pose, des_grip, _, _ = self._unshortcut_action(nn_action)
        des_pose = self.iface.residual_action(np.array(des_pose), self.control_dt)
        self.action_hist.append((des_pose, des_grip))
        return des_pose, des_grip

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Teleoperation script for Ethernet Plugging task')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str,
                        default='/home/atkesonlab4/Desktop/YiqiProject/100%_Project/dataset/ethernet_plugin_unplug_rl',
                        help='Base dataset directory')
    parser.add_argument('--control_freq', '--hz', type=float, default=20,
                        help='control/command frequency (Hz) for the real-time loop')
    parser.add_argument('--weight_decay', type=float, default=0.5,
                        help='recency-weighting rate (1/s) for ensembling overlapping chunks')
    parser.add_argument('-d', '--debug', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if not args.debug:
        indices = [
            int(d.removeprefix('episode'))
            for d in os.listdir(args.log_dir)
            if d.startswith('episode') and d.removeprefix('episode').isdigit()
        ] if os.path.exists(args.log_dir) else []
        args.id = max(indices, default=0) + 1
        print(f'Auto-selected episode ID: {args.id}')
        print(f"Saving data to: {args.log_dir}, Episode {args.id}")

        os.makedirs(args.log_dir, exist_ok=True)
        path = args.log_dir
    else:
        path = None
    teleop = TeleoperationRL(
        ckpt=args.ckpt, device=args.device,
        log_dir=path,
        control_freq=args.control_freq, weight_decay=args.weight_decay
    )
    teleop.run()
