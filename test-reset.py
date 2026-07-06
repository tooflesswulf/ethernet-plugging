from agent.eval.eval_realtime import EvalRealtimeChunking
import numpy as np
import argparse
import os

class TeleoperationReset(EvalRealtimeChunking):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Teleoperation script for Ethernet Plugging task')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str,
                        default='/home/atkesonlab4/Desktop/YiqiProject/100%_Project/dataset/ethernet_plugin_unplug_rl2',
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
    teleop = TeleoperationReset(
        ckpt=args.ckpt, device=args.device,
        log_dir=path,
        control_freq=args.control_freq, weight_decay=args.weight_decay
    )
    teleop.run()
