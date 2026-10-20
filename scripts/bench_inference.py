"""
Unloaded policy inference speed: the realtime prediction loop's work (agent/eval/eval_realtime.py)
on a dummy observation, with no robot, env threads or camera. Compare the chunk rate it reports with
the one eval_realtime prints (chunks/s): a big gap means the eval process is starving inference.

    python scripts/bench_inference.py --ckpt ../ckpts/<run>/ckpt_final.pth
    python scripts/bench_inference.py --ckpt ... --worker    # through PolicyWorker (child process)
"""
import argparse
import collections
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent.model.policy import DiffusionPolicy
from agent.utils.robot_utils import get_actions
from util import URPose


def dummy_obs():
    pose = URPose(-0.147, 0.612, 0.184, 2.44, 2.44, 0.633)
    state = {'actual_pose': pose, 'target_pose': pose, 'gripper_width': 30.0, 'gripper_force': 0.0,
             'actual_force': URPose(*[0.0] * 6), 'filtered_force': URPose(*[0.0] * 6),
             'stiffness': np.r_[[300.0] * 3, [10.0] * 3]}
    image = np.random.default_rng(0).integers(0, 256, (480, 480, 3), dtype=np.uint8)  # camera frame, BGR
    return {'image': image, 'state': state}


def report(name, ms):
    ms = np.asarray(ms)
    print(f'{name:28s} p50 {np.median(ms):7.1f} ms   p95 {np.percentile(ms, 95):7.1f} ms   max {ms.max():7.1f} ms')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n', type=int, default=100, help='timed chunks (after 5 warm-up)')
    ap.add_argument('--worker', action='store_true', help='time get_actions through agent.eval.policy_worker')
    args = ap.parse_args()

    obs = collections.deque([dummy_obs()], maxlen=1)
    if args.worker:
        from agent.eval.policy_worker import PolicyWorker
        worker = PolicyWorker(args.ckpt, args.device)
        worker.wait_ready()
        for _ in range(5):
            worker.get_actions(obs)
        total = []
        for _ in range(args.n):
            t0 = time.perf_counter()
            worker.get_actions(obs)
            total.append(1e3 * (time.perf_counter() - t0))
        worker.close()
        report('get_actions via worker', total)
    else:
        policy = DiffusionPolicy.from_checkpoint(args.ckpt, args.device)
        policy.eval()
        print(f'device {args.device} ({torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu"}), '
              f'{type(policy.noise_scheduler).__name__} {len(policy.noise_scheduler.timesteps)} steps, '
              f'obs_fields {policy.obs_fields}')
        sync = torch.cuda.synchronize if args.device.startswith('cuda') else (lambda: None)
        # full get_actions, as the prediction loop calls it
        with torch.no_grad():
            for _ in range(5):
                get_actions(policy, obs, args.device, return_gains=True)
            total = []
            for _ in range(args.n):
                t0 = time.perf_counter()
                get_actions(policy, obs, args.device, return_gains=True)
                total.append(1e3 * (time.perf_counter() - t0))
            # the diffusion sample alone (network only)
            cond = {'rgb': torch.rand(1, policy.obs_horizon, 3, policy.img_size, policy.img_size, device=args.device),
                    'state': torch.rand(1, policy.obs_horizon, len(policy.state_min), device=args.device)}
            net = []
            for _ in range(args.n):
                sync()
                t0 = time.perf_counter()
                policy.predict_action(cond)
                sync()
                net.append(1e3 * (time.perf_counter() - t0))
        report('get_actions (full)', total)
        report('  diffusion sample only', net)
    rate = 1e3 / np.median(total)
    print(f'=> unloaded chunk rate ~{rate:.1f} chunks/s '
          f'({rate * policy_horizon_s(args.ckpt):.0f} overlapping chunks over the 0.8 s horizon)')


def policy_horizon_s(ckpt):
    cfg = torch.load(ckpt, map_location='cpu', weights_only=False)['config']
    return cfg['action_horizon'] / cfg.get('framerate', 20.0)


if __name__ == '__main__':
    main()
