from agent.eval.realtime_chunking import RealtimeActionChunkingBuffer
from agent.utils.robot_utils import get_actions, build_states, wait_for_circle, apply_gains
from agent.dataset.sequence import GripperStats
from agent.model.policy import DiffusionPolicy
import robot_execution
import collections
import numpy as np
import threading
import argparse
import torch
import time
import os


class EvalRealtimeChunking(robot_execution.RobotExecution):
    def __init__(self, ckpt, device='cuda', log_dir=None, control_freq=None, weight_decay=0.5, done_threshold=0.5):
        # Architecture config, weights, and normalization stats all come from the checkpoint.
        self.policy = DiffusionPolicy.from_checkpoint(ckpt, device)
        self.policy.eval()
        self.device = device
        # End the episode once the policy's predicted completion score crosses this.
        self.done_threshold = done_threshold
        grip = GripperStats(*self.policy.grip_stats)

        # Actions are spaced at the framerate the policy was trained on, and the chunking
        # buffer times them by control_dt, so mismatched rates replay the chunk too
        # fast/slow. Only override deliberately.
        if control_freq is None:
            control_freq = self.policy.framerate
        elif control_freq != self.policy.framerate:
            print(f'Warning: running at {control_freq}Hz, but the policy was trained at '
                  f'{self.policy.framerate}Hz. Actions will execute at the wrong speed.')

        # super().__init__() resets & starts the robot.
        super().__init__(
            path=log_dir,
            control_freq=control_freq,
            gwidth=grip.grip_width_mm,
            gforce=grip.grip_force_n,
            gspeed=grip.grip_speed_mmps,
            gpullback=grip.grip_pullback_mm,
        )

        self.buffer = RealtimeActionChunkingBuffer(action_dt=self.control_dt, weight_decay=weight_decay)
        self.prediction_thread = threading.Thread(target=self.prediction_loop)

    def pre_run(self):
        wait_for_circle(self.env, self.iface, close_gripper=False)
        print("Starting real-time chunked evaluation loop...")

        self.prediction_thread.start()

    def runtime_info(self):
        zf = self.last_obs['state']['filtered_force']
        kz = f'  Kz: {self.env.imp.K[2]:6.0f}' if self.policy.impedance_fields else ''
        print(self.buffer._chunk_count / (time.time() - self.env.t0), f'zforce: {zf[2]:.05f}{kz}', end='\r')

    def get_action(self):
        if self.buffer.is_empty():
            return None
        # Env.step blends from the previous command to this one over control_dt, so this
        # command is only reached a period from now: ask for the action due then.
        act = self.buffer.get_action(time.time() + self.control_dt)
        if act is None:
            return None
        des_pose, des_width, done, log_gains = act
        # Predicted impedance, split back into fields and blended over one command period
        # (no-op without an impedance head)
        apply_gains(self.env, {f: log_gains[6 * i:6 * (i + 1)]
                               for i, f in enumerate(self.policy.impedance_fields)},
                    ramp_s=self.control_dt)
        # End-of-episode signal: stop once the executed action's done score crosses the threshold.
        if self.policy.predict_done and done > self.done_threshold:
            print(f"Policy thinks the task is complete (done={done:.3f} > threshold={self.done_threshold:.3f}).")
            self.stop()
        return des_pose, des_width

    def prediction_loop(self):
        action_horizon = self.policy.action_horizon
        obs_horizon = self.policy.obs_horizon

        obs_deque = collections.deque(maxlen=obs_horizon)
        while not self.stop_event.is_set():
            t_obs = time.time()  # observation time the chunk is anchored to
            obs_deque.append(self.env.get_obs())
            if len(obs_deque) < obs_horizon:
                continue

            # get_actions builds images + the obs_fields state vector from the deque.
            with torch.no_grad():
                des_poses, des_grips, des_done, des_gains = get_actions(
                    self.policy, obs_deque, self.device, return_gains=True)
            # (H, 6 * n_fields) log10 gains, in policy.impedance_fields order
            des_gains = np.concatenate([des_gains[f] for f in self.policy.impedance_fields], axis=1) \
                if des_gains else None

            # the executable chunk starts at index obs_horizon-1, which aligns with t_obs.
            # The done score rides through the buffer so it's ensembled and thresholded at
            # execution time in get_action (not averaged over the chunk here).
            start = obs_horizon - 1
            end = start + action_horizon
            chnk = self.buffer.add_chunk(
                t_obs, des_poses[start:end], des_grips[start:end], des_done[start:end],
                None if des_gains is None else des_gains[start:end])
            obs_state = build_states(obs_deque, self.policy.obs_fields)  # for offline logging
            self.buffer.dolog(chnk, obs_state, time.time())


def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Policy Evaluation.')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='where to save robot log data + evaluation video (None disables logging)')
    parser.add_argument('--control_freq', '--hz', type=float, default=None,
                        help='control/command frequency (Hz) for the real-time loop '
                             "(default: the policy's training framerate)")
    parser.add_argument('--weight_decay', type=float, default=0.5,
                        help='recency-weighting rate (1/s) for ensembling overlapping chunks')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
    evaluation = EvalRealtimeChunking(
        ckpt=args.ckpt,
        log_dir=args.log_dir,
        control_freq=args.control_freq,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    evaluation.run()
