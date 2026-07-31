from agent.eval.realtime_chunking import RealtimeActionChunkingBuffer
from agent.utils.robot_utils import get_actions, build_states, wait_for_circle
from agent.dataset.sequence import GripperStats
from agent.model.policy import DiffusionPolicy
import robot_execution
import collections
import threading
import argparse
import torch
import time
import os


class EvalRealtimeChunking(robot_execution.RobotExecution):
    def __init__(self, ckpt, device='cuda', log_dir=None, control_freq=20, weight_decay=0.5, done_threshold=0.5):
        # Architecture config, weights, and normalization stats all come from the checkpoint.
        self.policy = DiffusionPolicy.from_checkpoint(ckpt, device)
        self.policy.eval()
        self.device = device
        # End the episode once the policy's predicted completion score crosses this.
        self.done_threshold = done_threshold
        grip = GripperStats(*self.policy.grip_stats)

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

    def get_action(self):
        if self.buffer.is_empty():
            return None
        act = self.buffer.get_action(time.time())
        if act is None:
            return None
        des_pose, des_width, done = act
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
                des_poses, des_grips, des_done = get_actions(self.policy, obs_deque, self.device)

            # the executable chunk starts at index obs_horizon-1, which aligns with t_obs.
            # The done score rides through the buffer so it's ensembled and thresholded at
            # execution time in get_action (not averaged over the chunk here).
            start = obs_horizon - 1
            end = start + action_horizon
            chnk = self.buffer.add_chunk(
                t_obs, des_poses[start:end], des_grips[start:end], des_done[start:end])
            obs_state = build_states(obs_deque, self.policy.obs_fields)  # for offline logging
            self.buffer.dolog(chnk, obs_state, time.time())


def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Policy Evaluation.')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='where to save robot log data + evaluation video (None disables logging)')
    parser.add_argument('--control_freq', '--hz', type=float, default=20,
                        help='control/command frequency (Hz) for the real-time loop')
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
