from agent.eval.realtime_chunking import RealtimeActionChunkingBuffer
from agent.utils.robot_utils import get_actions, wait_for_circle
from agent.model.policy import DiffusionPolicy
from agent.utils.utils import resize_image
from util import URPose
import robot_execution
import collections
import numpy as np
import threading
import argparse
import einops
import torch
import time
import os


GRIP_WIDTH_MM = 10
GRIP_FORCE_N = 40
GRIP_SPEED_MMPS = 50
GRIP_PULLBACK_MM = 5


class EvalRealtimeChunking(robot_execution.RobotExecution):
    def __init__(self, ckpt, device='cuda', log_dir=None, control_freq=20, weight_decay=0.5):
        # Architecture config, weights, and normalization stats all come from the checkpoint.
        self.policy = DiffusionPolicy.from_checkpoint(ckpt, device)
        self.policy.eval()
        self.device = device

        # super().__init__() resets & starts the robot.
        super().__init__(
            path=log_dir,
            control_freq=control_freq,
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
        des_pose, des_grip = act
        return URPose(*des_pose), int(round(des_grip))

    def prediction_loop(self):
        action_horizon = self.policy.action_horizon
        obs_horizon = self.policy.obs_horizon
        img_size = self.policy.img_size
        device = self.device

        obs_deque = collections.deque(maxlen=obs_horizon)
        while not self.stop_event.is_set():
            t_obs = time.time()  # observation time the chunk is anchored to
            obs_deque.append(self.env.get_obs())
            if len(obs_deque) < obs_horizon:
                continue

            images = np.stack([
                resize_image(x['image'], (img_size, img_size), flip_channel=True) for x in obs_deque])
            obs_state = np.stack([x['state']['actual_pose'] for x in obs_deque])
            agent_gwidth = np.stack([[x['state']['gripper_width']] for x in obs_deque])
            agent_force = np.stack([x['state']['actual_force'] for x in obs_deque])
            agent_gforce = np.stack([[x['state']['gripper_force']] for x in obs_deque])
            curr_pose, curr_gripper = obs_state[-1], agent_gwidth[-1][0]
            obs_state = np.c_[obs_state, agent_gwidth]  # raw; policy normalizes internally

            nimages = einops.rearrange(
                torch.from_numpy(images).to(device, dtype=torch.float32), 't h w c -> t c h w')
            nobs_state = torch.from_numpy(obs_state).to(device, dtype=torch.float32)
            with torch.no_grad():
                des_poses, des_grips = get_actions(self.policy, nimages, nobs_state, curr_pose, curr_gripper)

            # the executable chunk starts at index obs_horizon-1, which aligns with t_obs
            start = obs_horizon - 1
            end = start + action_horizon
            chnk = self.buffer.add_chunk(t_obs, des_poses[start:end], des_grips[start:end])
            self.buffer.dolog(chnk, obs_state, time.time())


def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Policy Evaluation.')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='where to save robot log data + evaluation video (None disables logging)')
    parser.add_argument('--control_freq', '--hz', type=float, default=10,
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
