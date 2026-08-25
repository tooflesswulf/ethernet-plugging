from agent.utils.robot_utils import get_actions, wait_for_circle
from agent.dataset.sequence import GripperStats
from agent.model.policy import DiffusionPolicy
import robot_execution
import collections
import argparse
import torch
import os


class EvalPolicySerialChunks(robot_execution.RobotExecution):
    def __init__(self, ckpt, device='cuda', log_dir=None, control_freq=None, done_threshold=0.5):
        # Architecture config, weights, and normalization stats all come from the checkpoint.
        self.policy = DiffusionPolicy.from_checkpoint(ckpt, device)
        self.policy.eval()
        self.device = device
        self.done_threshold = done_threshold
        grip = GripperStats(*self.policy.grip_stats)

        # Actions are spaced at the framerate the policy was trained on; replaying them
        # at any other rate distorts the demonstrated speed. Only override deliberately.
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

        self.action_chunk = []
        self.obs_deque = collections.deque([self.env.get_obs()], maxlen=self.policy.obs_horizon)  # obs_horizon=1

    def pre_run(self):
        wait_for_circle(self.env, self.iface, close_gripper=False)
        print("Starting evaluation...")

    def post_step(self, obs, act):
        self.obs_deque.append(obs)

    def runtime_info(self):
        return super().runtime_info()

    def get_action(self):
        if len(self.action_chunk) == 0:
            self.obs_deque.append(self.env.get_obs())
            self.action_chunk = self.do_prediction()[1:]
        des_pose, des_width, done = self.action_chunk.pop(0)
        if done > self.done_threshold:
            print(f"Policy thinks the task is complete (done={done:.3f} > threshold={self.done_threshold:.3f}).")
            self.stop()
        return des_pose, des_width

    def do_prediction(self):
        obs_horizon = self.policy.obs_horizon
        action_horizon = self.policy.action_horizon

        # get_actions builds images + the obs_fields state vector from the deque.
        with torch.no_grad():
            des_poses, des_widths, des_done = get_actions(self.policy, self.obs_deque, self.device)
            start = obs_horizon - 1
            end = start + action_horizon
            des_poses, des_widths, des_done = des_poses[start:end], des_widths[start:end], des_done[start:end]

        return [(p, w, d) for p, w, d in zip(des_poses, des_widths, des_done)]


def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Policy Evaluation.')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='where to save robot log data + evaluation video (None disables logging)')
    parser.add_argument('--control_freq', '--hz', type=float, default=None,
                        help='control/command frequency (Hz) for the real-time loop '
                             "(default: the policy's training framerate)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
    evaluation = EvalPolicySerialChunks(
        ckpt=args.ckpt,
        log_dir=args.log_dir,
        control_freq=args.control_freq,
        device=args.device,
    )
    evaluation.run()
