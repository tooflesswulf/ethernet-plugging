from agent.utils.robot_utils import get_actions, wait_for_circle, interrupt
from agent.dataset.sequence import GripperStats
from env import URPose, GRIP_OPEN, GRIP_CLOSED
from agent.model.policy import DiffusionPolicy
import robot_execution
import collections
import argparse
import torch
import os

class DebugPolicy(robot_execution.RobotExecution):
    def __init__(self, ckpt, device='cuda', control_freq=None):
        # Architecture config, weights, and normalization stats all come from the checkpoint.
        self.policy = DiffusionPolicy.from_checkpoint(ckpt, device)
        self.policy.eval()
        self.device = device
        self.done_threshold = 0.5
        grip = GripperStats(*self.policy.grip_stats)

        print

        # Actions are spaced at the framerate the policy was trained on; replaying them
        # at any other rate distorts the demonstrated speed. Only override deliberately.
        if control_freq is None:
            control_freq = self.policy.framerate
        elif control_freq != self.policy.framerate:
            print(f'Warning: running at {control_freq}Hz, but the policy was trained at '
                  f'{self.policy.framerate}Hz. Actions will execute at the wrong speed.')

        # super().__init__() resets & starts the robot.
        super().__init__(
            path=None,
            control_freq=control_freq,
            gwidth=grip.grip_width_mm,
            gforce=grip.grip_force_n,
            gspeed=grip.grip_speed_mmps,
            gpullback=grip.grip_pullback_mm,
        )

        self.action_chunk = []
        self.obs_deque = collections.deque([self.env.get_obs()], maxlen=self.policy.obs_horizon)  # obs_horizon=1

    def pre_run(self):
        seq = interrupt(self)
        port_pose = URPose(x=-0.1205, y=0.4358, z=0.0489, rx=1.7429, ry=1.5768, rz=-0.7897)
        seq.move_to(port_pose)
        seq.teleop(until=lambda: self.iface.dualsense.state.Cross) \
            .then(lambda x: print('Starting policy evaluation'))

    def runtime_info(self):
        zf = self.last_obs['state']['filtered_force']
        print(f'Force={zf[2]:.05f}', end='\r')

    def get_action(self):
        if len(self.action_chunk) == 0:
            seq = interrupt(self)
            seq.teleop(until=lambda: self.iface.dualsense.state.Cross) \
                .then(lambda x: self.predict_and_add_chunk())
            return self.get_action()
            # self.obs_deque.append(self.env.get_obs())
            # # self.action_chunk = self.do_prediction()[1:]
            # self.action_chunk = self.do_prediction()[0:4]
            # return self.wait_for_cross()
        des_pose, des_width, done = self.action_chunk.pop(0)
        if done > self.done_threshold:
            print(f"Policy thinks the task is complete (done={done:.3f} > threshold={self.done_threshold:.3f}).")
            self.stop()
        return des_pose, des_width

    def wait_for_cross(self):
        seq = interrupt(self)
        seq.teleop(until=lambda: self.iface.dualsense.state.Cross)
        return self.get_action()

    def predict_and_add_chunk(self):
        self.obs_deque.append(self.env.get_obs())
        self.action_chunk = self.do_prediction()[:2]

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


if __name__ == '__main__':
    ckpt = '../ckpts/slide-in/force/ckpt_final.pth'
    evaluation = DebugPolicy(ckpt=ckpt)
    evaluation.run()