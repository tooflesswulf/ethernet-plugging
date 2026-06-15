import interface
import numpy as np
from tqdm import tqdm
from einops import rearrange
import cv2
import time
import collections
import argparse
import os
import wandb
import torch
import torch.nn as nn

from env import Env, URPose
from agent.utils.logging import NoOpLogger, setup_logger
from agent.model.policy import DiffusionPolicy
from agent.utils.utils import resize_image

GRIP_WIDTH_MM = 8
GRIP_FORCE_N = 40
GRIP_SPEED_MMPS = 50
GRIP_PULLBACK_MM = 10


def get_actions(policy, num_diffusion_iters, nimages, nagent_poses, curr_pose, curr_gripper_width):
    """
    nimages:      (T, C, H, W) in [0, 255]
    nagent_poses: (T, state_dim) raw/unnormalized
    Returns (des_poses (H, 6) absolute [trans, rotvec], des_widths (H,)) ready to execute.
    """

    conditions = {
        'rgb': (nimages / 255.0).unsqueeze(0),  # (1, T, C, H, W)
        'state': nagent_poses.unsqueeze(0),     # (1, T, state_dim); policy normalizes internally
    }
    naction = policy.predict_action(conditions, num_inference_steps=num_diffusion_iters)
    naction = naction.detach().to('cpu').numpy()[0]

    # integrate deltas (per the policy's action_mode) into absolute poses + widths
    return policy.integrate_actions(naction, curr_pose, curr_gripper_width)


def wait_for_circle(env, iface, disable=False):
    freq = 250
    print('Waiting the circle ...')
    while True and not disable:
        flag = iface.update(1 / freq)
        if flag == -1:
            raise RuntimeError('Square pressed, exiting.')

        des_pose = URPose(*iface.target_pose)
        des_gripper = iface.gripper_state
        # obs = env.step(
        #     des_pose=des_pose,
        #     des_gripper_state=des_gripper,
        #     des_zforce=iface.target_zforce,
        #     adaptive_mode=iface.adaptive_mode,
        # )
        if des_gripper == 1:
            break
        time.sleep(1 / 250)

    time.sleep(0.1)
    
    env.gripper.wait_idle()
    time.sleep(1)


def evaluate(policy, log_dir=None, fps=20, device='cuda'):
    # network-specific parameters come from the loaded checkpoint
    obs_horizon = policy.obs_horizon
    action_horizon = policy.action_horizon
    num_diffusion_iters = policy.num_diffusion_iters
    img_size = policy.img_size

    # home_pose = URPose(-0.125,0.545,0.305,2.44,2.44,0.653, )
    home_pose = URPose(-0.147, 0.612, 0.184, 2.44, 2.44, 0.633)  # low-position (cable easy to see, Yiqi)
    iface = interface.DualSenseInterface(
        home_pose,
        xyzspeed=0.01,
        rpyspeed=0.1,
    )  # press square to end evaluation
    env = Env(
        robot_ip="192.168.0.100",
        gripper_ip="192.168.0.20",
        camera_crop_mode=1,
        dataset_path=log_dir,  # None disables robot data logging
        save_interval=1.0 / fps,
        gforce=GRIP_FORCE_N,
        gwidth=GRIP_WIDTH_MM,
        gspeed=GRIP_SPEED_MMPS,
        gpullback=GRIP_PULLBACK_MM,
    )
    env.reset(home_pose)
    env.start()  # start threads

    target_ix = 0
    g_thr = 15

    wait_for_circle(env, iface, disable=False)
    print("Starting evaluation loop...")
  
    obs_deque = collections.deque([env.get_obs()], maxlen=obs_horizon)  # obs_horizon=1
    save_frames = []
    while True:
        if iface.update(.1) == -1:
            break  # -1 indicates square is pressed and an error is thrown.

        images = np.stack([resize_image(x['image'], (img_size, img_size), flip_channel=True) for x in obs_deque])
        
        agent_poses = np.stack([x['state']['actual_pose'] for x in obs_deque])
        agent_gwidth = np.stack([[x['state']['gripper_width']] for x in obs_deque])
        agent_force = np.stack([x['state']['actual_force'] for x in obs_deque])
        agent_gforce = np.stack([[x['state']['gripper_force']] for x in obs_deque])

        curr_pose, curr_gripper = agent_poses[-1], agent_gwidth[-1][0]
        # raw observations: normalization happens inside the policy
        # agent_poses = np.c_[agent_poses, agent_gwidth, agent_force, agent_gforce, target_ix]
        agent_poses = np.c_[agent_poses, agent_gwidth]

        nimages = rearrange(torch.from_numpy(images).to(device, dtype=torch.float32), 't h w c -> t c h w')
        nagent_poses = torch.from_numpy(agent_poses).to(device, dtype=torch.float32)  # txd
        with torch.no_grad():
            des_poses, des_widths = get_actions(
                policy, num_diffusion_iters, nimages, nagent_poses, curr_pose, curr_gripper)
            start = obs_horizon - 1
            end = start + action_horizon
            des_poses, des_widths = des_poses[start:end], des_widths[start:end]
            print('des grippers:', des_widths)

            for i in tqdm(range(len(des_poses)), desc=f'Open-loop execution'):
                t0 = time.perf_counter()
                des_pose = URPose(*des_poses[i])  # already absolute: deltas integrated by the policy
                obs = env.step(
                    des_pose=des_pose,
                    des_gripper_state=des_widths[i],
                )

                obs_deque.append(obs)
                sleep_time = 0.2
                time.sleep(sleep_time)
                save_frames.append(obs['image'].astype(np.uint8))

    # save video
    if log_dir is not None and save_frames:
        video_path = os.path.join(log_dir, 'evaluation_video.mp4')
        # create mp4 from a list of HxWxC images
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w, c = save_frames[0].shape

        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        for frame in save_frames:
            # convert RGB to BGR for opencv
            frame_bgr = frame  # [:, :, ::-1]
            out.write(frame_bgr)
        out.release()
        print(f"Saved evaluation video → {video_path}")
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Policy Evaluation.')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='where to save robot log data + evaluation video (None disables logging)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Architecture config, weights, and normalization stats all come from the checkpoint.
    policy = DiffusionPolicy.from_checkpoint(args.ckpt, args.device)
    policy.eval()

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)

    evaluate(policy, log_dir=args.log_dir, device=args.device)
