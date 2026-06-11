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
from agent.utils.utils import load_checkpoint, resize_image


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


def evaluate(policy, fps, save_dir, obs_horizon=1, action_horizon=16, num_diffusion_iters=100, img_size=128, device='cuda'):
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
        dataset_path=None,
        save_interval=1.0 / fps,
    )
    env.reset(home_pose)
    env.start()  # start threads
    print("Starting evaluation loop...")
    obs_deque = collections.deque([env.get_obs()], maxlen=obs_horizon)  # obs_horizon=1
    save_frames = []
    while True:
        if iface.update(env.dt) == -1:
            break  # -1 indicates square is pressed and an error is thrown.

        images = np.stack([resize_image(x['image'], (img_size, img_size)) for x in obs_deque])
        agent_poses, agent_grippers = np.stack([x['state']['actual_pose'] for x in obs_deque]), np.stack([
            [x['state']['gripper_width']] for x in obs_deque])

        curr_pose, curr_gripper = agent_poses[-1], agent_grippers[-1][0]
        # raw observations: normalization happens inside the policy
        agent_poses = np.concatenate([agent_poses, agent_grippers], -1)

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
                sleep_time = 0.1
                time.sleep(sleep_time)
                save_frames.append(obs['image'].astype(np.uint8))

    # save video
    video_path = os.path.join(save_dir, 'evaluation_video.mp4')
    # create mp4 from a list of HxWxC images
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w, c = save_frames[0].shape

    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    for frame in save_frames:
        # convert RGB to BGR for opencv
        frame_bgr = frame  # [:, :, ::-1]
        out.write(frame_bgr)
    out.release()
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Policy Evaluation.')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--dataset_dir', type=str, default='/zfsauton/scratch/yiqiw2/100%/datasets')
    parser.add_argument('--ckpt_dir', type=str, default='/zfsauton/scratch/yiqiw2/100%/ckpts')
    parser.add_argument('--save_dir', type=str, default='/home/atkesonlab4/Desktop/YiqiProject/100%_Project/results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--task', type=str, default='ethernet_unplug_red_topdown')
    parser.add_argument('--ckptname', type=str, default='ckpt_ep_80.pth')
    parser.add_argument('--ckpt_path', type=str, default='.')
    parser.add_argument('--ep_id', type=int, default=11)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--horizon', type=int, default=16)
    parser.add_argument('--num_diffusion_iters', type=int, default=100)
    parser.add_argument('--state_dim', type=int, default=7)
    parser.add_argument('--action_dim', type=int, default=7)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_dir = args.dataset_dir
    ckpt_dir = args.ckpt_dir
    save_dir = os.path.join(args.save_dir, args.task, f"h{args.horizon}", f"ep_{args.ep_id}")
    dataset_path, ckpt_path = os.path.join(
        dataset_dir, args.task + '_dataset'), os.path.join(ckpt_dir, args.task, f"h{args.horizon}", args.ckptname)
    ckpt_path = args.ckpt_path
    # Normalization stats live in the checkpoint as buffers — no dataset access needed at eval time.
    policy = DiffusionPolicy(
        action_horizon=args.horizon,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        num_diffusion_iters=args.num_diffusion_iters,
    )
    policy = load_checkpoint(policy, ckpt_path, args.device)
    policy.eval()

    # create save dir if not exists and corresponding parent
    os.makedirs(save_dir, exist_ok=True)

    evaluate(policy, args.fps, save_dir,
             num_diffusion_iters=args.num_diffusion_iters,
             img_size=args.img_size,
             action_horizon=args.horizon,
             device=args.device)
