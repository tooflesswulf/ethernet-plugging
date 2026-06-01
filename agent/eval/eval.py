import interface
import numpy as np
from tqdm import tqdm 
from einops import rearrange
import collections, argparse, os, wandb, torch, torch.nn as nn 

from env import Env, URPose
from agent.utils.logging import NoOpLogger, setup_logger
from agent.model.diffusion import build_diffusion_policy
from agent.utils.utils import load_checkpoint, get_stats, normalize, denormalize

def get_actions(nets, stats, noise_scheduler, num_diffusion_iters, nimages, nagent_poses, pred_horizon=16, action_dim=7, device='cuda'):
    B, image_features = 1, nets['vision_encoder'](nimages)

    obs_features = torch.cat([image_features, nagent_poses], dim=-1)
    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
    noisy_action = torch.randn(
        (B, pred_horizon, action_dim), device=device)
    naction = noisy_action

    # init scheduler
    noise_scheduler.set_timesteps(num_diffusion_iters)

    for k in noise_scheduler.timesteps:
        # predict noise
        noise_pred = nets['noise_pred_net'](
            sample=naction,
            timestep=k,
            global_cond=obs_cond
        )

        # inverse diffusion step (remove noise)
        naction = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=naction
        ).prev_sample

    # unnormalize action
    naction = naction.detach().to('cpu').numpy()[0]
    return denormalize(naction, stats['actions'])

def evaluate(nets, noise_scheduler, stats, fps, ep_id=0, obs_horizon=1, action_horizon=16, num_diffusion_iters=100, device='cuda'):
    home_pose = URPose(-0.125,0.545,0.305,2.44,2.44,0.653, )
    iface = interface.DualSenseInterface(
        home_pose,
        xyzspeed=0.01,
        rpyspeed=0.1,
    ) # press square to end evaluation
    env = Env(
        robot_ip="192.168.0.100",
        gripper_ip="192.168.0.20",
        camera_crop_mode=1,
        dataset_path=None,
        save_interval=1.0 / fps, 
    )
    obs = (home_pose) # start camera, robot go home, gripper open
    obs_deque = collections.deque( [obs] , maxlen=obs_horizon) # obs_horizon=1
    print("Starting evaluation loop...")
    env.start() # start threads

    while True:
        
        if iface.update(env.dt) == -1:
            break # -1 indicates square is pressed and an error is thrown.
        
        images = np.stack([x['rgb'] for x in obs_deque])/255.0 - 0.5
        agent_poses, agent_grippers = np.stack([x['actual_pose'] for x in obs_deque]), np.stack([ [x['gripper_width']] for x in obs_deque])
        curr_pose, curr_gripper = agent_poses[-1], agent_grippers[-1][0]
        agent_poses = np.concatenate( [agent_poses, agent_grippers], -1)
        agent_poses = normalize(agent_poses, stats['states']) # normalize between -1 to 1.

        nimages = rearrange(torch.from_numpy(images).to(device, dtype=torch.float32), 't h w c -> t c h w')
        nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)

        with torch.no_grad():
            actions = get_actions(nets, stats, noise_scheduler, num_diffusion_iters, nimages, nagent_poses)
            start = obs_horizon - 1
            end = start + action_horizon
            actions = actions[start:end] # (action_horizon, action_dim)
            des_poses = curr_pose + np.consum(actions[:, :-1])
            des_grippers_widths = actions[:, -1]
            assert False, f"Convert des_gripper_widths to des_grippers"
            des_grippers = None
            for i, (des_pose, des_gripper) in enumerate(zip(des_poses, des_grippers)): # open loop execution of actions:
                obs = env.step(
                    des_pose=des_pose,
                    des_gripper_state=des_gripper,
                )
                obs_deque.append(obs)


def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Policy Evaluation.')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--device',    type=str,  default='cuda')
    parser.add_argument('--task',      type=str,  default='ethernet_unplug')
    parser.add_argument('--ckptname',    type=str,  default='ckpt_ep_150.pth')
    parser.add_argument('--ep_id',    type=int,  default=0)
    parser.add_argument('--fps',    type=int,  default=20)
    parser.add_argument('--num_diffusion_iters', type=int,  default=100)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset_dir = '/zfsauton/scratch/yiqiw2/100%/datasets'
    ckpt_dir = '/zfsauton/scratch/yiqiw2/100%/ckpts'
    dataset_path, ckpt_path = os.path.join(dataset_dir, args.task),os.path.join(ckpt_dir, args.task, args.ckptname) 
    nets, _, _, _, noise_scheduler = build_diffusion_policy(  num_training_steps=0, device=args.device )
    nets = load_checkpoint(nets, ckpt_path, args.device)
    stats = get_stats(dataset_path)
    evaluate(nets, noise_scheduler, stats, args.fps, args.ep_id, num_diffusion_iters=args.num_diffusion_iters, device=args.device)