import interface
import numpy as np
from tqdm import tqdm 
from einops import rearrange
import cv2, time, collections, argparse, os, wandb, torch, torch.nn as nn 

from env import Env, URPose
from agent.utils.logging import NoOpLogger, setup_logger
from agent.model.diffusion import build_diffusion_policy
from agent.utils.utils import load_checkpoint, get_stats, normalize, denormalize, resize_image

def get_actions(nets, stats, noise_scheduler, num_diffusion_iters, nimages, nagent_poses, action_horizon=16, action_dim=7, device='cuda'):
    B, image_features = 1, nets['vision_encoder'](nimages)

    obs_features = torch.cat([image_features, nagent_poses], dim=-1)
    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
    noisy_action = torch.randn(
        (B, action_horizon, action_dim), device=device)
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
    return naction # denormalize(naction, stats['actions'])

def evaluate(nets, noise_scheduler, stats, fps, save_dir, obs_horizon=1, action_horizon=16, num_diffusion_iters=100, img_size=128, device='cuda'):
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
    env.reset(home_pose)
    env.start() # start threads
    print("Starting evaluation loop...")
    obs_deque = collections.deque( [env.get_obs()] , maxlen=obs_horizon) # obs_horizon=1
    save_frames = []
    while True:
        
        if iface.update(env.dt) == -1:
            break # -1 indicates square is pressed and an error is thrown.
        
        images = np.stack([resize_image(x['image'], (img_size, img_size)) for x in obs_deque])/255.0 # - 0.5
        agent_poses, agent_grippers = np.stack([x['state']['pose'] for x in obs_deque]), np.stack([ [x['state']['gripper_width']] for x in obs_deque])
        curr_pose, curr_gripper = agent_poses[-1], agent_grippers[-1][0]
        agent_poses = np.concatenate( [agent_poses, agent_grippers], -1)
        # agent_poses = normalize(agent_poses, stats['states']) # normalize between -1 to 1.

        nimages = rearrange(torch.from_numpy(images).to(device, dtype=torch.float32), 't h w c -> t c h w')
        nagent_poses = torch.from_numpy(agent_poses).to(device, dtype=torch.float32) # txd
        
        with torch.no_grad():
            actions = get_actions(nets, stats, noise_scheduler, num_diffusion_iters, nimages, nagent_poses, action_horizon=action_horizon)
            
            start = obs_horizon - 1
            end = start + action_horizon
            actions = actions[start:end] # (action_horizon, action_dim)

            des_grippers_widths = actions[:, -1]
            # binary des_grippers, given threshold of 0.5. >0.5 is 1, otherwise 0
            des_grippers = (des_grippers_widths > 0.5).astype(int)

            for i in tqdm(range(len(actions)), desc=f'Open-loop execution'):
                delta_des_pose, des_gripper = actions[i, :-1], des_grippers[i]
                t0 = time.perf_counter()
                des_pose = curr_pose + delta_des_pose # s_0 + delta_t --> s_t
                des_pose = URPose(*des_pose)
                obs = env.step(
                    des_pose=des_pose,
                    des_gripper_state=des_gripper,
                )

                obs_deque.append(obs)
                sleep_time = 0.1
                time.sleep(sleep_time)
                save_frames.append(obs['image'].astype(np.uint8))
    # save video
    video_path = os.path.join(save_dir, 'evaluation_video.mp4')
    # create mp4 from a list of HxWxC images 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h,w,c = save_frames[0].shape
    
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    for frame in save_frames:
        # convert RGB to BGR for opencv
        frame_bgr = frame[:, :, ::-1]
        out.write(frame_bgr)
    out.release()

    env.close()
                
        
def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Policy Evaluation.')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--dataset_dir', type=str, default='/zfsauton/scratch/yiqiw2/100%/datasets')
    parser.add_argument('--ckpt_dir', type=str, default='/zfsauton/scratch/yiqiw2/100%/ckpts')
    parser.add_argument('--save_dir', type=str, default='/home/atkesonlab4/Desktop/YiqiProject/100%_Project/results')
    parser.add_argument('--device',    type=str,  default='cuda')
    parser.add_argument('--task',      type=str,  default='ethernet_unplug')
    parser.add_argument('--ckptname',    type=str,  default='ckpt_ep_190.pth')
    parser.add_argument('--ep_id',    type=int,  default=2)
    parser.add_argument('--fps',    type=int,  default=20)
    parser.add_argument('--img_size',    type=int,  default=128)
    parser.add_argument('--horizon',    type=int,  default=4)
    parser.add_argument('--num_diffusion_iters', type=int,  default=100)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset_dir = args.dataset_dir
    ckpt_dir = args.ckpt_dir
    save_dir = os.path.join(args.save_dir, args.task, f"h{args.horizon}", f"ep_{args.ep_id}")
    dataset_path, ckpt_path = os.path.join(dataset_dir, args.task+'_dataset'),os.path.join(ckpt_dir, args.task, f"h{args.horizon}", args.ckptname) 
    nets, _, _, _, noise_scheduler = build_diffusion_policy(  num_training_steps=0, device=args.device )
    nets = load_checkpoint(nets, ckpt_path, args.device)
    stats = get_stats(dataset_path)

    # create save dir if not exists and corresponding parent 
    os.makedirs(save_dir, exist_ok=True)

    evaluate(nets, noise_scheduler, stats, args.fps, save_dir,
        num_diffusion_iters=args.num_diffusion_iters, 
        img_size = args.img_size,
        action_horizon=args.horizon,
        device=args.device)