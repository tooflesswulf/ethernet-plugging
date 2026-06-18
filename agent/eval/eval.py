import interface
import numpy as np
from tqdm import tqdm
from einops import rearrange
import cv2
import time
import threading
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
from agent.eval.realtime_chunking import RealtimeActionChunkingBuffer

GRIP_WIDTH_MM = 8
GRIP_FORCE_N = 40
GRIP_SPEED_MMPS = 50
GRIP_PULLBACK_MM = 5


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


def wait_for_circle(env, iface, close_gripper=False):
    freq = 250
    print('Waiting the circle ...')
    while True:
        flag = iface.update(1 / freq)
        if flag == -1:
            raise RuntimeError('Square pressed, exiting.')

        des_pose = URPose(*iface.target_pose)
        des_gripper = iface.gripper_state
        if close_gripper:
            obs = env.step(
                des_pose=des_pose,
                des_gripper_state=des_gripper,
                des_zforce=iface.target_zforce,
                adaptive_mode=iface.adaptive_mode,
            )
        if des_gripper == 1:
            break
        time.sleep(1 / 250)

    time.sleep(0.1)
    env.gripper.wait_idle()
    time.sleep(1)


def evaluate(policy, log_dir=None, control_freq=20, device='cuda'):
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
        save_interval=1.0 / 20.,
        control_frequency=control_freq,
        gforce=GRIP_FORCE_N,
        gwidth=GRIP_WIDTH_MM,
        gspeed=GRIP_SPEED_MMPS,
        gpullback=GRIP_PULLBACK_MM,
    )
    env.reset(home_pose)
    env.start()  # start threads

    wait_for_circle(env, iface, close_gripper=False)
    print("Starting evaluation loop...")

    obs_deque = collections.deque([env.get_obs()], maxlen=obs_horizon)  # obs_horizon=1
    save_frames = []
    while True:
        if iface.update(.1) == -1:
            break  # -1 indicates square is pressed and an error is thrown.

        images = np.stack([resize_image(x['image'], (img_size, img_size), flip_channel=True) for x in obs_deque])

        obs_state = np.stack([x['state']['actual_pose'] for x in obs_deque])
        agent_gwidth = np.stack([[x['state']['gripper_width']] for x in obs_deque])
        agent_force = np.stack([x['state']['actual_force'] for x in obs_deque])
        agent_gforce = np.stack([[x['state']['gripper_force']] for x in obs_deque])

        curr_pose, curr_gripper = obs_state[-1], agent_gwidth[-1][0]
        # raw observations: normalization happens inside the policy
        # agent_poses = np.c_[agent_poses, agent_gwidth, agent_force, agent_gforce, target_ix]
        obs_state = np.c_[obs_state, agent_gwidth]

        nimages = rearrange(torch.from_numpy(images).to(device, dtype=torch.float32), 't h w c -> t c h w')
        nobs_state = torch.from_numpy(obs_state).to(device, dtype=torch.float32)  # txd
        with torch.no_grad():
            des_poses, des_widths = get_actions(
                policy, num_diffusion_iters, nimages, nobs_state, curr_pose, curr_gripper)
            start = obs_horizon - 1
            end = start + action_horizon
            des_poses, des_widths = des_poses[start:end], des_widths[start:end]
            print('des grippers:', des_widths)

            for i in tqdm(range(1, len(des_poses)), desc=f'Open-loop execution'):
                env.init_period()
                des_pose = URPose(*des_poses[i])  # already absolute: deltas integrated by the policy
                obs = env.step(
                    des_pose=des_pose,
                    des_gripper_state=des_widths[i],
                )

                obs_deque.append(obs)
                save_frames.append(obs['image'].astype(np.uint8))
                env.wait_period()
            # time.sleep(.5)
            obs_deque.append(env.get_obs())

    env.close()
    _save_video(save_frames, log_dir)


def _save_video(save_frames, log_dir, fps=20):
    if log_dir is None or not save_frames:
        return
    video_path = os.path.join(log_dir, 'evaluation_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w, c = save_frames[0].shape
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    for frame in save_frames:
        out.write(frame)  # frames are already BGR
    out.release()
    print(f"Saved evaluation video → {video_path}")


def evaluate_realtime(policy, log_dir=None, control_freq=20, device='cuda',
                      weight_decay=2.0):
    """
    Real-time action chunking evaluation.

    A background thread runs the diffusion policy as fast as it can: it grabs an
    observation at time ``t``, predicts an action chunk, and pushes it into a
    RealtimeActionChunkingBuffer anchored at ``t``. The main thread runs the
    control loop at ``control_freq`` Hz, asking the buffer for the recency-weighted
    average action at the current time and executing it. The two loops are
    decoupled, so control stays smooth regardless of how long inference takes.
    """
    obs_horizon = policy.obs_horizon
    action_horizon = policy.action_horizon
    num_diffusion_iters = policy.num_diffusion_iters
    img_size = policy.img_size
    control_dt = 1.0 / control_freq

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
        save_interval=1.0 / 20.,
        control_frequency=control_freq,
        gforce=GRIP_FORCE_N,
        gwidth=GRIP_WIDTH_MM,
        gspeed=GRIP_SPEED_MMPS,
        gpullback=GRIP_PULLBACK_MM,
    )
    env.reset(home_pose)
    env.start()  # start threads

    wait_for_circle(env, iface, close_gripper=False)
    print("Starting real-time chunked evaluation loop...")

    buffer = RealtimeActionChunkingBuffer(action_dt=control_dt, weight_decay=weight_decay)
    stop_event = threading.Event()
    save_frames = []

    def prediction_loop():
        """As-fast-as-possible inference; anchors each chunk at its observation time."""
        obs_deque = collections.deque(maxlen=obs_horizon)
        while not stop_event.is_set():
            t_obs = time.time()  # observation time the chunk is anchored to
            obs_deque.append(env.get_obs())
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

            nimages = rearrange(
                torch.from_numpy(images).to(device, dtype=torch.float32), 't h w c -> t c h w')
            nobs_state = torch.from_numpy(obs_state).to(device, dtype=torch.float32)
            with torch.no_grad():
                des_poses, des_grips = get_actions(
                    policy, num_diffusion_iters, nimages, nobs_state, curr_pose, curr_gripper)

            # the executable chunk starts at index obs_horizon-1, which aligns with t_obs
            start = obs_horizon - 1
            end = start + action_horizon
            chnk = buffer.add_chunk(t_obs, des_poses[start:end], des_grips[start:end])
            buffer.dolog(chnk, obs_state, time.time())
        
        import pickle
        pickle.dump(buffer._logs, open('bedug-rca.pkl', 'wb'))
        pickle.dump((env.t0, env.robot_obs), open('bedrug-robs.pkl', 'wb'))
        print('Saved debug thingy')

    pred_thread = threading.Thread(target=prediction_loop, daemon=True)
    pred_thread.start()

    try:
        # wait for the first chunk to land before commanding the robot
        while buffer.is_empty() and not stop_event.is_set():
            if iface.update(control_dt) == -1:
                stop_event.set()
                break
            time.sleep(control_dt)

        start_timing = time.time()
        last_action = None
        action_logs = []
        while not stop_event.is_set():
            env.init_period()
            if iface.update(control_dt) == -1:
                break  # -1 indicates square is pressed.

            pred_freq = buffer._chunk_count / (time.time() - start_timing)
            print(f'INFO: buffer size={len(buffer._chunks)}, nn freq={pred_freq:7.3f}')

            action = buffer.get_action(time.time())
            if action is None:
                action = last_action  # hold last command through a prediction stall
            else:
                action_logs.append((time.time(), action))
                des_pose, des_grip = action
                obs = env.step(
                    des_pose=URPose(*des_pose),  # absolute, recency-weighted average
                    des_gripper_state=int(round(des_grip)),
                )
                save_frames.append(obs['image'].astype(np.uint8))
                last_action = action
            env.wait_period()
        import pickle
        pickle.dump(action_logs, open('bedug-acts.pkl', 'wb'))

    finally:
        stop_event.set()
        pred_thread.join(timeout=5.0)

    env.close()
    _save_video(save_frames, log_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Policy Evaluation.')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='where to save robot log data + evaluation video (None disables logging)')
    parser.add_argument('--mode', type=str, default='openloop', choices=['realtime', 'openloop'],
                        help='realtime: async chunking buffer; openloop: predict-then-execute chunk')
    parser.add_argument('--control_freq', type=int, default=10,
                        help='control/command frequency (Hz) for the real-time loop')
    parser.add_argument('--weight_decay', type=float, default=0.5,
                        help='recency-weighting rate (1/s) for ensembling overlapping chunks')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Architecture config, weights, and normalization stats all come from the checkpoint.
    policy = DiffusionPolicy.from_checkpoint(args.ckpt, args.device)
    policy.eval()

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)

    if args.mode == 'realtime':
        evaluate_realtime(
            policy,
            log_dir=args.log_dir,
            control_freq=args.control_freq,
            weight_decay=args.weight_decay,
            device=args.device,
        )
    else:
        evaluate(policy, log_dir=args.log_dir, control_freq=args.control_freq, device=args.device)
