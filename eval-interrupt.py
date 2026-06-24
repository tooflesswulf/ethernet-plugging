import interface
import numpy as np
from einops import rearrange
import time
import threading
import collections
import argparse
import os
import cv2
import torch

from env import Env, URPose
from agent.model.policy import DiffusionPolicy
from agent.utils.utils import resize_image
from agent.eval.realtime_chunking import RealtimeActionChunkingBuffer
from agent.utils.robot_utils import get_actions, wait_for_circle

GRIP_WIDTH_MM = 10  # 8
GRIP_FORCE_N = 40
GRIP_SPEED_MMPS = 50
GRIP_PULLBACK_MM = 5


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
                des_poses, des_grips = get_actions(policy, nimages, nobs_state, curr_pose, curr_gripper)

            # the executable chunk starts at index obs_horizon-1, which aligns with t_obs
            start = obs_horizon - 1
            end = start + action_horizon
            chnk = buffer.add_chunk(t_obs, des_poses[start:end], des_grips[start:end])
            buffer.dolog(chnk, obs_state, time.time())

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
        undo_action_buffer = []

        while not stop_event.is_set():
            env.init_period()
            if iface.update(control_dt) == -1:
                break  # -1 indicates square is pressed.

            pred_freq = buffer._chunk_count / (time.time() - start_timing)
            print(f'INFO: buffer size={len(buffer._chunks)}, nn freq={pred_freq:7.3f}', end='\r')
            if iface.dualsense.state.DpadLeft and len(undo_action_buffer) == 0:
                print('\nINTERRUPTED!')

                # Got interrupt signal; populate undo_action_buffer with some sequence
                undo_action_buffer = [act for t, act in action_logs[-60:]]
                action_logs = action_logs[:-60]
                print(f'action log size', len(action_logs))

                # Add some xy noise to undo buffer? 1cm stdev
                vec = .001 * np.random.normal(size=2)
                print(f'Adding {vec}')
                vec = np.r_[vec, 0, 0, 0, 0]
                tt = np.linspace(0, 1, len(undo_action_buffer))[::-1]
                undo_action_buffer = [(act + t * vec, grip) for t, (act, grip) in zip(tt, undo_action_buffer)]

            if len(undo_action_buffer) > 0:
                action = undo_action_buffer.pop()
                if len(undo_action_buffer) == 0:
                    # Finished undoing actions. Empty pred buffer.
                    buffer.clear()
            else:
                action = buffer.get_action(time.time())
                if action is None:
                    action = last_action
                action_logs.append((time.time(), action))

            des_pose, des_grip = action
            obs = env.step(
                des_pose=URPose(*des_pose),  # absolute, recency-weighted average
                des_gripper_state=int(round(des_grip)),
            )
            save_frames.append(obs['image'].astype(np.uint8))
            last_action = action

            cv2.imshow('RGB', obs['image'])
            cv2.waitKey(1)
            env.wait_period()

    finally:
        stop_event.set()
        pred_thread.join(timeout=5.0)

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Policy Evaluation.')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='where to save robot log data + evaluation video (None disables logging)')
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

    evaluate_realtime(
        policy,
        log_dir=args.log_dir,
        control_freq=args.control_freq,
        weight_decay=args.weight_decay,
        device=args.device,
    )
