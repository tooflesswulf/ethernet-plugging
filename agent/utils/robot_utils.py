import time

import einops
import numpy as np
import torch

from env import URPose, Env, GRIP_CLOSED
from interface import DualSenseInterface
from agent.model.policy import DiffusionPolicy
from agent.utils.utils import resize_image
from agent.utils.interrupt_sequence import InterruptSequence


# Maps a dataset.h5 observation field (the names used in policy.obs_fields, see
# scripts/rawdata_to_dataset.py) to how it's pulled from a live env obs dict.
# Each getter returns a 1-D array for one timestep; concatenating them in
# obs_fields order reproduces the state vector StitchedSequenceDataset trained on.
# NOTE: 'force' is the smoothed force — dataset uses an EWMA of actual_force, and
# filtered_force is the env's smoothed counterpart.
OBS_FIELD_GETTERS = {
    'pose': lambda s: np.asarray(s['actual_pose'], dtype=np.float32),
    'force': lambda s: np.asarray(s['filtered_force'], dtype=np.float32),
    'gripper_width': lambda s: np.asarray([s['gripper_width']], dtype=np.float32),
    'gripper_force': lambda s: np.asarray([s['gripper_force']], dtype=np.float32),
}


def build_states(obs_deque, obs_fields):
    states = []
    for obs in obs_deque:
        s = obs['state']
        try:
            states.append(np.concatenate([OBS_FIELD_GETTERS[f](s) for f in obs_fields]))
        except KeyError as e:
            raise KeyError(
                f"obs_field {e} has no live-eval mapping in OBS_FIELD_GETTERS "
                f"(policy.obs_fields={obs_fields}). Add a getter or retrain without it.")
    return np.stack(states)  # (T, state_dim)


def get_actions(policy: DiffusionPolicy, obs_deque, device='cuda'):
    """
    obs_deque: sequence of env obs dicts (len == policy.obs_horizon), each with
               'image' and 'state'. Images and the proprio state vector are built
               here; the state follows policy.obs_fields so it matches training.
    Returns (des_poses (H, 6) absolute [trans, rotvec], des_widths (H,),
    des_done (H,) end-of-episode score in [0, 1]) ready to execute.
    """
    img_size = policy.img_size
    images = np.stack([resize_image(o['image'], (img_size, img_size), flip_channel=True) for o in obs_deque])
    states = build_states(obs_deque, policy.obs_fields)  # (T, state_dim)

    # current absolute pose/width to integrate the (possibly delta) actions from
    last = obs_deque[-1]['state']
    curr_pose, curr_gripper_width = np.asarray(last['actual_pose']), last['gripper_width']

    nimages = einops.rearrange(
        torch.from_numpy(images).to(device, dtype=torch.float32), 't h w c -> t c h w')
    nstates = torch.from_numpy(states).to(device, dtype=torch.float32)

    conditions = {
        'rgb': (nimages / 255.0).unsqueeze(0),  # (1, T, C, H, W)
        'state': nstates.unsqueeze(0),          # (1, T, state_dim); policy normalizes internally
    }
    naction = policy.predict_action(conditions)
    naction = naction.detach().to('cpu').numpy()[0]

    # integrate deltas (per the policy's action_mode) into absolute poses + widths
    return policy.integrate_actions(naction, curr_pose, curr_gripper_width)


def interrupt(rexec):
    """The active InterruptSequence on `rexec`, installing one if needed."""
    return InterruptSequence.current(rexec)


def wait_for_circle(env: Env, iface: DualSenseInterface, close_gripper=False):
    freq = 250
    print('Waiting the circle...')
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
        if des_gripper == GRIP_CLOSED:
            break
        time.sleep(1 / 250)

    time.sleep(0.1)
    env.gripper.wait_idle()
    time.sleep(1)
