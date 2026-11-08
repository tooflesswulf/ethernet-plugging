
import torch
from PIL import Image
import torch.nn as nn
from pathlib import Path
import os
import copy
import numpy as np
from tensordict import TensorDict
from diffusers.training_utils import EMAModel
from torchrl.data import  TensorDictPrioritizedReplayBuffer, LazyMemmapStorage
from agent.utils.rb_transforms import MultiStepTransform
from agent.utils.dtype import to_uint8


def save_checkpoint(
    nets: nn.ModuleDict,
    ema: EMAModel,
    save_path: str | Path,
    epoch: int | None = None,
) -> None:
    """
    Save model checkpoint using EMA weights, then restore original weights.

    For mid-training saves (epoch provided): snapshots EMA weights without
    disturbing `nets`, so training can continue unaffected.
    For final save (epoch=None): copies EMA weights directly into nets and saves.

    Args:
        nets:      The live network being trained.
        ema:       The EMAModel tracking a shadow copy of nets.
        save_path: Directory to save checkpoints into.
        epoch:     Current epoch number. If None, treated as the final save.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Architecture config (e.g. DiffusionPolicy.config) so checkpoints are
    # self-describing and loadable via DiffusionPolicy.from_checkpoint.
    config = getattr(nets, 'config', None)

    if epoch is not None:
        # --- Mid-training checkpoint ---
        # Deep-copy nets so we can apply EMA to the copy without touching
        # the original weights that training depends on.
        nets_copy = copy.deepcopy(nets)
        ema.copy_to(nets_copy.parameters())

        filename = save_path / f"ckpt_ep_{epoch}.pth"
        torch.save({"epoch": epoch, "config": config, "model_state_dict": nets_copy.state_dict()}, filename)

    else:
        # --- Final save ---
        # Permanently apply EMA to nets (training is done, no need to restore).
        ema.copy_to(nets.parameters())

        filename = save_path / f"ckpt_final.pth"
        torch.save({"config": config, "model_state_dict": nets.state_dict()}, filename)
        print(f"\n[Checkpoint] Final model saved → {filename}")


def load_checkpoint(
    nets: nn.ModuleDict,
    ckpt_path: str | Path,
    device: str | torch.device,
) -> nn.ModuleDict:
    """
    Load checkpoint weights into nets.

    Args:
        nets:      The network to load weights into.
        ckpt_path: Path to the .pth checkpoint file.
        device:    Device to map the weights to ('cuda', 'cpu', etc.)

    Returns:
        nets with loaded weights, moved to device.
    """
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    nets.load_state_dict(state_dict)
    nets.to(device)

    print(f"[Checkpoint] Loaded from {ckpt_path}")
    return nets


def compute_norm_stats(dataset) -> dict:
    """
    Compute min/max normalization statistics from a StitchedSequenceDataset.

    Stats are computed over the precomputed action chunks (so they reflect
    the dataset's action_mode: absolute/local_delta/global_delta) and over
    the stitched observation array.

    Args:
        dataset: A StitchedSequenceDataset with .actions (num_samples, horizon,
                 action_dim) and .obs (N, obs_dim) populated.

    Returns:
        {'actions': {'min': (action_dim,), 'max': (action_dim,)},
         'states':  {'min': (obs_dim,),    'max': (obs_dim,)}}
    """
    actions = np.asarray(dataset.actions)
    flat_actions = actions.reshape(-1, actions.shape[-1])
    obs = dataset.obs
    return {
        'actions': {'min': flat_actions.min(0), 'max': flat_actions.max(0)},
        'states': {'min': obs.min(0), 'max': obs.max(0)},
    }


def resize_image(np_array, new_size=(128, 128), flip_channel=False):
    if flip_channel:
        np_array = np_array[:, :, ::-1]
    img = Image.fromarray(np_array)
    img = img.resize(new_size, )
    return np.array(img)


def init_buffer(total_transitions, offline_scratch_dir, online_scratch_dir,  cfg ):

    alpha = cfg.algo.priority_alpha if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0
    beta = cfg.algo.priority_beta if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0

    online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
    offline_batch_size = int(cfg.algo.batch_size * cfg.algo.offline_fraction)

    storage = LazyMemmapStorage(total_transitions, scratch_dir=offline_scratch_dir)
    offline_rb = TensorDictPrioritizedReplayBuffer(
        # storage=LazyTensorStorage(max_size=max_offline_transitions, device="cpu"),  # keep in RAM
        storage=storage,
        alpha=alpha,
        beta=beta,
        eps=1e-6,  # Small epsilon added to priorities to prevent zero values
        priority_key="_priority",
        transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,  # Add prefetching
        batch_size=max(offline_batch_size, 1),  # Ensure batch_size is at least 1
    )
    storage = LazyMemmapStorage(cfg.algo.buffer_size, scratch_dir=online_scratch_dir)
    online_rb = TensorDictPrioritizedReplayBuffer(
        storage=storage,  # on disk storage
        alpha=alpha,
        beta=beta,
        eps=1e-6,  # Small epsilon added to priorities to prevent zero values
        priority_key="_priority",
        transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,  # Add prefetching
        batch_size=online_batch_size,
    )
    return offline_rb, online_rb

def zforce_reward(zforce, target_force=5, min_force=0, min_reward=-1):
    d = (target_force - min_force)**2
    reward = - ( (zforce - target_force)**2 ) / d
    return np.clip(reward, min_reward, 0)

def _add_transitions_to_buffer(
    obs: dict,
    next_obs: dict,
    actions: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    online_rb: TensorDictPrioritizedReplayBuffer,
) -> None:
    """Helper function to create transitions and add them to the replay buffer.

    Handles terminal observations correctly and convert images to uint8 for storage.
    """
    obs_keys_set = ['observation.base_action',
                    #  'observation.rgb', 
                     'observation.state']

    # Keep only relevant keys & convert images to uint8 for storage
    curr_obs_i = {k: v for k, v in obs.items() if k in obs_keys_set}
    next_obs_i = {k: v for k, v in next_obs.items() if k in obs_keys_set}
    to_uint8(curr_obs_i, ['observation.rgb'])
    to_uint8(next_obs_i, ['observation.rgb'])
    for k in obs_keys_set:
        curr_v, next_v = curr_obs_i[k], next_obs_i[k]

        if not isinstance(curr_v, torch.Tensor):
            curr_obs_i[k] = torch.tensor(curr_v)
        if not isinstance(next_v, torch.Tensor):
            next_obs_i[k] = torch.tensor(next_v)

    td = TensorDict(
        {
            "obs": TensorDict(curr_obs_i, batch_size=[]),
            "next": TensorDict(
                {
                    "obs": TensorDict(next_obs_i, batch_size=[]),
                    "done": torch.tensor(done, dtype=torch.bool),
                    "reward": torch.tensor(reward, dtype=torch.float32),
                },
                batch_size=[],
            ),
            "action": actions,
            "_priority": torch.tensor(10.0, dtype=torch.float32),  # High initial priority for new samples
        },
        batch_size=[],
    ).unsqueeze(0)

    online_rb.add(td)


def add_episode_to_buffer(
        ep,
        obs_fields,
        is_success,
        rb
):
    """Helper function to create transitions and add them to the replay buffer.

    Handles terminal observations correctly and convert images to uint8 for storage.
    """
    obs_keys_set = ['observation.base_action', 'observation.rgb', 'observation.state']
    images, states, base_poses, des_poses = ep['image'], ep['state'], ep['base_pose'], ep['dpose_req']
    
    actions = base_poses + des_poses
    zforces = states['filtered_force'][:,2]; rewards = zforce_reward(zforces); dones = [False] * len(images)
    states = np.concatenate([states[k] for k in obs_fields], -1); 
    T = len(states)
    obs_list, reward_list, action_list, done_list = [], [], [], []
    for t in range(0, T):
        obs_list.append( {
            'observation.base_action': torch.from_numpy(base_poses[t]).float(),
            # 'observation.rgb': torch.from_numpy( images[t]).float(),
            'observation.state': torch.from_numpy(states[t]).float(),
        }); reward_list.append(rewards[t]); action_list.append(actions[t])

    # range: 0:T-1
    next_obs_list = obs_list[1:]
    obs_list = obs_list[:-1]; action_list = action_list[:-1]
    # reward/done belongs to next step: 1:T
    reward_list = reward_list[1:] #
    rewards[-1] = float(is_success)
    done_list =  dones[1:]; done_list[-1] = True

    for (obs, next_obs, action, reward, done) in zip( obs_list, next_obs_list, action_list, reward_list, done_list ):
        _add_transitions_to_buffer(obs, next_obs, action, reward, done, rb)
