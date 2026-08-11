
import torch
from PIL import Image
import torch.nn as nn
from pathlib import Path
import os
import copy
import numpy as np
from diffusers.training_utils import EMAModel
from scipy.spatial.transform import Rotation as R, RigidTransform as Tf

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


def normalize(arr: np.ndarray, stats: dict) -> np.ndarray:
    """
    Normalize a numpy array to [-1, 1] using precomputed min/max stats.
    Dimensions where max == min are left unchanged.

    Args:
        arr:   Array to normalize.
        stats: Dict with keys 'min' and 'max' (scalars or arrays matching arr).

    Returns:
        Normalized array with same shape as input.
    """
    min_val = np.array(stats['min'])
    max_val = np.array(stats['max'])

    range_val = max_val - min_val
    safe_range = np.where(range_val == 0, 1.0, range_val)

    return np.where(range_val == 0, arr, 2 * (arr - min_val) / safe_range - 1)


def denormalize(arr: np.ndarray, stats: dict) -> np.ndarray:
    """
    Denormalize a numpy array from [-1, 1] back to original scale using
    precomputed min/max stats. Dimensions where max == min are left unchanged.

    Args:
        arr:   Normalized array to denormalize.
        stats: Dict with keys 'min' and 'max' (scalars or arrays matching arr).

    Returns:
        Denormalized array with same shape as input.
    """
    min_val = np.array(stats['min'])
    max_val = np.array(stats['max'])

    range_val = max_val - min_val

    denormalized = ((arr + 1) / 2) * range_val + min_val

    return denormalized


def resize_image(np_array, new_size=(128, 128), flip_channel=False):
    if flip_channel:
        np_array = np_array[:, :, ::-1]
    img = Image.fromarray(np_array)
    img = img.resize(new_size, )
    return np.array(img)

def gripper_action(g_widths, threshold=20):
    """
    Binary gripper predictions. 1=open, -1=closed
    """
    return 2 * (g_widths > threshold).astype(int).reshape(-1, 1) - 1

def _pose_action_local_delta( poses):
    # Returns (N, 6): [rx, ry, rz, tx, ty, tz] (SE(3) exp coords, NOT the same ordering as absolute)
    transforms = [Tf.from_components(pos[:3], R.from_rotvec(pos[3:])) for pos in poses]
    t0 = transforms[0]
    deltas = [t0.inv() * t for t in transforms]
    return np.array([delta.as_exp_coords() for delta in deltas])

def get_pose_action(poses, action_mode):
    
    if action_mode == 'local_delta':
        return _pose_action_local_delta(poses)
    
    else:
        raise ValueError(f"Invalid action_mode: {action_mode}")

def get_dataset(dataset_path, lowdim_keys, action_mode, num_episodes, g_thr=18):
    # return individual episode path, and total count of transitions
    episode_names = sorted( os.listdir(dataset_path), key = lambda x: int(x.replace('episode', '')) )[:num_episodes]
    total_transitions = 0
    episode_paths, states, actions = [], [], []
    for episode_name in episode_names:
        episode_path = os.path.join(dataset_path, episode_name)
        episode_paths.append(episode_path)

        loaded = np.load(os.path.join( episode_path, 'states.npz'))
        state =  np.concatenate( [ loaded[k] if loaded[k].ndim == 2 else loaded[k][:, None] for k in lowdim_keys    ] , -1)
        pose_action = get_pose_action(loaded['pose'], action_mode); g_action = gripper_action(loaded['gripper_width'], threshold=g_thr)
        action = np.concatenate([pose_action, g_action ], -1)
        states.append(state); actions.append(action); total_transitions += len(action)
    print('Loaded episode count:', len(episode_paths), '\tTotal transitions:', total_transitions)
    return episode_paths, states, actions, total_transitions
    