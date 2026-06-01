
import torch
import torch.nn as nn
from pathlib import Path
import os, copy, numpy as np
from diffusers.training_utils import EMAModel
from agent.dataset.sequence import pose2actions

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

    if epoch is not None:
        # --- Mid-training checkpoint ---
        # Deep-copy nets so we can apply EMA to the copy without touching
        # the original weights that training depends on.
        nets_copy = copy.deepcopy(nets)
        ema.copy_to(nets_copy.parameters())

        filename = save_path / f"ckpt_ep_{epoch}.pth"
        torch.save({"epoch": epoch, "model_state_dict": nets_copy.state_dict()}, filename)
        print(f"[Checkpoint] Epoch {epoch} saved → {filename}")

    else:
        # --- Final save ---
        # Permanently apply EMA to nets (training is done, no need to restore).
        ema.copy_to(nets.parameters())

        filename = save_path / f"ckpt_final.pth"
        torch.save({"model_state_dict": nets.state_dict()}, filename)
        print(f"[Checkpoint] Final model saved → {filename}")
    
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

def get_stats(dataset_path, max_n_episodes=10000 ):
    
    state_path = os.path.join(dataset_path, 'states.npz')
    dataset = np.load(state_path, allow_pickle=False)  # only np arrays
    traj_lengths = dataset["traj_length"][:max_n_episodes]  # 1-D array
    actions = pose2actions(dataset['actual_pose'],  dataset['gripper_width'], traj_lengths)
    states = np.concatenate([dataset['actual_pose'], dataset['gripper_width'][:, None] ], axis = -1)

    return {
        'actions': {'min': actions.min(0), 'max': actions.max(0)},
        'states':  {'min': states.min(0), 'max': states.max(0)},
    }

def normalize(arr: np.ndarray, stats: dict) -> np.ndarray:
    """
    Normalize a numpy array to [-1, 1] using precomputed min/max stats.
    Dimensions where max == min are left unchanged.

    Args:
        arr:   Input array to normalize.
        stats: Dict with keys 'min' and 'max' (scalars or arrays matching arr).

    Returns:
        Normalized array with same shape as input.
    """
    min_val = np.array(stats['min'])
    max_val = np.array(stats['max'])

    range_val = max_val - min_val
    safe_range = np.where(range_val == 0, 1, range_val)  # avoid division by zero

    normalized = 2 * (arr - min_val) / safe_range - 1

    return np.where(range_val == 0, arr, normalized)

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

    denormalized = (arr + 1) / 2 * range_val + min_val

    return np.where(range_val == 0, arr, denormalized)
