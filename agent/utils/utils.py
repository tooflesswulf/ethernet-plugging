import copy
import torch
import torch.nn as nn
from diffusers.training_utils import EMAModel
from pathlib import Path


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