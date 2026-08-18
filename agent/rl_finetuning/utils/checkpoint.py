# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Utilities for loading and saving model checkpoints."""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import wandb

from agent.rl_finetuning.off_policy.rl.q_agent import QAgent


def _download_from_wandb(checkpoint_spec: str) -> tuple[Path, dict]:
    """Download checkpoint from W&B.

    Args:
        checkpoint_spec: W&B specification in format:
            - "entity/project/runs/run_id/files/path/to/checkpoint.pt"
            - "run_id" (uses current project/entity from wandb.run)

    Returns:
        (checkpoint_path, wandb_config): Path to downloaded file and W&B run config

    Raises:
        ImportError: If wandb is not available
        ValueError: If checkpoint_spec format is invalid
    """
    # Parse the checkpoint specification
    # Extract entity/project/runs/run_id from the full path
    parts = checkpoint_spec.split("/")

    entity = parts[0]
    project = parts[1]
    run_id = parts[3]

    # Extract the file path within the run (everything after "files/")
    files_idx = parts.index("files")
    file_path = "/".join(parts[files_idx + 1:])

    # Create API instance
    api = wandb.Api()

    # Construct run path
    run_path = f"{entity}/{project}/{run_id}"

    print(f"Downloading checkpoint from W&B run: {run_path}")
    print(f"File path: {file_path}")

    run = api.run(run_path)

    # Get the W&B config
    wandb_config = dict(run.config)

    # Create temporary directory for download
    temp_dir = Path(tempfile.mkdtemp(prefix="wandb_checkpoint_"))

    # Download the file
    downloaded_file = run.file(file_path).download(root=str(temp_dir), replace=True)
    checkpoint_path = Path(downloaded_file.name)

    print(f"✅ Downloaded checkpoint to: {checkpoint_path}")
    return checkpoint_path, wandb_config


def save_checkpoint(
    agent: QAgent,
    checkpoint_path: str | Path,
    global_step: int,
    config: Any = None,
    success_rate: float | None = None,
    **extra_data: Any,
) -> None:
    """Save a QAgent checkpoint.

    Args:
        agent: The QAgent to save
        checkpoint_path: Path where to save the checkpoint
        global_step: Current training step
        config: Training configuration (optional)
        success_rate: Success rate when checkpoint was saved (optional)
        **extra_data: Additional data to include in checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "agent_state_dict": agent.state_dict(),
        "global_step": global_step,
        **extra_data,
    }

    # Save optimizer and scheduler states
    optimizer_state_dict = {
        "actor_opt": agent.actor_opt.state_dict(),
        "critic_opt": agent.critic_opt.state_dict(),
        "encoder_opt": agent.encoder_opt.state_dict(),
    }
    scheduler_state_dict = {}
    if hasattr(agent, "actor_scheduler") and agent.actor_scheduler is not None:
        scheduler_state_dict["actor_scheduler"] = agent.actor_scheduler.state_dict()
    if hasattr(agent, "critic_scheduler") and agent.critic_scheduler is not None:
        scheduler_state_dict["critic_scheduler"] = agent.critic_scheduler.state_dict()
    if hasattr(agent, "encoder_scheduler") and agent.encoder_scheduler is not None:
        scheduler_state_dict["encoder_scheduler"] = agent.encoder_scheduler.state_dict()

    checkpoint_data["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict:  # Only add if there are schedulers
        checkpoint_data["scheduler_state_dict"] = scheduler_state_dict

    if config is not None:
        # Convert OmegaConf to plain dict to avoid PyTorch 2.6+ loading issues
        try:
            from omegaconf import OmegaConf

            if hasattr(config, "_metadata"):  # Check if it's an OmegaConf object
                checkpoint_data["config"] = OmegaConf.to_container(config, resolve=True)
            else:
                checkpoint_data["config"] = config
        except ImportError:
            checkpoint_data["config"] = config
    if success_rate is not None:
        checkpoint_data["success_rate"] = success_rate

    torch.save(checkpoint_data, checkpoint_path)
    print(f"💾 Saved checkpoint to: {checkpoint_path}")


# Normalization buffers were added to QAgent after the first checkpoints were written;
# a checkpoint without them loads fine and leaves the agent's identity-transform defaults.
_NORM_STAT_KEYS = {"action_min", "action_max", "state_mean", "state_std", "_norm_stats_set"}


def load_checkpoint(
    agent: QAgent,
    checkpoint_path: str | Path,
    load_optimizers: bool = True,
) -> dict:
    """Load a QAgent checkpoint in place.

    Restores weights *and* the dataset normalization stats (they are registered
    buffers, so they travel inside ``agent_state_dict``).

    Args:
        agent: The QAgent to load into; must be built with the same architecture.
        checkpoint_path: Path to a file written by :func:`save_checkpoint`.
        load_optimizers: Also restore optimizer / scheduler state.

    Returns:
        The checkpoint dict, minus the state dicts already applied (so callers can
        read ``global_step``, ``config``, ``success_rate``, ...).
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_data = torch.load(checkpoint_path, map_location=agent.cfg.device, weights_only=False)

    missing, unexpected = agent.load_state_dict(checkpoint_data["agent_state_dict"], strict=False)
    unknown_missing = [k for k in missing if k not in _NORM_STAT_KEYS]
    if unknown_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not match this agent: "
            f"missing={unknown_missing}, unexpected={list(unexpected)}")
    if missing:
        print(f"⚠️  Checkpoint predates normalization stats ({sorted(missing)}); "
              "agent keeps its identity-transform defaults.")
    elif not agent.norm_stats_set:
        print("⚠️  Checkpoint carries identity-transform normalization stats "
              "(set_norm_stats was never called before saving).")

    if load_optimizers and "optimizer_state_dict" in checkpoint_data:
        optimizer_state = checkpoint_data["optimizer_state_dict"]
        agent.actor_opt.load_state_dict(optimizer_state["actor_opt"])
        agent.critic_opt.load_state_dict(optimizer_state["critic_opt"])
        agent.encoder_opt.load_state_dict(optimizer_state["encoder_opt"])

        for name, scheduler_state in checkpoint_data.get("scheduler_state_dict", {}).items():
            scheduler = getattr(agent, name, None)
            if scheduler is not None:
                scheduler.load_state_dict(scheduler_state)

    print(f"📂 Loaded checkpoint from: {checkpoint_path}")
    return {k: v for k, v in checkpoint_data.items()
            if k not in ("agent_state_dict", "optimizer_state_dict", "scheduler_state_dict")}


# -----------------------------------------------------------------------------
# Replay-buffer checkpoints ----------------------------------------------------
# -----------------------------------------------------------------------------
# A dumped buffer is only meaningful next to the base policy it was generated with:
# the offline buffer stores base-policy actions, and the warm-up buffer stores
# transitions the base policy actually drove. Reusing one across policies silently
# trains on the wrong data, so the buffers are filed under a fingerprint of what
# produced them -- a different base policy simply resolves to a different directory
# and the caller regenerates instead of loading something incompatible. The sidecar
# metadata records the same fingerprint in readable form and is re-checked on load,
# which catches directories that were moved or hand-edited.

_BUFFER_NAMES = ('offline_rb', 'warmup_rb')


def _file_sha256(path: str | Path) -> str:
    """Content hash of a file, read in chunks so large checkpoints stay off the heap."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def buffer_fingerprint(base_policy_ckpt: str | Path, **params: Any) -> dict:
    """Describe what a replay buffer was generated from.

    Args:
        base_policy_ckpt: Path to the base-policy checkpoint used to fill the buffer.
            Hashed by content, not by path -- checkpoints get overwritten in place, and
            a stale path match is exactly the failure this is meant to prevent.
        **params: Any other generation inputs that make buffers incompatible
            (dataset name, control frequency, ...). They become part of the identity,
            so changing one routes to a fresh buffer rather than reusing a stale one.

    Returns:
        A dict describing the buffer's provenance; pass it to
        :func:`save_replay_buffers` / :func:`load_replay_buffers`.
    """
    base_policy_ckpt = Path(base_policy_ckpt)
    return {
        "base_policy_ckpt": str(base_policy_ckpt.resolve()),
        "base_policy_sha256": _file_sha256(base_policy_ckpt),
        "params": {k: params[k] for k in sorted(params)},
    }


def _fingerprint_id(fingerprint: dict) -> str:
    """Short, stable directory name for a fingerprint.

    Deliberately ignores ``base_policy_ckpt``: the same weights moved to a new path are
    the same policy, and keying on the path would rebuild the buffer for no reason.
    """
    payload = json.dumps(
        {"sha256": fingerprint["base_policy_sha256"], "params": fingerprint["params"]},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _buffer_paths(checkpoint_dir: str | Path, name: str, fingerprint: dict) -> tuple[Path, Path]:
    assert name in _BUFFER_NAMES, f"Unknown buffer {name!r}, expected one of {_BUFFER_NAMES}"
    root = Path(checkpoint_dir) / _fingerprint_id(fingerprint)
    return root / name, root / f"{name}.meta.json"


def save_replay_buffers(
    rb,
    name: str,
    checkpoint_dir: str | Path,
    fingerprint: dict,
):
    """Dump ``rb`` under the directory owned by ``fingerprint``, with its provenance."""
    buffer_path, meta_path = _buffer_paths(checkpoint_dir, name, fingerprint)
    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    rb.dumps(buffer_path)
    meta_path.write_text(json.dumps(
        {
            "buffer": name,
            "num_transitions": len(rb),
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            **fingerprint,
        },
        indent=2,
        sort_keys=True,
        default=str,
    ))
    print(f"💾 Saved {name} ({len(rb)} transitions) to: {buffer_path}")


def load_replay_buffers(
    rb,
    name: str,
    checkpoint_dir: str | Path,
    fingerprint: dict,
):
    """Load a dumped buffer, but only the one generated from ``fingerprint``.

    Returns:
        ``(rb, True)`` if a matching dump was loaded, ``(rb, False)`` if none exists for
        this fingerprint (the caller should regenerate and save).

    Raises:
        RuntimeError: If a dump exists but its recorded provenance disagrees with
            ``fingerprint``, or if the dump is present but unreadable. Both mean the
            on-disk state is not what it claims to be, and silently regenerating would
            overwrite it.
    """
    buffer_path, meta_path = _buffer_paths(checkpoint_dir, name, fingerprint)
    if not buffer_path.exists():
        return rb, False

    if not meta_path.exists():
        raise RuntimeError(
            f"Replay buffer {buffer_path} has no {meta_path.name} recording which base "
            "policy generated it, so it cannot be verified. Delete it to regenerate.")

    meta = json.loads(meta_path.read_text())
    mismatched = {
        key: (meta.get(key), fingerprint[key])
        for key in ("base_policy_sha256", "params")
        if meta.get(key) != fingerprint[key]
    }
    if meta.get("buffer") != name:
        mismatched["buffer"] = (meta.get("buffer"), name)
    if mismatched:
        details = "\n".join(f"  {k}: on disk {found!r} != requested {want!r}"
                             for k, (found, want) in mismatched.items())
        raise RuntimeError(
            f"Replay buffer {buffer_path} was generated from a different setup:\n{details}\n"
            f"(on-disk buffer came from {meta.get('base_policy_ckpt')}). "
            "Delete that directory to regenerate.")

    rb.loads(buffer_path)
    print(f"📂 Loaded {name} ({len(rb)} transitions) from: {buffer_path}")
    return rb, True
