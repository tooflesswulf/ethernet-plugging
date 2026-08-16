"""Offline dataset (``dataset.h5``) → residual-RL replay-buffer transitions.

The dataset layout is the one written by ``scripts/rawdata_to_dataset.py``:
flat, episode-stitched arrays (``pose``, ``force``, ``gripper_width``, ...) plus a
``metadata/length`` array giving the per-episode frame counts.  Images are either
stored inline (``images.attrs['stored_as'] == 'image'``) or as paths relative to
the dataset directory (``'filepath'``).

Two stages, mirroring how training uses them:
  1. :func:`parse_offline_dataset` — read states/actions/rewards for every episode
     (cheap, no images).  Its transition count is needed to size the buffer.
  2. :func:`populate_offline_buffer` — stream the images back in and push the
     transitions into the replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from einops import rearrange
from tensordict import TensorDict
from torchrl.data import ReplayBuffer
from tqdm import tqdm
from PIL import Image
import numpy as np
import pathlib
import torch
import h5py

from agent.utils.utils import gripper_action, find_chunks
from env import GRIP_OPEN, GRIP_CLOSED
from agent.rl_finetuning.utils.dtype import to_uint8
from agent.model.policy import DiffusionPolicy

@dataclass
class OfflineEpisode:
    """One parsed episode; `states`/`commands`/`rewards` are aligned per frame."""

    ep_idx: int  # index of the episode in the dataset
    start: int  # offset of the episode's first frame in the stitched arrays
    length: int
    states: np.ndarray  # (T, state_dim)
    # The reference command the demonstration recorded, in the same space the base
    # policy's integrated commands live in: absolute pose [tx, ty, tz, rx, ry, rz]
    # plus gripper state (GRIP_OPEN=0 / GRIP_CLOSED=1).
    commands: np.ndarray  # (T, 7)
    rewards: np.ndarray  # (T,)
    gripper_widths: np.ndarray  # (T,) raw widths, used to integrate the base policy chunk


def resolve_dataset_path(dataset_path: str | pathlib.Path) -> pathlib.Path:
    """Accept either a dataset directory or the ``dataset.h5`` file itself."""
    path = pathlib.Path(dataset_path)
    if path.is_dir():
        path = path / 'dataset.h5'
    if not path.is_file():
        raise FileNotFoundError(f"No dataset.h5 found at {dataset_path}")
    return path


def _episode_bounds(f: h5py.File, num_episodes: int | None) -> list[tuple[int, int]]:
    """Return (start, length) for each of the first *num_episodes* episodes."""
    lengths = np.asarray(f['metadata/length'][:]).astype(int).ravel()
    if num_episodes is not None:
        lengths = lengths[:num_episodes]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]) if len(lengths) else np.array([], dtype=int)
    return [(int(s), int(n)) for s, n in zip(starts, lengths)]


def parse_offline_dataset(
    dataset_path: str | pathlib.Path,
    lowdim_keys: list[str],
    num_episodes: int | None = None,
    g_thr: float = 18,
) -> tuple[list[OfflineEpisode], int]:
    """Read the per-episode states, reference commands and rewards out of ``dataset.h5``.

    The recorded command is absolute (pose + gripper state) rather than one of the
    delta encodings the policy is *trained* on: a delta is only meaningful next to the
    pose it is anchored to, and the base policy's predictions are integrated back into
    absolute commands before they enter the buffer (see :func:`populate_offline_buffer`),
    so both sides of the residual live in the same space.

    Rewards are derived from the gripper open/close phases: the episode must split
    into the 5 expected segments (approach, pick-and-transport-plugin,
    release-to-unplug, unplug-place, release); episodes that do not are skipped.
    Reward is 1 from shortly before the plug-in until just after the release, and
    2 afterwards.

    Returns the parsed episodes and the total number of frames they cover.
    """
    h5_path = resolve_dataset_path(dataset_path)

    episodes: list[OfflineEpisode] = []
    total_transitions = 0
    with h5py.File(h5_path, 'r') as f:
        for ep_idx, (start, length) in enumerate(_episode_bounds(f, num_episodes)):
            sl = slice(start, start + length)
            fields = []
            for key in lowdim_keys:
                if key not in f:
                    raise KeyError(f"obs field {key!r} is not in {h5_path} (has: {list(f.keys())})")
                value = np.asarray(f[key][sl])
                fields.append(value if value.ndim == 2 else value[:, None])
            state = np.concatenate(fields, -1)

            poses = np.asarray(f['pose'][sl])
            gripper_widths = np.asarray(f['gripper_width'][sl])
            g_action = gripper_action(gripper_widths, threshold=g_thr)

            chunks = find_chunks(g_action.flatten())
            if not (len(chunks) == 5 and chunks[2][-1] == 1):
                print(f"Skip episode {ep_idx}: expected 5 gripper segments, got {len(chunks)}")
                continue

            phase1 = chunks[1][1] - 30
            phase2 = chunks[2][1] + 10
            reward = np.zeros(length)
            reward[phase1:phase2] = 1.0
            reward[phase2:] = 2.0

            # Same encoding integrate_actions produces: +1/-1 open/closed -> 0/1 state
            g_state = np.where(g_action > 0, GRIP_OPEN, GRIP_CLOSED)

            episodes.append(
                OfflineEpisode(
                    ep_idx=ep_idx,
                    start=start,
                    length=length,
                    states=state,
                    commands=np.concatenate([poses, g_state], -1),
                    rewards=reward,
                    gripper_widths=gripper_widths,
                )
            )
            total_transitions += length

    print('Loaded episode count:', len(episodes), '\tTotal transitions:', total_transitions)
    return episodes, total_transitions


def _read_episode_images(
    f: h5py.File,
    dataset_dir: pathlib.Path,
    episode: OfflineEpisode,
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    """Load and resize one episode's images → float tensor (T, H, W, C).

    Channel order is RGB in both storage modes, so no flip is needed here: the camera
    hands out BGR (``camera.py``, saved to rawdata as ``camera_obs/image_bgr``) but
    ``scripts/rawdata_to_dataset.py`` flips it to RGB before writing ``images`` /
    the PNGs.  The live env is the side that has to flip (``resize_image(...,
    flip_channel=True)`` in ``agent/utils/robot_utils.py``) to match this.
    """
    sl = slice(episode.start, episode.start + episode.length)
    images = f['images']

    if images.attrs.get('stored_as') == 'filepath' or h5py.check_string_dtype(images.dtype):
        frames = []
        for raw_path in images[sl]:
            path = pathlib.Path(raw_path.decode() if isinstance(raw_path, bytes) else raw_path)
            if not path.is_absolute():
                path = dataset_dir / path
            with Image.open(path) as img:
                frames.append(np.asarray(img.convert('RGB').resize((img_w, img_h))))
    else:
        frames = [np.asarray(Image.fromarray(img).resize((img_w, img_h))) for img in images[sl]]

    return torch.from_numpy(np.stack(frames)).float()


def populate_offline_buffer(
    dataset_path: str | pathlib.Path,
    episodes: list[OfflineEpisode],
    rb: ReplayBuffer,
    policy_base_actions: bool = True,
    base_policy: DiffusionPolicy | None = None,
    img_h: int = 128,
    img_w: int = 128,
    device: str | torch.device = 'cuda',
) -> int:
    """Turn consecutive frames of each episode into residual RL transitions.

    The base policy predicts *deltas* (per its action_mode), which only mean something
    next to the pose they are anchored to, so each predicted chunk is integrated against
    the frame's recorded pose (``policy.integrate_actions``) into absolute commands --
    the same thing ``get_base_action`` hands the online path. The transition then stores
    the recorded command as the action, so the residual the QAgent has to learn is
    exactly ``recorded_command - base_policy_command``.

    Two modes:
    1. GT-as-base (policy_base_actions=False):
        Uses the recorded command as both the base action (in observations) and the
        target action (in transitions), i.e. the residual is 0 everywhere.

    2. Base-policy-as-base (policy_base_actions=True):
        Base action = the base policy's integrated command, target = recorded command.
        Matches online training, where the env executes base command + residual.

    Returns the number of transitions added.
    """
    if policy_base_actions and base_policy is None:
        raise ValueError("base_policy must be provided when policy_base_actions=True")

    h5_path = resolve_dataset_path(dataset_path)
    print("Populating offline buffer from dataset...")
    transitions = 0

    with h5py.File(h5_path, 'r') as f:
        for episode in tqdm(episodes, desc="Processing offline dataset"):
            images = _read_episode_images(f, h5_path.parent, episode, img_h, img_w)
            states = torch.tensor(episode.states).float()
            commands = torch.tensor(episode.commands).float()
            rewards = torch.tensor(episode.rewards).float()
            assert len(images) == len(states) == len(commands) == len(rewards)

            # Cached previous frame, paired with the current one to form a transition
            prev_obs: dict | None = None
            prev_command: torch.Tensor | None = None
            # Base policy predicts a chunk at a time; integrated once, consumed one per step
            base_commands = None

            for step, (image, state, command, reward) in enumerate(zip(images, states, commands, rewards)):
                # ------------------------------------------------------------------
                # Build observation and action directly for replay buffer ----------
                # ------------------------------------------------------------------
                # Extract data and keep on CPU (replay buffer uses CPU storage)
                done_flag = step == len(images) - 1

                # Generate the base command based on the selected mode
                if policy_base_actions:
                    # Use base policy to generate base command from current observation
                    # Build raw observation first for base policy inference
                    raw_obs = {'rgb': rearrange(image.unsqueeze(0).unsqueeze(0).to(
                        device) / 255.0, 'B T H W C -> B T C H W'), 'state': state.unsqueeze(0).unsqueeze(0).to(device), }

                    # Predict a chunk and integrate it against this frame's recorded pose,
                    # which anchors the deltas the policy emits (command[:6] is that pose)
                    if base_commands is None:
                        with torch.no_grad():
                            chunk = base_policy.predict_action(raw_obs).squeeze(0)
                        des_poses, des_gripper, _ = base_policy.integrate_actions(
                            chunk,
                            curr_pose=command[:6].numpy(),
                            curr_gripper_width=float(episode.gripper_widths[step]),
                        )
                        base_commands = torch.from_numpy(
                            np.concatenate([des_poses, des_gripper[:, None]], -1)).float()
                    base_command = base_commands[0]

                    base_commands = base_commands[1:] if len(base_commands) > 1 else None
                else:
                    assert False, f"Not Implemented"

                # Build observation dict directly in target format
                curr_obs = {"observation.state": state.cpu(), "observation.base_action": base_command.cpu(),
                            "observation.rgb": image.cpu()}

                # Convert images to uint8 for memory-efficient storage
                to_uint8(curr_obs, ["observation.rgb"])

                # ------------------------------------------------------------------
                # If we already cached the *previous* frame for this episode we can
                # create transitions now.
                # ------------------------------------------------------------------
                if prev_obs is not None:
                    transition = TensorDict(
                        {
                            "obs": TensorDict(prev_obs, batch_size=[]),
                            # the executed action is the recorded reference command
                            "action": prev_command,
                            "next": TensorDict(
                                {
                                    "obs": TensorDict(curr_obs, batch_size=[]),
                                    "done": torch.tensor(done_flag, dtype=torch.bool),
                                    "reward": torch.as_tensor(reward, dtype=torch.float32),
                                },
                                batch_size=[],
                            ),
                            # High initial priority for new samples
                            "_priority": torch.tensor(10.0, dtype=torch.float32),
                        },
                        batch_size=[],
                    ).unsqueeze(0)

                    rb.add(transition)
                    transitions += 1

                # Cache current frame for pairing with the next one ---------------
                prev_obs, prev_command = curr_obs, command

    # Log final statistics
    print(f"Added {transitions} transitions")

    return transitions
