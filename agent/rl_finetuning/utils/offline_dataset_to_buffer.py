"""Raw dataset (``episodeNNNNNN/rawdata.h5``) → residual-RL replay-buffer transitions.

Raw episodes are what the robot actually logged: each stream carries its own clock and
its own rate -- robot state at ~500 Hz, gripper at ~230 Hz, camera at ~30 Hz, operator
commands at ~100 Hz -- so nothing lines up frame to frame. Everything here is resampled
onto one grid at ``hz`` (zero-order hold on each stream's own timestamps, the same
scheme ``scripts/rawdata_to_dataset.py`` uses), which should be the rate the QAgent runs
at so the transitions match what it will see online.

Two stages, mirroring how training uses them:
  1. :func:`parse_offline_dataset` -- resample states/commands/rewards for every episode.
     Cheap: it never touches the images. Its transition count sizes the buffer.
  2. :func:`populate_offline_buffer` -- decode the images the grid selects and push the
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

from agent.evaluate.realtime_chunking import _Chunk
from agent.utils.utils import gripper_action, find_chunks
from agent.model.policy import DiffusionPolicy
from env import GRIP_OPEN, GRIP_CLOSED

# policy.obs_fields -> where that field lives in rawdata.h5. Mirrors what
# scripts/rawdata_to_dataset.py writes into dataset.h5 and OBS_FIELD_GETTERS reads live.
RAW_OBS_FIELDS = {
    'pose': ('robot_obs/time', 'robot_obs/actual_pose'),
    'force': ('robot_obs/time', 'robot_obs/actual_force'),  # smoothed below, like the dataset's
    'gripper_width': ('gripper_obs/time', 'gripper_obs/gripper_width'),
    'gripper_force': ('gripper_obs/time', 'gripper_obs/gripper_force'),
}
FORCE_EWMA_ALPHA = 0.03  # scripts/rawdata_to_dataset.py default

# Spacing of the actions *within* a predicted chunk. This is a property of the data the
# base policy was trained on, not of the agent's control rate -- the two are independent,
# and the chunk gets resampled onto the agent's grid below. It is not recorded in the
# policy checkpoint, so it has to be stated here; BasePolicy uses the same 1/20 default
# (agent/rl_finetuning/wrappers/rl_env.py).
BASE_POLICY_HZ = 20.0


@dataclass
class OfflineEpisode:
    """One resampled episode; every array is aligned to `times`, one row per frame."""

    ep_idx: int  # index of the episode in the dataset
    path: pathlib.Path  # the episodeNNNNNN directory
    length: int
    times: np.ndarray  # (T,) sample grid, seconds from the episode start
    states: np.ndarray  # (T, state_dim) in policy.obs_fields order
    # The reference command the demonstration recorded, in the same space the base
    # policy's integrated commands live in: absolute pose [tx, ty, tz, rx, ry, rz]
    # plus gripper state (GRIP_OPEN=0 / GRIP_CLOSED=1).
    commands: np.ndarray  # (T, 7)
    rewards: np.ndarray  # (T,)
    gripper_widths: np.ndarray  # (T,) raw widths, used to integrate the base policy chunk


def episode_dirs(dataset_path: str | pathlib.Path, num_episodes: int | None = None) -> list[pathlib.Path]:
    """The first *num_episodes* ``episodeNNNNNN`` directories holding a ``rawdata.h5``."""
    root = pathlib.Path(dataset_path)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a dataset directory")

    dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / 'rawdata.h5').is_file())
    if not dirs:
        raise FileNotFoundError(f"No episodeNNNNNN/rawdata.h5 found under {root}")
    return dirs[:num_episodes] if num_episodes is not None else dirs


def _ewma(x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponentially weighted moving average (scripts/rawdata_to_dataset.py:13)."""
    ema = np.zeros_like(x)
    ema[0] = x[0]
    for i in range(1, len(x)):
        ema[i] = alpha * x[i] + (1 - alpha) * ema[i - 1]
    return ema


def _sample_grid(f: h5py.File, hz: float) -> np.ndarray:
    """The common time grid the streams get resampled onto."""
    last = max(float(f[f'{grp}/time'][-1]) for grp in ('robot_obs', 'gripper_obs', 'camera_obs'))
    dt = 1.0 / hz
    return np.arange(dt, last, dt)


def _zero_order_hold(times: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    """Index of the most recent sample at or before each grid point."""
    return np.clip(np.searchsorted(times, sample_times, side='right') - 1, 0, len(times) - 1)


def parse_offline_dataset(
    dataset_path: str | pathlib.Path,
    base_policy: DiffusionPolicy,
    hz: float,
    num_episodes: int | None = None,
    g_thr: float = 18,
) -> tuple[list[OfflineEpisode], int]:
    """Resample each raw episode onto a `hz` grid: states, reference commands, rewards.

    The observation fields come from ``base_policy.obs_fields``, so the state vector is
    laid out exactly as the policy was trained on.

    The recorded command is absolute (pose + gripper state) rather than one of the delta
    encodings the policy is *trained* on: a delta is only meaningful next to the pose it
    is anchored to, and the base policy's predictions are integrated back into absolute
    commands before they enter the buffer (see :func:`populate_offline_buffer`), so both
    sides of the residual live in the same space.

    Rewards are derived from the gripper open/close phases: the episode must split into
    the 5 expected segments (approach, pick-and-transport-plugin, release-to-unplug,
    unplug-place, release); episodes that do not are skipped. Reward is 1 from shortly
    before the plug-in until just after the release, and 2 afterwards.

    Returns the parsed episodes and the total number of frames they cover.
    """
    episodes: list[OfflineEpisode] = []
    total_transitions = 0

    for ep_idx, ep_dir in enumerate(episode_dirs(dataset_path, num_episodes)):
        with h5py.File(ep_dir / 'rawdata.h5', 'r') as f:
            sample_times = _sample_grid(f, hz)

            fields = []
            for key in base_policy.obs_fields:
                if key not in RAW_OBS_FIELDS:
                    raise KeyError(f"obs field {key!r} has no raw-dataset mapping "
                                   f"(known: {sorted(RAW_OBS_FIELDS)})")
                time_key, value_key = RAW_OBS_FIELDS[key]
                values = np.asarray(f[value_key])
                if key == 'force':
                    values = _ewma(values, alpha=FORCE_EWMA_ALPHA)
                values = values[_zero_order_hold(np.asarray(f[time_key]), sample_times)]
                fields.append(values if values.ndim == 2 else values[:, None])
            state = np.concatenate(fields, -1)

            robot_idx = _zero_order_hold(np.asarray(f['robot_obs/time']), sample_times)
            gripper_idx = _zero_order_hold(np.asarray(f['gripper_obs/time']), sample_times)
            poses = np.asarray(f['robot_obs/actual_pose'])[robot_idx]
            gripper_widths = np.asarray(f['gripper_obs/gripper_width'])[gripper_idx]

        length = len(sample_times)
        g_action = gripper_action(gripper_widths, threshold=g_thr)

        chunks = find_chunks(g_action.flatten())
        if not (len(chunks) == 5 and chunks[2][-1] == 1):
            print(f"Skip episode {ep_dir.name}: expected 5 gripper segments, got {len(chunks)}")
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
                path=ep_dir,
                length=length,
                times=sample_times,
                states=state,
                commands=np.concatenate([poses, g_state], -1),
                rewards=reward,
                gripper_widths=gripper_widths,
            )
        )
        total_transitions += length

    print('Loaded episode count:', len(episodes), '\tTotal transitions:', total_transitions)
    return episodes, total_transitions


def _read_episode_images(episode: OfflineEpisode, img_size: int) -> torch.Tensor:
    """Decode the camera frames the sample grid lands on → uint8 tensor (T, H, W, C).

    The camera runs slower than the grid, so frames repeat; each one is decoded once.
    Raw frames are BGR (``camera_obs/image_bgr``) and are flipped to RGB here, the same
    conversion ``scripts/rawdata_to_dataset.py:56`` applies when building dataset.h5.

    Kept as uint8, which is both how the buffer stores images and what the online path
    puts in: converting to float here and letting ``to_uint8`` rescale would blow the
    images out, since that helper assumes floats already live in [0, 1].
    """
    with h5py.File(episode.path / 'rawdata.h5', 'r') as f:
        frame_idx = _zero_order_hold(np.asarray(f['camera_obs/time']), episode.times)
        unique_idx, inverse = np.unique(frame_idx, return_inverse=True)

        images = f['camera_obs/image_bgr']
        decoded = np.stack([
            np.asarray(Image.fromarray(np.asarray(images[i])[..., ::-1]).resize((img_size, img_size)))
            for i in unique_idx
        ])

    return torch.from_numpy(decoded[inverse])


def populate_offline_buffer(
    dataset_path: str | pathlib.Path,
    base_policy: DiffusionPolicy,
    hz: float,
    rb: ReplayBuffer,
    num_episodes: int | None = None,
    base_hz: float = BASE_POLICY_HZ,
) -> int:
    """Turn consecutive frames of each raw episode into residual RL transitions.

    Everything about the observation format -- image size, state fields, action encoding,
    device -- is read off *base_policy*, so the buffer matches what that policy expects.

    The base policy predicts *deltas* (per its action_mode), which only mean something
    next to the pose they are anchored to, so each predicted chunk is integrated against
    the frame's recorded pose (``policy.integrate_actions``) into absolute commands --
    the same thing ``get_base_action`` hands the online path. The transition then stores
    the recorded command as the action, so the residual the QAgent has to learn is
    exactly ``recorded_command - base_policy_command``.

    A chunk's actions are spaced at ``base_hz``, which is fixed by the base policy and
    independent of the rate the agent runs at. When ``hz > base_hz`` the chunk is
    resampled onto the agent's grid rather than replayed one action per frame (which
    would play the base policy back ``hz / base_hz`` times too fast). Resampling reuses
    the online path's own interpolation (``realtime_chunking._Chunk``), so both sides
    treat a chunk identically -- linear in translation, Slerp in rotation, clamped at the
    ends. Unlike online, only one chunk is live at a time here: offline there is no
    inference latency to model, so there are no overlapping chunks to ensemble.

    Returns the number of transitions added.
    """
    device = next(base_policy.parameters()).device
    img_size = base_policy.img_size

    episodes, _ = parse_offline_dataset(dataset_path, base_policy, hz, num_episodes=num_episodes)

    print("Populating offline buffer from raw dataset...")
    transitions = 0

    for episode in tqdm(episodes, desc="Processing offline dataset"):
        images = _read_episode_images(episode, img_size)
        states = torch.tensor(episode.states).float()
        commands = torch.tensor(episode.commands).float()
        rewards = torch.tensor(episode.rewards).float()
        assert len(images) == len(states) == len(commands) == len(rewards)

        # Cached previous frame, paired with the current one to form a transition
        prev_obs: dict | None = None
        prev_command: torch.Tensor | None = None
        # Current chunk of base commands, resampled to whatever time we ask it for
        chunk_commands: _Chunk | None = None

        for step, (image, state, command, reward) in enumerate(zip(images, states, commands, rewards)):
            # ------------------------------------------------------------------
            # Build observation and command directly for replay buffer ----------
            # ------------------------------------------------------------------
            # Extract data and keep on CPU (replay buffer uses CPU storage)
            done_flag = step == len(images) - 1

            # Predict a fresh chunk once the current one no longer reaches this frame.
            # The tolerance keeps the cadence off the floating-point noise in the grid,
            # so a chunk always covers its full span rather than one frame less.
            t = float(episode.times[step])
            if chunk_commands is None or t > chunk_commands.t_end + 1e-9:
                # Build raw observation first for base policy inference
                raw_obs = {'rgb': rearrange(image.unsqueeze(0).unsqueeze(0).float().to(
                    device) / 255.0, 'B T H W C -> B T C H W'), 'state': state.unsqueeze(0).unsqueeze(0).to(device), }

                # Predict a chunk and integrate it against this frame's recorded pose,
                # which anchors the deltas the policy emits (command[:6] is that pose)
                with torch.no_grad():
                    chunk = base_policy.predict_action(raw_obs).squeeze(0)
                des_poses, des_gripper, des_done = base_policy.integrate_actions(
                    chunk,
                    curr_pose=command[:6].numpy(),
                    curr_gripper_width=float(episode.gripper_widths[step]),
                )
                chunk_commands = _Chunk(t, des_poses, des_gripper, des_done, action_dt=1.0 / base_hz)

            des_pose, des_gripper_state, _ = chunk_commands.interp(t)
            # the gripper is a discrete state; interpolating it gives a fraction, and the
            # env rounds the same way before executing (rl_env.BasePolicyVecEnvWrapper.step)
            base_command = torch.tensor(
                np.concatenate([des_pose, [round(des_gripper_state)]]), dtype=torch.float32)

            # Build observation dict directly in target format; the image is already
            # uint8, which is the memory-efficient form the buffer stores
            curr_obs = {"observation.state": state.cpu(), "observation.base_action": base_command.cpu(),
                        "observation.rgb": image.cpu()}

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
