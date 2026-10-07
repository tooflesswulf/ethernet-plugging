"""
Real-time action chunking for asynchronous policy execution on hardware.

The diffusion policy predicts an action *chunk* — a sequence of absolute desired
poses + gripper widths — from an observation captured at some time ``t_obs``. The
i-th action of a chunk is meant to be executed at ``t_obs + i * action_dt``.

Diffusion inference is slow and runs asynchronously in its own loop, so by the
time a chunk is ready the world has already moved on, and several chunks (each
anchored at a different, slightly stale observation time) overlap in time. This
module keeps the recent chunks in a buffer and, when asked for the action to
execute *now*, interpolates every overlapping chunk to the query time and returns
a recency-weighted average. Fresher chunks (more recent observations) get more
weight, which both smooths the commanded trajectory and lets newer information
take over as it arrives — the same idea as ACT's temporal ensembling, generalized
to chunks that arrive at irregular, continuous times.

Poses are averaged in absolute world coordinates (translation linearly, rotation
via weighted rotation averaging), so the chunks must already be integrated into
absolute poses (see ``DiffusionPolicy.integrate_actions``). Impedance gains, when the
policy predicts them, ride along as log10 values and are averaged linearly in log space
(a weighted geometric mean of the gains).

Each chunk's weight is also tapered: it ramps in from zero over ``taper`` seconds after
the chunk arrives, and out to zero over the last ``taper`` seconds of its horizon. An
untapered chunk enters and leaves the average at full weight, so the command steps by its
share of how far it disagrees with the rest -- at 3-4 chunks/s that was most of the
4-10 Hz target jitter in logs-policy-imp (2026-10-07; halved by a 0.15 s taper in replay).
"""

import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


class _Chunk:
    """A single predicted action chunk, anchored at its observation time."""

    __slots__ = ('t_obs', 't_add', 'poses', 'widths', 'dones', 'gains', 'times')

    def __init__(self, t_obs, poses, widths, dones, action_dt, gains=None, t_add=None):
        self.t_obs = t_obs
        self.t_add = t_obs if t_add is None else t_add  # when it reached the buffer
        self.poses = np.asarray(poses, dtype=float)    # (H, 6) [tx,ty,tz, rx,ry,rz]
        self.widths = np.asarray(widths, dtype=float)  # (H,)
        self.dones = np.asarray(dones, dtype=float)    # (H,) end-of-episode score in [0, 1]
        # (H, P) log10 impedance gains; (H, 0) when the policy predicts none
        self.gains = np.zeros((len(self.poses), 0)) if gains is None else np.asarray(gains, dtype=float)
        # absolute execution time of each action in the chunk
        self.times = t_obs + np.arange(len(self.poses)) * action_dt

    @property
    def t_end(self):
        return self.times[-1]

    def interp(self, t_query):
        """
        Interpolate this chunk to ``t_query``.

        Returns ``(pose (6,), width, done, gains (P,))``. Queries outside the chunk
        are clamped to its first / last action.
        """
        times = self.times
        if t_query >= times[-1]:
            return self.poses[-1].copy(), float(self.widths[-1]), float(self.dones[-1]), self.gains[-1].copy()
        if t_query <= times[0]:
            return self.poses[0].copy(), float(self.widths[0]), float(self.dones[0]), self.gains[0].copy()

        # locate the segment [i, i+1] that brackets t_query
        i = int(np.searchsorted(times, t_query, side='right')) - 1
        i = min(max(i, 0), len(times) - 2)
        t0, t1 = times[i], times[i + 1]
        frac = (t_query - t0) / (t1 - t0)

        p0, p1 = self.poses[i], self.poses[i + 1]
        trans = (1.0 - frac) * p0[:3] + frac * p1[:3]
        rot = Slerp(
            [t0, t1], R.from_rotvec([p0[3:], p1[3:]]),
        )(t_query).as_rotvec()
        width = (1.0 - frac) * self.widths[i] + frac * self.widths[i + 1]
        done = (1.0 - frac) * self.dones[i] + frac * self.dones[i + 1]
        gains = (1.0 - frac) * self.gains[i] + frac * self.gains[i + 1]
        return np.concatenate([trans, rot]), float(width), float(done), gains


class RealtimeActionChunkingBuffer:
    """
    Thread-safe buffer that ensembles overlapping async action chunks.

    The producer (diffusion prediction loop) calls :meth:`add_chunk` whenever a new
    chunk is ready, tagging it with the time the *observation* was captured. The
    consumer (real-time control loop) calls :meth:`get_action` at the control rate
    to obtain the action to execute now.

    Args:
        action_dt:    seconds between consecutive actions within a chunk
                      (i.e. 1 / control_frequency).
        weight_decay: exponential recency-weighting rate (1/seconds). The weight of
                      a chunk whose observation is ``age`` seconds old at query time
                      is ``exp(-weight_decay * age)``. Larger -> trust fresh chunks
                      more / older chunks fade faster. ``0`` gives a plain average.
        taper:        seconds over which a chunk's weight ramps in after it arrives and
                      out before its last action (see the module docstring). ``0`` = off.
        max_chunks:   hard cap on retained chunks (oldest dropped first).
    """

    def __init__(self, action_dt, weight_decay=2.0, taper=0.15, max_chunks=32):
        self.action_dt = float(action_dt)
        self.weight_decay = float(weight_decay)
        self.taper = float(taper)
        self.max_chunks = int(max_chunks)

        self._chunks: list[_Chunk] = []
        self._chunk_count: int = 0
        self._lock = threading.Lock()
        self._logs = []

    def dolog(self, chunk, obs_state, time):
        self._logs.append({
            'chunk': chunk,
            'obs': obs_state,
            't': time  # Time of chunk add
        })

    def add_chunk(self, t_obs, des_poses, des_widths, des_dones, des_gains=None, t_add=None):
        """
        Insert a freshly predicted chunk anchored at observation time ``t_obs``.
        des_gains: optional (H, P) log10 impedance gains.
        t_add:     arrival time, where its taper starts (default: time.time() now).
        """
        chunk = _Chunk(t_obs, des_poses, des_widths, des_dones, self.action_dt, des_gains,
                       time.time() if t_add is None else t_add)
        with self._lock:
            self._chunks.append(chunk)
            # keep newest first; bound memory
            self._chunks.sort(key=lambda c: c.t_obs, reverse=True)
            if len(self._chunks) > self.max_chunks:
                self._chunks = self._chunks[:self.max_chunks]
            self._chunk_count += 1
        return chunk

    def get_action(self, t_query, t_now=None):
        """
        Recency-weighted, tapered average of every chunk still active at ``t_query``.
        t_now: the current time (default time.time()), for the ramp-in; ``t_query`` may
        look ahead of it.

        Returns ``(des_pose (6,), des_width float, done float, gains (P,))`` or ``None``
        when no chunk covers the query time (e.g. before the first prediction lands, or
        after a long prediction stall). ``done`` is the recency-weighted end-of-episode
        score in [0, 1] for the action executed now; ``gains`` the recency-weighted log10
        impedance gains (empty without them). The caller decides how to handle
        ``None`` — e.g. hold the previous command.
        """
        with self._lock:
            # prune expired / stale chunks while we hold the lock
            self._chunks = [c for c in self._chunks if c.t_end > t_query]
            chunks = list(self._chunks)

        t_now = time.time() if t_now is None else t_now
        poses, widths, dones, gains, log_w, taper = [], [], [], [], [], []
        for c in chunks:
            interp = c.interp(t_query)
            if interp is None:
                continue
            pose, width, done, gain = interp
            age = max(t_query - c.t_obs, 0.0)
            poses.append(pose)
            widths.append(width)
            dones.append(done)
            gains.append(gain)
            log_w.append(-self.weight_decay * age)
            if self.taper > 0:
                taper.append(np.clip((t_now - c.t_add) / self.taper, 0.0, 1.0)
                             * np.clip((c.t_end - t_query) / self.taper, 0.0, 1.0))

        if not poses:
            return None

        # Relative to the freshest chunk: exp(-weight_decay * age) alone underflowed to all
        # zeros for a large weight_decay, and the normalization below made NaN poses.
        log_w = np.asarray(log_w, dtype=float)
        weights = np.exp(log_w - log_w.max())
        if taper and np.dot(weights, taper) > 0:
            # all-zero only with every chunk just arrived (the first one) or about to end
            # (a prediction stall): then plain recency weights, rather than no action
            weights = weights * np.asarray(taper)
        weights /= weights.sum()
        poses = np.asarray(poses)

        trans = (weights[:, None] * poses[:, :3]).sum(axis=0)
        rot = R.from_rotvec(poses[:, 3:]).mean(weights=weights).as_rotvec()
        width = float(np.dot(weights, widths))
        done = float(np.dot(weights, dones))
        gain = weights @ np.asarray(gains)
        return np.concatenate([trans, rot]), width, done, gain

    def is_empty(self):
        with self._lock:
            return len(self._chunks) == 0

    def clear(self):
        with self._lock:
            self._chunks.clear()
