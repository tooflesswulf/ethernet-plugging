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
absolute poses (see ``DiffusionPolicy.integrate_actions``).
"""

import threading

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


class _Chunk:
    """A single predicted action chunk, anchored at its observation time."""

    __slots__ = ('t_obs', 'poses', 'widths', 'times')

    def __init__(self, t_obs, poses, widths, action_dt):
        self.t_obs = t_obs
        self.poses = np.asarray(poses, dtype=float)    # (H, 6) [tx,ty,tz, rx,ry,rz]
        self.widths = np.asarray(widths, dtype=float)  # (H,)
        # absolute execution time of each action in the chunk
        self.times = t_obs + np.arange(len(self.poses)) * action_dt

    @property
    def t_end(self):
        return self.times[-1]

    def interp(self, t_query):
        """
        Interpolate this chunk to ``t_query``.

        Returns ``(pose (6,), width)`` or ``None`` if ``t_query`` is past the end
        of the chunk (the chunk no longer has anything to say about that time).
        Queries before the chunk start are clamped to the first action.
        """
        times = self.times
        if t_query >= times[-1]:
            return self.poses[-1].copy(), float(self.widths[-1])
        if t_query <= times[0]:
            return self.poses[0].copy(), float(self.widths[0])

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
        return np.concatenate([trans, rot]), float(width)


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
        max_age:      drop chunks whose observation is older than this many seconds.
        max_chunks:   hard cap on retained chunks (oldest dropped first).
    """

    def __init__(self, action_dt, weight_decay=2.0, max_chunks=32):
        self.action_dt = float(action_dt)
        self.weight_decay = float(weight_decay)
        self.max_chunks = int(max_chunks)
        self.rm_age = -np.log(.1) / weight_decay

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

    def add_chunk(self, t_obs, des_poses, des_widths):
        """Insert a freshly predicted chunk anchored at observation time ``t_obs``."""
        chunk = _Chunk(t_obs, des_poses, des_widths, self.action_dt)
        with self._lock:
            self._chunks.append(chunk)
            # keep newest first; bound memory
            self._chunks.sort(key=lambda c: c.t_obs, reverse=True)
            if len(self._chunks) > self.max_chunks:
                self._chunks = self._chunks[:self.max_chunks]
            self._chunk_count += 1
        return chunk

    def get_action(self, t_query):
        """
        Recency-weighted average of every chunk still active at ``t_query``.

        Returns ``(des_pose (6,), des_width float)`` or ``None`` when no chunk
        covers the query time (e.g. before the first prediction lands, or after a
        long prediction stall). The caller decides how to handle ``None`` — e.g.
        hold the previous command.
        """
        with self._lock:
            # prune expired / stale chunks while we hold the lock
            self._chunks = [
                c for c in self._chunks
                if c.t_end > t_query #or t_query - c.t_obs < self.rm_age
            ]
            chunks = list(self._chunks)

        poses, widths, weights = [], [], []
        for c in chunks:
            interp = c.interp(t_query)
            if interp is None:
                continue
            pose, width = interp
            age = max(t_query - c.t_obs, 0.0)
            poses.append(pose)
            widths.append(width)
            weights.append(np.exp(-self.weight_decay * age))

        if not poses:
            return None

        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        poses = np.asarray(poses)

        trans = (weights[:, None] * poses[:, :3]).sum(axis=0)
        rot = R.from_rotvec(poses[:, 3:]).mean(weights=weights).as_rotvec()
        width = float(np.dot(weights, widths))
        return np.concatenate([trans, rot]), width

    def is_empty(self):
        with self._lock:
            return len(self._chunks) == 0

    def clear(self):
        with self._lock:
            self._chunks.clear()
