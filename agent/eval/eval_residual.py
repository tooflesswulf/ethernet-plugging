"""
Residual evaluation / online RL on the split pipeline (eval_splits.py): a correction added
to the insert policy's commands, one episode per ResidualEnv.run(), with the loop -- and
any learning between episodes -- owned by the calling script:

    env = ResidualEnv(insert_ckpt=...)          # spawns the policy workers, homes the robot
    for _ in range(N):
        env.register(actor)                     # step -> Correction, every `period` s of insert
        episode = env.run()                     # one episode, then the robot is idle
        ...                                     # update the actor, save, decide whether to go on
    env.close()

run() plays the eval_splits phases once:

    pickup     pickup policy
    transport  scripted, to Teleoperation.port_pose
    insert     insert policy + the registered correction: every `period` the callback gets
               an InsertStep and returns a Correction (pose offset, log-gain offsets),
               held in between. Ends on the network, DpadUp, the done score (insert_done)
               or insert_timeout.
    unplug     scripted, as Teleoperation.unplug_and_release
    done       the robot stops (Env threads stopped: nothing runs in this process but the
               script, so it can train freely); run() returns the Episode

The next run() re-homes the robot and, with wait_circle, waits for Circle (time to fix the
scene). Square aborts the episode (run() returns it with aborted=True); Square while
waiting for Circle raises RuntimeError, as in eval_splits. The policy workers live until
close().

Safety: each correction is clamped to max_dpos / max_drot / max_dlog_gain and the pose
offset slew-limited to max_dpos_rate / max_drot_rate, whatever the callback asks for.

With log_dir, every run() is its own episode<NNNNNN>/ (rawdata.h5, phases.json,
chunks_<phase>.npz, as eval_splits) plus insert.pkl, the returned Episode.
"""
from agent.eval.eval_splits import EvalSplits
from agent.eval.realtime_chunking import RealtimeActionChunkingBuffer
from agent.utils.robot_utils import GAIN_KWARGS
from robot.env import GRIP_OPEN
from robot import fdcc
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, field
import numpy as np
import threading
import argparse
import pickle
import time
import os

DEFAULT_PICKUP_CKPT = '../ckpts/ethernet-impedance-v2v3-pickup-tgt-noforce/ckpt_final.pth'


# ============================================================
# What the correction callback sees and returns
# ============================================================
@dataclass
class InsertStep:
    """What the callback sees at a decision, all at the same control tick."""
    t: float                    # s since insert started
    state: dict                 # env obs['state'] (actual_pose, filtered_force, target_pose, ...);
                                # target_pose includes the correction in effect
    network_status: bool
    image: np.ndarray | None    # BGR camera frame, only if registered with images=True
    base_pose: np.ndarray       # (6,) insert policy's command [x, y, z, rotvec], after the low-pass
    base_width: float
    base_done: float            # its done score ([0, 1]; 0 without a done head)
    base_log_gains: dict        # {field: (6,) log10 gains} the policy predicts; {} without a head
    prev: 'Correction'          # the correction applied since the last decision (after limits)


@dataclass
class Correction:
    """
    Added to the insert policy's command, and held until the next decision.

    dpose:      (6,) [dx, dy, dz (m), rotvec (rad)] in base axes: pose = [p + d, R(d) R(p)].
    dlog_gains: {field: (6,)} added to the commanded log10 gains, field in 'stiffness' /
                'damping' / 'mass'. For a field the insert policy does not predict, added
                to fdcc.toml's gains.
    info:       anything to keep with the step (log-prob, the actor's raw action, ...).

    The insert policy observes the corrected robot, target_pose included, and a
    target-anchored policy integrates each chunk from that target: a held correction is
    taken into the policy's own command chunk by chunk (step.base_pose carries it), and
    added again on top. To hold the command at an offset from where the policy would put
    it, recompute dpose from step.base_pose at every decision.
    """
    dpose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    dlog_gains: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


class ResidualActor:
    """Optional base for a correction callback: register(actor) calls actor(step) = act(step)."""

    def act(self, step: InsertStep) -> Correction:
        raise NotImplementedError

    def __call__(self, step):
        return self.act(step)


@dataclass
class Episode:
    """
    What run() returns. Step and tick arrays share their first axis; times are seconds
    since insert started. Empty steps / ticks if insert was never reached.

    steps (one per decision):
        t, state {key: (n, ...)}, network_status, image (n, H, W, 3) or None,
        base_pose (n, 6), base_width, base_done, base_log_gains {field: (n, 6)},
        dpose_req (n, 6) / dpose (n, 6)   asked for / clamped (the pose offset slews to it),
        dlog_gains_req / dlog_gains       {field: (n, 6)}, likewise,
        info                              list of Correction.info
        act_s                             s the callback took
    ticks (every control tick of insert with a command, so force peaks between decisions
    show up): t, filtered_force (m, 6), actual_pose (m, 6), target_pose (m, 6),
        base_pose (m, 6), dpose (m, 6) applied, network_status (m,)
    success:  the network came up during insert
    end:      what ended insert: 'network', 'DpadUp', 'done=0.87', 'timeout', 'aborted',
              or None if it was never reached
    duration: s of insert
    aborted:  Square (or an error) ended the episode
    phases:   phases.json's list (phase, t_start, t_end, end)
    path:     the episode's log directory (None without log_dir)
    timing:   {'act_max_s': slowest callback, 'late': decisions whose callback overran the
              control period (delaying that command), 'over_period': ... overran `period`,
              'missed': decisions skipped because the loop fell behind}
    """
    index: int
    steps: dict
    ticks: dict
    success: bool
    end: str | None
    duration: float
    aborted: bool
    phases: list
    path: str | None
    timing: dict = field(default_factory=dict)


# ============================================================
# Helpers
# ============================================================
def apply_dpose(pose, d):
    """pose (6,) corrected by d (6,) [dx, dy, dz, rotvec] in base axes: [p + d, R(d) R(p)]."""
    pose = np.asarray(pose, float)
    if not d[3:].any():
        return np.r_[pose[:3] + d[:3], pose[3:]]
    return np.r_[pose[:3] + d[:3], (R.from_rotvec(d[3:]) * R.from_rotvec(pose[3:])).as_rotvec()]


def _stack(rows):
    """[{key: value}] -> {key: (n, ...) array}."""
    return {k: np.asarray([r[k] for r in rows]) for k in rows[0]} if rows else {}


def _stack_gains(rows):
    """[{field: (6,)}] -> {field: (n, 6)}; a row without a field has 0 there."""
    fields = sorted(set().union(*rows)) if rows else []
    return {f: np.asarray([r.get(f, np.zeros(6)) for r in rows], float) for f in fields}


# ============================================================
# The robot side
# ============================================================
class _ResidualSplits(EvalSplits):
    """EvalSplits with the insert correction, re-armed for another episode by rearm()."""

    def __init__(self, pickup_ckpt, insert_ckpt, insert_timeout=None, limits=None, **kwargs):
        self.correction, self.period, self.images = None, 0.05, False
        self.insert_timeout = insert_timeout
        self.limits = limits
        self.index = 0
        self._dpose = np.zeros(6)                   # correction applied to the last command
        super().__init__(pickup_ckpt, insert_ckpt, **kwargs)     # homes the robot, starts env
        self._begin_episode()

    def _begin_episode(self):
        self.armed = True               # Env threads running, the robot servoing
        self.error = None               # an exception from the correction callback
        self.result = None              # the Episode, once end_episode has run
        self._t_insert = None
        self._end = None
        self._steps, self._ticks = [], []

    def rearm(self):
        """Back to the state __init__ leaves: robot homed and servoing, at pickup."""
        self.index += 1
        # A sequence cut short by Square still shadows get_action
        if getattr(self, '_interrupt_sequence', None) is not None:
            del self.get_action
            self._interrupt_sequence = None
        env = self.env
        env.threads = []                # joined by Env.close, which saved them; reset() would save again
        env.network_status = False      # until its loop polls again
        env.reset(self.home_pose)
        env.start()                     # a new episode directory
        self.stop_event.clear()
        self.last_action = (self.home_pose, GRIP_OPEN, 0., False)
        iface = self.iface              # as RobotExecution.__init__ left it: wait_for_circle
        iface.targ_pose = np.array(self.home_pose)          # waits for Circle to close it
        iface.targ_zforce, iface.adaptive_mode, iface.gripper_state = 0., False, GRIP_OPEN

        self.buffers = {p: RealtimeActionChunkingBuffer(action_dt=b.action_dt, weight_decay=b.weight_decay,
                                                        taper=b.taper) for p, b in self.buffers.items()}
        self.phase, self.phase_t_start, self.phase_log = 'pickup', time.time(), []
        (self.worker, self.policy), self.buffer = self.policies['pickup'], self.buffers['pickup']
        if self.lowpass:
            self.lowpass.reset()
        self.prediction_thread = threading.Thread(target=self.prediction_loop)
        self._dpose = np.zeros(6)
        self._begin_episode()

    # ============================================================
    # Insert
    # ============================================================
    def set_phase(self, phase, reason=None):
        if self.phase == 'insert' and phase != 'insert':
            self.end_insert(reason)
        super().set_phase(phase, reason)
        if phase == 'insert':
            self.begin_insert()

    def begin_insert(self):
        self._t_insert = time.time()
        self._next_decision = 0.
        self._dpose = np.zeros(6)
        self._corr = Correction()           # the clamped correction, held between decisions
        self._gain_fields = set()           # fields corrected this episode (keep commanding them)
        self._t_cmd = None
        self._timing = {'act_max_s': 0., 'late': 0, 'over_period': 0, 'missed': 0}
        self._t_warn = 0.

    def end_insert(self, reason):
        self._end, self._duration = reason, time.time() - self._t_insert
        self._dpose = np.zeros(6)
        tm = self._timing
        if tm['late'] or tm['missed']:
            print(f'\nresidual: {tm["late"]}/{len(self._steps)} decisions overran the control period '
                  f'({1000 * self.control_dt:.0f} ms; slowest {1000 * tm["act_max_s"]:.1f} ms), '
                  f'{tm["over_period"]} overran the decision period, {tm["missed"]} decisions missed')

    def warn(self, msg):
        """At most once a second: the control loop prints its status line every tick."""
        now = time.time()
        if now - self._t_warn > 1.0:
            self._t_warn = now
            print(f'\nresidual: {msg}')

    def get_action(self):
        if (self.phase == 'insert' and self.insert_timeout is not None
                and time.time() - self._t_insert > self.insert_timeout):
            return self.start_unplug('timeout')
        return super().get_action()

    def command(self, des_pose, des_width, done, log_gains):
        if self.phase != 'insert':
            return super().command(des_pose, des_width, done, log_gains)
        now = time.time()
        t = now - self._t_insert
        obs = self.env.get_obs()
        base_pose = np.asarray(des_pose, float)

        if self.period is None or t >= self._next_decision:
            step = InsertStep(t=t, state=obs['state'], network_status=obs['network_status'],
                              image=obs['image'] if self.images else None,
                              base_pose=base_pose, base_width=float(des_width), base_done=float(done),
                              base_log_gains=log_gains, prev=self._corr)
            t_act = time.perf_counter()
            try:
                req = (self.correction(step) if self.correction else None) or Correction()
                act_s = time.perf_counter() - t_act
                self._corr = self.clamp(req)
            except Exception as e:
                # end the episode cleanly; ResidualEnv.run raises it once the robot has stopped
                self.error = e
                self.stop()
                return des_pose, des_width
            self._gain_fields |= set(self._corr.dlog_gains)
            self._steps.append({
                't': t, 'state': obs['state'], 'network_status': obs['network_status'],
                'image': step.image,
                'base_pose': base_pose, 'base_width': float(des_width), 'base_done': float(done),
                'base_log_gains': log_gains,
                'dpose_req': np.asarray(req.dpose, float), 'dpose': self._corr.dpose,
                'dlog_gains_req': {f: np.asarray(v, float) for f, v in req.dlog_gains.items()},
                'dlog_gains': self._corr.dlog_gains, 'info': req.info, 'act_s': act_s,
            })
            # The callback runs on the control thread: while it runs, no command goes out
            tm = self._timing
            tm['act_max_s'] = max(tm['act_max_s'], act_s)
            over_period = self.period is not None and act_s > self.period
            tm['over_period'] += over_period
            if act_s > self.control_dt:
                tm['late'] += 1
                self.warn(f'callback took {1000 * act_s:.1f} ms, over the {1000 * self.control_dt:.0f} ms '
                          'control period: this command went out late'
                          + (f' (and over the {1000 * self.period:.0f} ms decision period)' if over_period else ''))
            if self.period is not None:
                self._next_decision += self.period
                if self._next_decision <= t:        # fell behind (a stalled loop): skip ahead
                    missed = int((t - self._next_decision) // self.period) + 1
                    tm['missed'] += missed
                    self.warn(f'{missed} decision(s) missed: the control loop fell '
                              f'{1000 * (t - self._next_decision + self.period):.0f} ms behind the '
                              f'{1000 * self.period:.0f} ms decision period')
                    self._next_decision = t + self.period

        # Pose offset: slewed toward the held correction (a step would be fed forward by fdcc)
        dt = self.control_dt if self._t_cmd is None else min(now - self._t_cmd, 0.1)
        self._t_cmd = now
        lim = self.limits
        err = self._corr.dpose - self._dpose
        for sl, rate in ((slice(0, 3), lim['max_dpos_rate']), (slice(3, 6), lim['max_drot_rate'])):
            n = np.linalg.norm(err[sl])
            if n > rate * dt:
                err[sl] *= rate * dt / n
        self._dpose = self._dpose + err
        pose = apply_dpose(base_pose, self._dpose)

        # Gains: the policy's (or fdcc.toml's) plus the correction
        gains = dict(log_gains)
        for f in self._gain_fields:
            base = gains[f] if f in gains else np.log10(fdcc._six(getattr(self.env.imp.p, GAIN_KWARGS[f])))
            gains[f] = base + self._corr.dlog_gains.get(f, 0.)
        self.apply_gains(gains)

        s = obs['state']
        self._ticks.append({'t': t, 'filtered_force': np.asarray(s['filtered_force'], float),
                            'actual_pose': np.asarray(s['actual_pose'], float),
                            'target_pose': np.asarray(s['target_pose'], float),
                            'base_pose': base_pose, 'dpose': self._dpose,
                            'network_status': obs['network_status']})
        return pose, des_width

    def clamp(self, c):
        """The correction within the limits."""
        lim = self.limits
        d = np.asarray(c.dpose, float).copy()
        assert d.shape == (6,), d.shape
        d[:3] = np.clip(d[:3], -lim['max_dpos'], lim['max_dpos'])
        n = np.linalg.norm(d[3:])
        if n > lim['max_drot']:
            d[3:] *= lim['max_drot'] / n
        g = {}
        for f, v in c.dlog_gains.items():
            if f not in GAIN_KWARGS:
                raise ValueError(f'unknown gain field {f!r} (one of {list(GAIN_KWARGS)})')
            g[f] = np.clip(np.asarray(v, float), -lim['max_dlog_gain'], lim['max_dlog_gain'])
        return Correction(dpose=d, dlog_gains=g, info=c.info)

    # ============================================================
    # Run loop hooks
    # ============================================================
    def runtime_info(self):
        print(f'ep {self.index:3d}  dz {1000 * self._dpose[2]:+5.1f} mm  ', end='')
        super().runtime_info()

    def close(self):
        """End of RobotExecution.run: this episode only (the workers live on, see shutdown)."""
        self.end_episode()

    def end_episode(self):
        if not self.armed:
            return self.result
        aborted = self.phase != 'done'
        if self.phase == 'insert':
            self.end_insert('aborted')
        self.stop()
        self.env.stop_flag = True
        super().end_episode()           # stops the robot, saves rawdata.h5 / phases.json / chunks
        self.armed = False
        steps, reached = self._steps, self._t_insert is not None
        self.result = Episode(
            index=self.index,
            steps={
                **_stack([{k: v for k, v in s.items()
                           if k not in ('state', 'base_log_gains', 'dlog_gains_req', 'dlog_gains', 'info')}
                          for s in steps]),
                'state': _stack([s['state'] for s in steps]),
                **{k: _stack_gains([s[k] for s in steps]) for k in ('base_log_gains', 'dlog_gains_req', 'dlog_gains')},
                'info': [s['info'] for s in steps],
            } if steps else {},
            ticks=_stack(self._ticks),
            success=self._end == 'network', end=self._end, duration=self._duration if reached else 0.,
            aborted=aborted, phases=self.phase_log,
            path=None if self.env.dataset_path is None else str(self.env.epi_path),
            timing=dict(self._timing) if reached else {})
        if self.result.path is not None:
            with open(self.env.epi_path / 'insert.pkl', 'wb') as f:
                pickle.dump(self.result, f)
        return self.result

    def shutdown(self):
        if self.armed:
            self.end_episode()
        for worker, _ in self.policies.values():
            worker.close()


class ResidualEnv:
    """
    The split pipeline with a correction on insert, one episode per run(); see the module
    docstring. Construction spawns the policy workers and homes the robot (build your own
    models first: until the first run() the robot is servoing in this process).
    """

    def __init__(self, insert_ckpt, pickup_ckpt=DEFAULT_PICKUP_CKPT, log_dir=None, insert_timeout=None,
                 wait_circle=True, max_dpos=0.01, max_drot=0.1, max_dlog_gain=0.5,
                 max_dpos_rate=0.05, max_drot_rate=0.5, insert_done=False, control_freq=100.0,
                 weight_decay=0.5, taper=0.15, lpf_hz=3.0, done_threshold=0.5, device='cuda'):
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
        self._runner = _ResidualSplits(
            pickup_ckpt, insert_ckpt, insert_timeout=insert_timeout,
            limits=dict(max_dpos=max_dpos, max_drot=max_drot, max_dlog_gain=max_dlog_gain,
                        max_dpos_rate=max_dpos_rate, max_drot_rate=max_drot_rate),
            insert_done=insert_done, log_dir=log_dir, control_freq=control_freq, weight_decay=weight_decay,
            taper=taper, lpf_hz=lpf_hz, done_threshold=done_threshold, device=device)
        self.wait_circle = wait_circle
        self._fresh = True              # the constructor armed the first episode

    def register(self, correction, period=0.05, images=False):
        """
        Correct insert from the next run() on: correction(step: InsertStep) -> Correction
        (None = no correction), called on the control thread every `period` s of insert
        (None = every control tick), so it must be quick (~1 ms: numpy, or a small net on
        the CPU). images: pass camera frames in InsertStep / Episode (~1 MB each).
        register(None) runs the insert policy alone (still recorded, with zero corrections).
        """
        r = self._runner
        r.correction, r.period, r.images = correction, period, images

    def run(self) -> Episode:
        """One episode (waiting for Circle first, with wait_circle). Returns once the plug is
        released (or the episode aborted) and the robot has stopped."""
        r = self._runner
        if not self._fresh:
            r.rearm()
        self._fresh = False
        r.wait_circle = self.wait_circle
        try:
            r.run()                     # RobotExecution.run: ends in r.close() = r.end_episode()
        finally:
            r.end_episode()             # no-op unless run() raised
        if r.error is not None:
            raise r.error
        return r.result

    @property
    def limits(self):
        return dict(self._runner.limits)

    def close(self):
        """Stop the policy workers (and the robot, if run() was never called)."""
        self._runner.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ============================================================
    # Command line, for scripts that build a ResidualEnv
    # ============================================================
    @staticmethod
    def add_args(parser):
        parser.add_argument('--insert_ckpt', type=str, required=True, help='insert policy checkpoint')
        parser.add_argument('--pickup_ckpt', type=str, default=DEFAULT_PICKUP_CKPT, help='pickup policy checkpoint')
        parser.add_argument('--log_dir', type=str, default=None,
                            help='where to save robot log data (None disables logging)')
        parser.add_argument('--insert_timeout', type=float, default=None,
                            help='end insert (unplug, a failed episode) after this many seconds')
        parser.add_argument('--wait_circle', action=argparse.BooleanOptionalAction, default=False,
                            help='wait for Circle before each episode')
        parser.add_argument('--max_dpos', type=float, default=0.01, help='per-axis bound on the pose offset (m)')
        parser.add_argument('--max_drot', type=float, default=0.1, help='bound on the rotation offset (rad)')
        parser.add_argument('--max_dlog_gain', type=float, default=0.5, help='bound on each log10 gain offset')
        parser.add_argument('--max_dpos_rate', type=float, default=0.05, help='slew limit on the pose offset (m/s)')
        parser.add_argument('--max_drot_rate', type=float, default=0.5, help='slew limit on the rotation offset (rad/s)')
        parser.add_argument('--insert_done', action='store_true', default=False,
                            help='also end insert on the insert policy\'s done score, not only on the network')
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--control_freq', '--hz', type=float, default=100.0,
                            help='command frequency (Hz) of the real-time loop')
        parser.add_argument('--weight_decay', type=float, default=0.5,
                            help='recency-weighting rate (1/s) for ensembling overlapping chunks')
        parser.add_argument('--taper', type=float, default=0.15,
                            help='seconds over which a chunk\'s weight ramps in and out (0 = off)')
        parser.add_argument('--lpf', type=float, default=3.0,
                            help='cutoff (Hz) of the low-pass on the commanded pose (0 = off)')
        parser.add_argument('--done_threshold', type=float, default=0.5,
                            help='done score that ends pickup (and insert, with --insert_done)')

    @classmethod
    def from_args(cls, args, **overrides):
        """A ResidualEnv from add_args's arguments; overrides replace any of them."""
        kw = {k: getattr(args, k) for k in (
            'insert_ckpt', 'pickup_ckpt', 'log_dir', 'insert_timeout', 'wait_circle',
            'max_dpos', 'max_drot', 'max_dlog_gain', 'max_dpos_rate', 'max_drot_rate', 'insert_done',
            'device', 'control_freq', 'weight_decay', 'taper', 'done_threshold')}
        kw['lpf_hz'] = args.lpf
        return cls(**{**kw, **overrides})


if __name__ == '__main__':
    # Baseline: the insert policy alone (no correction registered), --episodes times
    parser = argparse.ArgumentParser(description='Split evaluation, one ResidualEnv episode at a time.')
    ResidualEnv.add_args(parser)
    parser.add_argument('--episodes', type=int, default=1)
    args = parser.parse_args()

    results = []
    with ResidualEnv.from_args(args) as env:
        for _ in range(args.episodes):
            ep = env.run()
            results.append(ep.success)
            print(f'\nepisode {ep.index}: {"plugged in" if ep.success else "failed"} ({ep.end})')
    print(f'{sum(results)}/{len(results)} plugged in')
