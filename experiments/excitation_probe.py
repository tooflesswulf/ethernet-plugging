"""
Excitation probe: teleoperate to a spot, then let the robot wiggle around it at randomized
impedance gains, to log how force and motion respond to K and D (the demos only ever
used K 300/3000 on z and D 1000/20). Data for learning a gain module / contact model.

    python experiments/excitation_probe.py --path ../data/excitation-probe

Teleop is teleoperation.py's (sticks, Circle gripper, Triangle z-force, Cross stiff z,
DpadDown move to port, DpadUp unplug), without the plug-in autodetect unless
--autounplug, so a probe can run with the plug seated. Then:

    DpadRight   start a probe session at the current target (where the spring pulls now)
    DpadLeft    stop the session: back to the anchor at the fdcc.toml gains, teleop resumes
    Square      exit (as always)

A session is --trials trials. Each trial, around the anchor and on TCP axes:
    settle   new gains (K log-uniform per axis, D critical by default), hold the anchor
    ramp in  min-jerk to a random static offset (bias: a preload when in contact)
    excite   random-phase multisine on 1, 2 or all active axes, 1/f amplitudes (flat
             velocity spectrum), tapered, scaled to the position and velocity limits
    hold     the bias, still (steady-state force vs offset)
    ramp out min-jerk back to the anchor
The first trial of a session is the fdcc.toml gains (the demos' controller), unless
--no-nominal. Amplitudes shrink with stiffness so |K * offset| stays within --budget:
in contact the wiggle explores force, not geometry.

Feedforward fade is OFF during sessions (imp.p.ff_release = 0, restored after): with it
the target's motion is fed forward only until the contact resists, a nonlinear gate on
top of the gains under test. Off, every trial is M dV + D (V - V_t) = F + K xi, about the
moving target. The leash (fdcc.toml [teleop] leash_N) still caps the spring push, and a
trial aborts above --abort_force / --abort_torque (target snapped to the arm, gains
restored, then back to the anchor and on to the next trial).

Logged: everything rawdata.h5 has (control/K, D, target, hold, xi, F_c, ...), plus
probe.json in the episode directory: per session the anchor and config, per trial the
gains asked for and the gains imp actually had, the offsets' parameters and seed, and
the env-clock times of each phase. Gains are on TOOL axes and may be anisotropic, which
scripts/rawdata_to_dataset.py rejects (it wants base-frame diagonals): read these
episodes with their own loader. A gain change keeps the held force (fdcc.py step 2a), so
the effective target is control/target moved by control/hold.
"""
from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse
import json
import time
import os

from teleoperation import Teleoperation
from robot.robot_execution import RobotExecution
from robot.interrupt_sequence import min_jerk
from robot.util import URPose, interpolate

FS = 100.0                  # control_freq of Teleoperation: one offset sample per tick
AXES = ('x', 'y', 'z', 'rx', 'ry', 'rz')


def compose(anchor, off):
    """anchor moved by off = [dp; drotvec] on the anchor's (TCP) axes."""
    a = np.asarray(anchor, float)
    Ra = R.from_rotvec(a[3:])
    return URPose(*np.r_[a[:3] + Ra.apply(off[:3]), (Ra * R.from_rotvec(off[3:])).as_rotvec()])


def multisine(rng, n, f_lo, f_hi, n_freq):
    """Unit-free random-phase multisine with 1/f amplitudes (flat velocity spectrum)."""
    t = np.arange(n) / FS
    f = np.geomspace(f_lo, f_hi, n_freq) * rng.uniform(0.92, 1.08, n_freq)   # no common harmonics
    ph = rng.uniform(0, 2 * np.pi, n_freq)
    return (np.sin(2 * np.pi * f[:, None] * t + ph[:, None]) / f[:, None]).sum(0), f


def taper(n, ramp_s):
    w = np.ones(n)
    m = min(int(ramp_s * FS), n // 2)
    if m > 0:
        r = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
        w[:m], w[-m:] = r, r[::-1]
    return w


def min_jerk_path(a, b, n):
    s = np.array([min_jerk(i / max(n - 1, 1)) for i in range(n)])
    return a[None] + s[:, None] * (b - a)[None]


class Trial:
    """One trial's gains and its whole offset trajectory, one row per control tick."""

    def __init__(self, idx, rng, cfg, M, nominal=None):
        self.idx = idx
        self.seed = int(rng.integers(2 ** 31))
        r = np.random.default_rng(self.seed)
        lin, rot = slice(0, 3), slice(3, 6)
        if nominal is not None:
            self.K, self.D, self.zeta = np.asarray(nominal[0], float), np.asarray(nominal[1], float), None
        else:
            K = np.empty(6)
            iso = r.random() < cfg.iso_frac
            for sl, (lo, hi) in ((lin, cfg.k_lin), (rot, cfg.k_rot)):
                k = np.exp(r.uniform(np.log(lo), np.log(hi), 1 if iso else 3))
                K[sl] = k
            self.zeta = float(r.choice(cfg.zeta))
            self.K, self.D = K, 2 * self.zeta * np.sqrt(np.asarray(M, float) * K)

        # which axes move: one, two, or all of --axes
        allowed = [AXES.index(a) for a in cfg.axes]
        n_act = int(r.choice([1, 2, len(allowed)], p=[0.35, 0.3, 0.35]))
        self.active = sorted(r.choice(allowed, min(n_act, len(allowed)), replace=False).tolist())

        # per-axis limits: geometric, and |K * offset| <= budget
        amp_max = np.r_[[cfg.amp_lin] * 3, [cfg.amp_rot] * 3]
        budget = np.r_[[cfg.budget[0]] * 3, [cfg.budget[1]] * 3]
        vmax = np.r_[[cfg.vmax[0]] * 3, [cfg.vmax[1]] * 3]
        amp = np.minimum(amp_max, budget / self.K)
        bias_max = np.minimum(np.r_[[cfg.bias_lin] * 3, [cfg.bias_rot] * 3], 0.5 * amp)
        self.bias = np.zeros(6)
        self.amp = np.zeros(6)
        n_ex = int(cfg.excite_s * FS)
        sig = np.zeros((n_ex, 6))
        self.freqs = {}
        for i in self.active:
            self.bias[i] = r.uniform(-1, 1) * bias_max[i]
            x, f = multisine(r, n_ex, cfg.f_band[0], cfg.f_band[1], cfg.n_freq)
            x = x * taper(n_ex, cfg.taper_s)
            room = amp[i] - abs(self.bias[i])
            s = min(room / np.max(np.abs(x)), vmax[i] / np.max(np.abs(np.gradient(x) * FS)))
            sig[:, i] = s * x
            self.amp[i] = float(np.max(np.abs(sig[:, i])))
            self.freqs[AXES[i]] = f.round(3).tolist()

        n = lambda s: max(int(s * FS), 1)
        zero = np.zeros(6)
        parts = [('settle', np.zeros((n(cfg.settle_s), 6))),
                 ('ramp_in', min_jerk_path(zero, self.bias, n(cfg.ramp_s))),
                 ('excite', self.bias[None] + sig),
                 ('hold', np.repeat(self.bias[None], n(cfg.hold_s), 0)),
                 ('ramp_out', min_jerk_path(self.bias, zero, n(cfg.ramp_s))),
                 ('rest', np.zeros((n(cfg.rest_s), 6)))]
        self.offsets = np.concatenate([p for _, p in parts])
        self.phase_start = dict(zip([name for name, _ in parts], np.cumsum([0] + [len(p) for _, p in parts[:-1]])))
        self.times = {}             # env-clock time each phase started (filled while running)
        self.K_applied = None
        self.aborted = None

    def record(self):
        return dict(idx=self.idx, seed=self.seed, nominal=self.zeta is None, zeta=self.zeta,
                    K=self.K.tolist(), D=self.D.tolist(),
                    K_applied=None if self.K_applied is None else self.K_applied.tolist(),
                    active=[AXES[i] for i in self.active], amp=self.amp.tolist(), bias=self.bias.tolist(),
                    freqs_hz=self.freqs, phase_times=self.times, aborted=self.aborted)


class ProbeSession:
    """Plays the trials around an anchor, one target pose per tick."""

    def __init__(self, rexec, anchor, cfg, rng):
        self.rexec, self.env, self.cfg = rexec, rexec.env, cfg
        self.anchor = np.asarray(anchor, float)
        p = self.env.imp.p
        self.trials = []
        for i in range(cfg.trials):
            nominal = (p.K, p.D) if (i == 0 and cfg.nominal) else None
            self.trials.append(Trial(i, rng, cfg, p.M, nominal))
        self.k = 0                          # trial index
        self.state = 'prep'                 # prep -> trial <-> return -> done
        self.t_state = time.perf_counter()
        self.ret_from = None
        self.stop_reason = None
        self.done = False
        self.t_start = self.clock()
        self.last_pose = URPose(*self.anchor)

    def clock(self):
        return time.time() - self.env.t0    # rawdata.h5's clock

    def request_stop(self, reason):
        if self.stop_reason is None:
            self.stop_reason = reason
            print(f'\nprobe: stopping ({reason})')
            self._abort_to_anchor(reason)

    def _abort_to_anchor(self, why):
        """Spring off the contact (target to the arm), toml gains, then min-jerk to the anchor."""
        if self.state == 'trial':
            tr = self.trials[self.k]
            tr.aborted, tr.times['abort'] = why, self.clock()
            print(f'\nprobe: trial {self.k} aborted: {why}')
        self.env.reanchor()
        self.env.restore_gains()
        self.ret_from = np.asarray(self.env.get_obs()['state']['actual_pose'], float)
        self.state, self.t_state = 'return', time.perf_counter()

    def _start_trial(self):
        tr = self.trials[self.k]
        self.env.set_gains(K=tr.K, D=tr.D, frame='tool', ramp_s=self.cfg.gain_ramp_s)
        self.state, self.t_state = 'trial', time.perf_counter()
        self._phase = None
        Ks = ' '.join(f'{k:.0f}' for k in tr.K)
        print(f'\nprobe: trial {tr.idx + 1}/{len(self.trials)}  K [{Ks}]  '
              f'zeta {tr.zeta if tr.zeta is not None else "toml"}  axes {[AXES[i] for i in tr.active]}')

    def _next(self):
        self.rexec.save_probe_log()
        self.k += 1
        if self.k >= len(self.trials) or self.stop_reason is not None:
            self.done = True
        else:
            self._start_trial()

    def tick(self):
        now = time.perf_counter()
        el = now - self.t_state
        st = self.env.get_obs()['state']
        if self.env.fdcc_halt is not None and self.stop_reason is None:
            self.request_stop(f'FDCC halt: {self.env.fdcc_halt}')

        if self.state == 'prep':            # toml gains settle (stiff z / z-force just left)
            pose = URPose(*self.anchor)
            if el >= self.cfg.prep_s:
                self._start_trial()

        elif self.state == 'trial':
            tr = self.trials[self.k]
            W = np.asarray(st['filtered_force'], float)
            f, tq = np.linalg.norm(W[:3]), np.linalg.norm(W[3:])
            if f > self.cfg.abort_force or tq > self.cfg.abort_torque:
                self._abort_to_anchor(f'|F| {f:.1f} N |tau| {tq:.2f} Nm')
                return self.tick()
            i = int(el * FS)
            for name, s in tr.phase_start.items():          # phase start times, on the env clock
                if i >= s and name not in tr.times:
                    tr.times[name] = self.clock()
                    if name == 'excite':
                        tr.K_applied = np.asarray(self.env.imp.K, float).copy()
                        if not np.allclose(tr.K_applied, tr.K, rtol=0.02):
                            print(f'\nprobe: WARNING trial {tr.idx} K is {tr.K_applied.round(0)}, asked {tr.K.round(0)}')
            if i >= len(tr.offsets):
                tr.times['end'] = self.clock()
                pose = URPose(*self.anchor)
                self._next()
            else:
                pose = compose(self.anchor, tr.offsets[i])

        elif self.state == 'return':
            frac = min_jerk(el / self.cfg.return_s)
            pose = interpolate(URPose(*self.ret_from), URPose(*self.anchor), frac)
            if el >= self.cfg.return_s + self.cfg.rest_s:
                if self.stop_reason is not None:
                    self.done = True
                else:
                    self._next()
        else:
            pose = URPose(*self.anchor)

        self.last_pose = pose
        return pose

    def record(self):
        return dict(anchor=self.anchor.tolist(), t_start=self.t_start, t_end=self.clock() if self.done else None,
                    stop_reason=self.stop_reason, config=vars(self.cfg),
                    trials=[t.record() for t in self.trials[:self.k + 1]])


class ExcitationProbe(Teleoperation):
    def __init__(self, args):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.session = None
        self.sessions = []                  # records of finished sessions
        self._last_dpad = (False, False)
        self._ff_release = None
        super().__init__(args)

    def args2metadata(self, args):
        meta = super().args2metadata(args)
        meta['probe'] = json.dumps(vars(args))
        return meta

    def pre_run(self):
        print('============ EXCITATION PROBE ============')
        print('Teleop as teleoperation.py. DpadRight: probe here. DpadLeft: stop the probe.')
        print(f'{self.args.trials} trials per session, about {self.trial_seconds():.0f} s each.')
        print('==========================================')

    def trial_seconds(self):
        a = self.args
        return a.settle_s + 2 * a.ramp_s + a.excite_s + a.hold_s + a.rest_s

    def get_action(self):
        st = self.iface.dualsense.state
        right, left = st.DpadRight, st.DpadLeft
        start, stop = right and not self._last_dpad[0], left and not self._last_dpad[1]
        self._last_dpad = (right, left)

        if self.session is not None:
            if stop:
                self.session.request_stop('operator')
            pose = self.session.tick()
            if self.session.done:
                self._end_session()
            return pose, self.iface.gripper_state, False, 0.

        if start:
            self._start_session()
            return self.get_action()

        # teleop: Teleoperation.get_action, without the plug-in autodetect unless asked
        if self.iface.act.get('toggle_stiff_z'):
            self.toggle_stiff_z()
        if self.args.autounplug and self.last_obs and self.last_obs['network_status']:
            print('Autodetected plugin!')
            return self.unplug_and_release()
        if st.DpadDown:
            return self.move_to_port()
        if st.DpadUp:
            return self.unplug_and_release()
        return RobotExecution.get_action(self)

    def _start_session(self):
        if self.iface.adaptive_mode:        # leave z-force mode where the spring pulls now
            self.iface.adaptive_mode = False
            self.iface.deactivate_adaptive_mode()
        if self.stiff_z:
            self.toggle_stiff_z()           # back to the toml gains; prep_s lets the ramp finish
        else:
            self.env.restore_gains()
        anchor = self.env.restart_at_target()
        imp = self.env.imp
        self._ff_release = np.array(imp.p.ff_release, float)
        imp.p.ff_release = np.zeros(2)      # fade off: the gate's check is release > 0
        self.env.set_feedforward(True)
        self.session = ProbeSession(self, anchor, self.args, self.rng)
        print(f'\nprobe: session {len(self.sessions) + 1} at {URPose(*anchor)}')

    def _end_session(self):
        s = self.session
        self.env.restore_gains()
        if self._ff_release is not None:
            self.env.imp.p.ff_release = self._ff_release
        self.iface.targ_pose = np.array(s.last_pose, float)   # teleop resumes where the probe left it
        self.sessions.append(s.record())
        self.session = None
        self.save_probe_log()
        n_ok = sum(t['aborted'] is None for t in self.sessions[-1]['trials'])
        print(f'\nprobe: session done, {n_ok}/{len(self.sessions[-1]["trials"])} trials clean. Teleop.')

    def save_probe_log(self):
        path = getattr(self.env, 'epi_path', None)
        if path is None:
            return
        recs = self.sessions + ([self.session.record()] if self.session is not None else [])
        tmp = path / 'probe.json.tmp'
        with open(tmp, 'w') as fh:
            json.dump({'sessions': recs}, fh, indent=1)
        os.replace(tmp, path / 'probe.json')

    def runtime_info(self):
        if self.session is None:
            return super().runtime_info()

    def close(self):
        if self.session is not None:        # Square mid-session: log it, put the controller back
            self.session.stop_reason = self.session.stop_reason or 'exit'
            self.session.done = True
            self._end_session()
        super().close()


def main():
    parser = argparse.ArgumentParser(description='Randomized impedance excitation around teleoperated spots.')
    parser.add_argument('--path', type=str, default='../data/excitation-probe', help='Base dataset directory')
    parser.add_argument('--id', type=int, default=None, help='Episode ID (default: next available)')
    parser.add_argument('-d', '--debug', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--autounplug', action=argparse.BooleanOptionalAction, default=False,
                        help='unplug when the network comes up, as teleoperation.py does')
    g = parser.add_argument_group('trials')
    g.add_argument('--trials', type=int, default=8, help='trials per session')
    g.add_argument('--nominal', action=argparse.BooleanOptionalAction, default=True,
                   help='first trial at the fdcc.toml gains (the demos\' controller)')
    g.add_argument('--k_lin', type=float, nargs=2, default=[100., 3000.], help='K range [N/m], log-uniform')
    g.add_argument('--k_rot', type=float, nargs=2, default=[3., 100.], help='K range [Nm/rad], log-uniform')
    g.add_argument('--iso_frac', type=float, default=0.3, help='fraction of trials with K equal on each half')
    g.add_argument('--zeta', type=float, nargs='+', default=[1.0],
                   help='damping ratios to draw from (D = 2 zeta sqrt(M K)); 1 = critical in free space')
    g.add_argument('--axes', nargs='+', default=list(AXES), choices=AXES, help='TCP axes allowed to move')
    g = parser.add_argument_group('excitation')
    g.add_argument('--amp_lin', type=float, default=0.02, help='max offset [m]')
    g.add_argument('--amp_rot', type=float, default=0.1, help='max offset [rad]')
    g.add_argument('--bias_lin', type=float, default=0.005, help='max static offset [m]')
    g.add_argument('--bias_rot', type=float, default=0.03, help='max static offset [rad]')
    g.add_argument('--budget', type=float, nargs=2, default=[20., 1.5],
                   help='[N, Nm]: amplitudes shrink so |K * offset| stays under this')
    g.add_argument('--vmax', type=float, nargs=2, default=[0.03, 0.3], help='target speed limit [m/s, rad/s]')
    g.add_argument('--f_band', type=float, nargs=2, default=[0.1, 2.0], help='multisine band [Hz]')
    g.add_argument('--n_freq', type=int, default=10)
    g.add_argument('--excite_s', type=float, default=8.0)
    g.add_argument('--taper_s', type=float, default=0.5)
    g = parser.add_argument_group('timing and safety')
    g.add_argument('--prep_s', type=float, default=0.6, help='hold at the anchor before the first trial')
    g.add_argument('--settle_s', type=float, default=0.5, help='hold after each gain change')
    g.add_argument('--gain_ramp_s', type=float, default=0.3)
    g.add_argument('--ramp_s', type=float, default=0.5, help='min-jerk to and from the bias')
    g.add_argument('--hold_s', type=float, default=1.0, help='still at the bias after the wiggle')
    g.add_argument('--rest_s', type=float, default=0.3)
    g.add_argument('--return_s', type=float, default=1.5, help='min-jerk back to the anchor after an abort')
    g.add_argument('--abort_force', type=float, default=35.0, help='[N], filtered')
    g.add_argument('--abort_torque', type=float, default=3.0, help='[Nm], filtered, about the flange')
    args = parser.parse_args()

    if args.id is None and not args.debug:
        indices = [int(d.removeprefix('episode')) for d in os.listdir(args.path)
                   if d.startswith('episode') and d.removeprefix('episode').isdigit()] if os.path.exists(args.path) else []
        args.id = max(indices, default=0) + 1
        print(f'Auto-selected episode ID: {args.id}')
    ExcitationProbe(args).run()


if __name__ == '__main__':
    main()
