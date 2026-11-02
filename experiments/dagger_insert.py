"""
DAgger-style data collection on insert: the insert policy runs as in zforce_regulator.py, and
the operator's DualSense corrects it, as teleoperation_rl.py does on the whole task.

    python experiments/dagger_insert.py --insert_ckpt <ckpt> --log_dir <dir>

During insert:
    sticks, L2 / R2   move the plug: left stick x/y, L2 up (L1 + L2 down), right stick
                      roll/pitch, R2 yaw (R1 + R2 the other way), as in teleoperation_rl.py.
                      A deflection is a velocity: the offset is velocity * horizon, which the
                      policy absorbs (as ZForceRegulator), so holding a stick keeps moving
    Cross             toggle base z stiff (STIFF_Z_K) to push the plug in; again: back
    DpadUp            give up on this insert (unplug, a failed episode)
    Square            abort the episode; at the Circle prompt, end the session
Triangle (z-force mode) is off: it would zero the z input. The gripper is held closed
through insert, whatever the policy commands.

Each episode directory gets, besides eval_residual's rawdata.h5 (whose control/target is the
corrected command), phases.json and insert.pkl, a dagger.npz: every control tick of insert
with the policy's command, the offset applied, the command executed and whether the
operator was correcting -- the labels to aggregate into the next insert dataset.
"""
from agent.eval.eval_residual import ResidualEnv, ResidualActor, Correction, apply_dpose
from experiments.zforce_regulator import ZForceRegulator, summarize
from scipy.spatial.transform import Rotation as R
from robot import fdcc
import numpy as np
import argparse
import json
import os

STIFF_Z_K = 3000.0      # [N/m] on base z while Cross is toggled on (as teleoperation.py)
# DualSenseInterface's right_delta at full deflection (DualSense.input2action: 0.005 * 125, 0.005 * 50)
FULL_STICK = np.r_[0.625, 0.625, 0.625, 0.25, 0.25, 0.25]


class JoystickCorrection(ResidualActor):
    def __init__(self, env, speed=0.02, rot_speed=0.15, horizon=0.4, stiff_k=STIFF_Z_K, ramp_s=0.1,
                 regulator=None):
        self.env = env
        self.speed = speed          # [m/s] at full deflection
        self.rot_speed = rot_speed  # [rad/s] at full deflection
        self.horizon = horizon      # [s] the time over which the policy absorbs an offset
        self.log_k = np.log10(stiff_k)
        self.ramp_s = ramp_s        # [s] stiff z blends in and out over this long
        self.regulator = regulator  # a ZForceRegulator to add, or None
        # base z K when the policy does not predict stiffness: fdcc.toml's
        self.base_log_k = np.log10(fdcc._six(fdcc.load_config()['admittance']['stiffness']))
        self.stiff_z = False
        self._w = 0.                # stiff z blend weight, ramps toward stiff_z
        self._cross = False
        self._t = None

    def act(self, step):
        iface = self.env.iface
        iface.enable_zadaptive = iface.adaptive_mode = False    # Triangle would zero the z input
        # base axes (flip_actions), [-1, 1] at full deflection
        delta = np.clip(np.asarray(iface.act['right_delta'], float) / FULL_STICK, -1., 1.)

        # Cross: polled, since decisions can fall between the ticks its rising edge is on
        cross = bool(iface.dualsense.state.Cross)
        if cross and not self._cross:
            self.stiff_z = not self.stiff_z
            print(f'\nstiff z {"ON" if self.stiff_z else "off"}')
        self._cross = cross
        dt = 0. if self._t is None else step.t - self._t
        self._t = step.t
        self._w = float(np.clip(self._w + (dt if self.stiff_z else -dt) / self.ramp_s, 0., 1.))

        # Rotation composed as DualSenseInterface.residual_action (global, ZYX)
        drx, dry, drz = delta[3:] * self.rot_speed * self.horizon
        dpose = np.r_[delta[:3] * self.speed * self.horizon,
                      R.from_euler('ZYX', [-drz, -dry, drx]).as_rotvec()]
        dlog_gains = {}
        if self._w > 0:
            k = step.base_log_gains['stiffness'] if 'stiffness' in step.base_log_gains else self.base_log_k
            dk = np.zeros(6)
            dk[2] = (3 * self._w ** 2 - 2 * self._w ** 3) * (self.log_k - k[2])    # smoothstep, as fdcc ramps
            dlog_gains['stiffness'] = dk
        info = {'human': bool(delta.any() or self._w > 0), 'stick': delta, 'stiff_z': self._w}

        if self.regulator is not None:
            c = self.regulator.act(step)
            dpose = dpose + c.dpose
            info.update(c.info)
        return Correction(dpose=dpose, dlog_gains=dlog_gains, info=info)


def human_ticks(ep):
    """Per control tick: whether the decision in effect was the operator's correction."""
    i = np.searchsorted(ep.steps['t'], ep.ticks['t'], side='right') - 1
    human = np.array([s['human'] for s in ep.steps['info']])
    return (i >= 0) & human[np.maximum(i, 0)]


def save_dagger(ep):
    """dagger.npz into the episode directory (see the module docstring)."""
    tk = ep.ticks
    t_insert = next(p['t_start'] for p in ep.phases if p['phase'] == 'insert')
    np.savez(os.path.join(ep.path, 'dagger.npz'),
             t=tk['t'],                         # s since insert started
             t_raw=t_insert + tk['t'],          # on the rawdata.h5 clock
             human=human_ticks(ep),
             base_pose=tk['base_pose'],         # the insert policy's command
             dpose=tk['dpose'],                 # the offset applied (slewed)
             command=np.array([apply_dpose(b, d) for b, d in zip(tk['base_pose'], tk['dpose'])]),
             target_pose=tk['target_pose'], actual_pose=tk['actual_pose'],
             filtered_force=tk['filtered_force'], network_status=tk['network_status'],
             step_t=ep.steps['t'],              # per decision
             step_human=np.array([s['human'] for s in ep.steps['info']]),
             stick=np.array([s['stick'] for s in ep.steps['info']]),
             stiff_z=np.array([s['stiff_z'] for s in ep.steps['info']]),
             dlog_stiffness=ep.steps['dlog_gains'].get('stiffness', np.zeros((len(ep.steps['t']), 6))),
             success=ep.success)


def main():
    parser = argparse.ArgumentParser(description='Insert with joystick corrections (DAgger), episode after episode.')
    ResidualEnv.add_args(parser)
    # Cross takes base z K from the policy's (~300 N/m) to STIFF_Z_K: 1.0 in log10
    parser.set_defaults(max_dlog_gain=1.0)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--speed', type=float, default=0.02, help='velocity at full stick (m/s)')
    parser.add_argument('--rot_speed', type=float, default=0.15, help='angular velocity at full stick (rad/s)')
    parser.add_argument('--horizon', type=float, default=0.4,
                        help='offset = velocity * horizon: the time the policy takes to absorb an offset (s)')
    parser.add_argument('--stiff_k', type=float, default=STIFF_Z_K, help='base z stiffness while Cross is on (N/m)')
    parser.add_argument('--zforce', action='store_true', default=True,
                        help='also run the z-force regulator of zforce_regulator.py under the corrections')
    parser.add_argument('--f_floor', type=float, default=5.0, help='z-force above which the regulator acts (N)')
    parser.add_argument('--gain', type=float, default=3e-3, help='regulator: upward velocity per N over f_floor (m/s/N)')
    parser.add_argument('--max_speed', type=float, default=0.015, help='regulator: cap on the upward velocity (m/s)')
    parser.add_argument('--period', type=float, default=0.02, help='decision period (s)')
    args = parser.parse_args()

    last_successful = False
    results = []
    # The gripper stays closed through insert: an OOD grasp could make the policy let go
    with ResidualEnv.from_args(args, insert_grip_closed=True) as env:
        for _ in range(args.episodes):
            regulator = ZForceRegulator(args.f_floor, args.gain, args.max_speed, args.horizon) if args.zforce else None
            env.register(JoystickCorrection(env, args.speed, args.rot_speed, args.horizon, args.stiff_k,
                                            regulator=regulator), period=args.period)
            env.wait_circle = not last_successful
            ep = env.run()
            last_successful = ep.success
            s = summarize(ep, args.f_floor)
            if ep.ticks:
                human = human_ticks(ep)
                dt = np.diff(ep.ticks['t'], append=ep.ticks['t'][-1])
                s.update(s_human=round(float(dt[human].sum()), 2), human_frac=round(float(human.mean()), 2))
                if ep.path is not None:
                    save_dagger(ep)
            results.append(s)
            print(f'\nepisode {s["episode"]}: {"plugged in" if ep.success else "failed"} ({ep.end})  '
                  + '  '.join(f'{k} {s[k]}' for k in ('s_human', 'human_frac', 'fz_max', 'dz_max_mm') if k in s))
            if args.log_dir is not None:
                with open(os.path.join(args.log_dir, 'dagger_insert.json'), 'w') as f:
                    json.dump({'args': vars(args), 'episodes': results}, f, indent=2, default=float)

    n = len(results)
    print(f'\n{sum(r["success"] for r in results)}/{n} plugged in '
          f'({sum(r["aborted"] for r in results)} aborted, '
          f'{sum(r.get("s_human", 0) > 0 for r in results)} corrected)')


if __name__ == '__main__':
    main()
