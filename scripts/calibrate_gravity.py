"""Interactive gravity-residual calibration for getActualTCPForce().

Teleoperate the robot in free space (nothing grasped, no contact, adaptive
mode off). Press Dpad-Right to run a small-angle probe (~7 deg wiggles about
the tool x/y/z axes) around the current orientation; control returns to the
joystick when it finishes. Collect probes at several orientations spanning
the working range — include the large rotations where the drift is worst —
then press Dpad-Left to fit, save, and apply the model live so the force
readout can be sanity-checked immediately.

Model (see Env.compensate_gravity):

    f_tool = B u + (I - R^T R0) b,  tau_tool = q x u + (I - R^T R0) tb
    with u = (R^T - R0^T) ghat

The sensor's slow time drift is absorbed by a linear-in-time nuisance term,
anchored by the reading being zero right after zeroFtSensor(); B and b are
identified from the probe clusters at the different orientations.
"""
from scipy.spatial.transform import Rotation as R
import numpy as np

# same mechanism as robot_utils.interrupt(), imported directly to skip the
# torch/policy imports that robot_utils drags in
from agent.utils.interrupt_sequence import InterruptSequence, Step
import robot_execution
from util import URPose

GHAT = np.array([0., 0., 1.])  # gravity direction in the base frame (unit)
PROBE_ANGLE = 0.12             # rad, size of the small-angle probe
PROBE_AXES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
ROT_SPEED = 0.1                # rad/s, rotation speed between probe orientations
SETTLE_TIME = 0.5              # s, dwell before sampling
SAMPLE_TIME = 0.8              # s, force averaging window
RIDGE_B = 0.1                  # penalty on the anisotropic part of B


def skew(u):
    return np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])


def fit_residual(clusters, rot0):
    """Fit (B, b, q, tb) from probe clusters of (rotvecs [n,3], wrenches [n,6], times [n]).

    Wrenches must be raw (uncompensated) getActualTCPForce readings in the base
    frame; rot0 is the TCP orientation at zeroFtSensor time. Sensor time drift
    is modeled as linear in time (zero at zeroing time), fitted jointly and
    discarded.
    """
    R0 = rot0.as_matrix()
    u0 = R0.T @ GHAT
    us, drels, fs, taus, ts = [], [], [], [], []
    for rvs, ws, t in clusters:
        RTs = R.from_rotvec(rvs).inv().as_matrix()   # [n,3,3]
        us.append(RTs @ GHAT - u0)
        drels.append(np.eye(3) - RTs @ R0)
        fs.append(np.einsum('nij,nj->ni', RTs, ws[:, :3]))
        taus.append(np.einsum('nij,nj->ni', RTs, ws[:, 3:]))
        ts.append(t)
    us, drels, fs, taus, ts = (np.concatenate(a) for a in (us, drels, fs, taus, ts))

    # f_tool_i = (alpha I + B_dev) u_i + drel_i b + t_i d  ->  joint least-squares.
    # A handful of probe orientations can't fully separate an anisotropic B from
    # the bias term, so the anisotropic part B_dev is ridge-penalized: the
    # isotropic gravity gain alpha, bias b, and drift d stay unregularized and
    # soak up the real signal, while B_dev only keeps what the data insists on.
    A = np.vstack([np.hstack([u[:, None], np.kron(np.eye(3), u), d, t * np.eye(3)])
                   for u, d, t in zip(us, drels, ts)])
    ridge = np.zeros((9, 16))
    ridge[:, 1:10] = RIDGE_B * np.eye(9)
    x, *_ = np.linalg.lstsq(np.vstack([A, ridge]), np.r_[fs.ravel(), np.zeros(9)], rcond=None)
    B, b, d = x[0] * np.eye(3) + x[1:10].reshape(3, 3), x[10:13], x[13:]

    # tau_tool_i = q x u_i + drel_i tb + t_i dt.  q x u = -[u]x q is linear in q
    # and absorbs payload CoG config error (q = m g dcog) as well as the
    # mass-error lever, independent of whether the force term B is large.
    M = np.vstack([np.hstack([-skew(u), dr, t * np.eye(3)])
                   for u, dr, t in zip(us, drels, ts)])
    y, *_ = np.linalg.lstsq(M, taus.ravel(), rcond=None)
    q, tb = y[:3], y[3:6]

    f_pred = us @ B.T + np.einsum('nij,j->ni', drels, b) + np.outer(ts, d)
    rms_before = np.sqrt(np.mean(fs ** 2))
    rms_after = np.sqrt(np.mean((fs - f_pred) ** 2))
    print(f'\nGravity-shaped residual ~{np.trace(B) / 3 / 9.81 * 1000:.0f} g payload error equivalent, '
          f'bias-shaped residual |b| = {np.linalg.norm(b):.3f} N, '
          f'sensor drift {np.linalg.norm(d) * 60:.3f} N/min')
    print(f'Torque lever |q| = {np.linalg.norm(q):.4f} Nm '
          f'(~{np.linalg.norm(q) / (1.7 * 9.81) * 1000:.1f} mm CoG error at the 1.7 kg payload)')
    print(f'Force artifact RMS across probes: {rms_before:.3f} N -> {rms_after:.3f} N after fit '
          f'(worst sample {np.abs(fs - f_pred).max():.3f} N)')
    print('Fitted B [N]:\n', np.array_str(B, precision=3, suppress_small=True))
    return B, b, q, tb


class SampleStep(Step):
    """Interrupt step: hold the current pose, dwell SETTLE_TIME, then average
    the TCP orientation and raw wrench over a SAMPLE_TIME window and append
    (rotvec, wrench, time) to `out`."""

    def __init__(self, rexec, out):
        super().__init__(rexec)
        self.out = out
        self.i0 = None

    def tick(self, t):
        if self.i0 is None and t >= SETTLE_TIME:
            self.i0 = len(self.env.robot_obs)
        if self.i0 is not None and t >= SETTLE_TIME + SAMPLE_TIME \
                and len(self.env.robot_obs) > self.i0:
            obs = self.env.robot_obs[self.i0:]
            rvs = np.array([o.actual_pose for o in obs])[:, 3:]
            ws = np.array([o.actual_force for o in obs])
            tm = np.mean([o.time for o in obs])  # since start(), ~ time since zeroFtSensor
            self.out.append((rvs.mean(axis=0), ws.mean(axis=0), tm))
            return None
        return URPose(*self.env.des_pose), self.env.des_gripper_state, False, 0.


class GravityCalibration(robot_execution.RobotExecution):
    def __init__(self):
        self.clusters = []  # list of (rotvecs [n,3], wrenches [n,6]) per probe
        # start True so a button held during startup doesn't trigger
        self._prev_probe_btn = True
        self._prev_fit_btn = True
        super().__init__(path=None, control_freq=100, show_image=True)

    def pre_reset(self):
        print('=' * 64)
        print('Gravity residual calibration')
        print('  Teleoperate in free space: nothing grasped, no contact.')
        print('  Dpad-Right: probe around the current orientation')
        print('  Dpad-Left:  fit + save + apply live (needs >= 2 probes)')
        print('  Dpad-Down:  return to home pose')
        print('=' * 64)

    def post_reset(self):
        self._disable_compensation()

    def _disable_compensation(self):
        # probes must record raw forces, not ones already compensated by a
        # previously loaded (or just-fitted) model
        self.env.grav_residual_B = np.zeros((3, 3))
        self.env.grav_residual_b = np.zeros(3)
        self.env.grav_residual_q = np.zeros(3)
        self.env.grav_residual_tb = np.zeros(3)

    def get_action(self):
        state = self.iface.dualsense.state
        trigger_probe = state.DpadRight and not self._prev_probe_btn
        trigger_fit = state.DpadLeft and not self._prev_fit_btn
        go_home = state.DpadDown
        self._prev_probe_btn = state.DpadRight
        self._prev_fit_btn = state.DpadLeft

        if go_home:
            seq = InterruptSequence.current(self)
            seq.move_to(self.home_pose)
            return self.get_action()
        if trigger_fit:
            self.fit_and_save()
        if trigger_probe:
            self.run_probe()
            return self.get_action()  # sequence installed: plays its first tick
        return super().get_action()

    # ------------------------------------------------------------
    # probing
    # ------------------------------------------------------------
    def run_probe(self):
        """Queue the probe as an interrupt sequence (one action per control
        tick), so the run loop — camera display, logging, exit button — keeps
        ticking instead of blocking inside get_action(). While the sequence is
        active it shadows get_action, so joystick input and further button
        presses are ignored until it drains."""
        if np.any(self.env.grav_residual_B) or np.any(self.env.grav_residual_b):
            print('\nCompensation was live; disabling it for probing (refit when done).')
            self._disable_compensation()
        center = URPose(*self.env.des_pose)
        rot_c = R.from_rotvec([center.rx, center.ry, center.rz])
        probe_rots = [rot_c] + [rot_c * R.from_rotvec(np.array(ax) * PROBE_ANGLE) for ax in PROBE_AXES] + [rot_c]

        print(f'\nProbe {len(self.clusters) + 1}: wiggling around the current orientation, hands off...')
        samples = []
        seq = InterruptSequence.current(self)
        for rot in probe_rots:
            seq.move_to(URPose(center.x, center.y, center.z, *rot.as_rotvec()), rot_speed=ROT_SPEED)
            seq.add(SampleStep(self, samples))
        # returning to center ends the sequence, which re-syncs the joystick
        # target there before handing control back
        seq.move_to(center) \
           .then(lambda _: self._finish_probe(samples))

    def _finish_probe(self, samples):
        rvs, ws, ts = map(np.array, zip(*samples))
        self.clusters.append((rvs, ws, ts))
        print(f'Probe {len(self.clusters)} done. Move to another orientation and probe again.')

    # ------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------
    def fit_and_save(self):
        if len(self.clusters) < 2:
            print(f'\nNeed at least 2 probes at different orientations ({len(self.clusters)} so far); '
                  '4+ spanning the working range is better.')
            return
        B, b, q, tb = fit_residual(self.clusters, self.env._ft_zero_rot)
        np.savez(self.env.grav_cal_path,
                 residual_B=B, residual_b=b, residual_q=q, residual_tb=tb,
                 ft_zero_rotvec=self.env._ft_zero_rot.as_rotvec(),
                 probe_rotvecs=np.concatenate([rv for rv, _, _ in self.clusters]),
                 probe_wrenches=np.concatenate([w for _, w, _ in self.clusters]),
                 probe_times=np.concatenate([t for _, _, t in self.clusters]))
        print(f'Saved calibration to {self.env.grav_cal_path}')

        # apply live for an immediate sanity check of the force readout
        self.env.grav_residual_B = B
        self.env.grav_residual_b = b
        self.env.grav_residual_q = q
        self.env.grav_residual_tb = tb
        print('Compensation is now live in this session; rotate around and watch the forces. '
              'More probes will disable it again until the next fit.')

    def runtime_info(self):
        force = self.last_obs['state']['filtered_force']
        live = 'live' if np.any(self.env.grav_residual_B) or np.any(self.env.grav_residual_b) else 'off'
        print(f'probes: {len(self.clusters)} | comp: {live} | '
              f'fx {force[0]:6.2f} fy {force[1]:6.2f} fz {force[2]:6.2f} '
              '| Dpad-Right: probe, Dpad-Left: fit+save', end='\r')


if __name__ == '__main__':
    GravityCalibration().run()
