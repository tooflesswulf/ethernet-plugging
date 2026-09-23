from scipy.spatial.transform import Rotation as R, Slerp
from collections import namedtuple
from typing import Literal
import rtde_control
import rtde_receive
import numpy as np
import threading
import pathlib
import h5py
import time
import cv2
import os

from net_isup import is_network_up
from util import URPose, clamp, slerp, interpolate, episode_index, dict2hdf5
from camera import Camera
import fdcc
import wsg

# Gripper command states (des_gripper_state / gripper_state)
GRIP_OPEN = 0
GRIP_CLOSED = 1
GRIP_MOVING = -1


class RobotObs(namedtuple('RobotObs', ('time', 'actual_pose', 'actual_force', 'filtered_force'))):
    pass


class GripperObs(namedtuple('GripperObs', ('time', 'gripper_width', 'gripper_force'))):
    pass


class CameraObs(namedtuple('CameraObs', ('time', 'image'))):
    pass


class Command(namedtuple('Command', ('time', 'des_pose', 'des_gripper', 'adaptive_mode', 'des_zforce', 'controller_state'))):
    pass


class Env:
    """
    Minimal robot environment wrapper for:
        - Teleoperation
        - Robot policy evaluation
        - Data collection

    Components:
        - Single RGB camera
        - UR robot arm
        - WSG gripper
    """

    def __init__(
        self,
        robot_ip="192.168.0.100",  # could be 101 or 100 depending on your setup
        gripper_ip="192.168.0.20",
        network_iface='enx7cc2c6453f68',  # network interface to check for connectivity
        camera_crop_mode=1,  # crop on the right half of the image to focus on the workspace
        control_frequency=20,
        servo_frequency=500,
        gripper_query_frequency=250,
        max_position_step=(0.008, 0.008, 0.008),
        max_orientation_step=0.05,
        lookahead_time=0.1,
        servo_gain=500,
        obs_mode: Literal['latest', 'mean'] = 'latest',
        dataset_path=None,
        save_interval=0.1,
        save_eps=1e-3,
        gwidth=20,
        gforce=40,
        gspeed=50,
        gpullback=10,
        metadata={}
    ):
        # ============================================================
        # Internal states
        # ============================================================
        self.t0 = None
        self.open_width = gwidth + 2 * gpullback
        self.home_pose = URPose(-0.125, 0.545, 0.305, 2.44, 2.44, 0.653)
        self.gripper_state = GRIP_OPEN
        self.des_pose, self.des_gripper_state = self.home_pose, self.gripper_state
        self.des_zforce = 0.
        self.adaptive_mode = False
        self.last_step_t = time.perf_counter()
        self._zero_ft_request = False  # serviced by _control_loop; see request_zero_ft()

        # ============================================================
        # Control parameters
        # ============================================================
        self.g_force = gforce
        self.g_width = gwidth
        self.g_speed = gspeed
        self.g_pullback = gpullback

        # ============================================================
        # Camera
        # ============================================================
        self.camera_crop_mode = camera_crop_mode

        # ============================================================
        # Robot interfaces
        # ============================================================
        self.robot_ip = robot_ip
        self.gripper_ip = gripper_ip
        self.ctrl = rtde_control.RTDEControlInterface(robot_ip)
        self.recv = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.gripper = wsg.WSG(ip=gripper_ip)
        self.gripper_query_frequency = gripper_query_frequency
        self.network_iface = network_iface
        self.network_query_frequency = 2.0
        self.network_status = False

        # ============================================================
        # Servo parameters
        # ============================================================
        self.input_frequency = control_frequency
        self.servo_frequency = servo_frequency
        self.dt = 1.0 / servo_frequency
        self.max_position_step = np.array(max_position_step)
        self.max_orientation_step = max_orientation_step
        self.lookahead_time = lookahead_time
        self.servo_gain = servo_gain

        # ============================================================
        # FDCC admittance over speedL (fdcc.py, numbers in fdcc.toml)
        # ============================================================
        self.fdcc_cfg = fdcc.load_config()
        self.imp = fdcc.Impedance(fdcc.ImpedanceParams.from_config(self.fdcc_cfg),
                                  tcp_offset=self.ctrl.getTCPOffset())
        if abs(self.imp.p.dt - self.dt) > 1e-9:
            print(f'WARNING: fdcc.toml rate {1 / self.imp.p.dt:.0f} Hz != servo_frequency {servo_frequency} Hz')
        abort = np.array(self.fdcc_cfg['limits']['abort_wrench'], float)
        self.abort_wrench = np.where(abort > 0, abort, np.inf)          # 0 or inf = no limit
        self.leash_N = np.array(self.fdcc_cfg['teleop']['leash_N'], float)  # [N, Nm], see _leash()
        self._leashed = None             # the leashed target fed to fdcc (None = restart from the arm)
        self._leash_held = False
        rt = self.fdcc_cfg['rtde']
        self.speedl_time = rt['speedl_time_cycles'] * self.imp.p.dt       # ONE cycle: 8 ms buzzed at 125 Hz
        self.watchdog_hz = rt['watchdog_hz']
        self.stop_decel = rt['stop_decel']
        self.rest_speed = np.array(rt['rest_speed'], float)
        self.ramp_timeout = rt['ramp_down_timeout_s']
        self.fdcc_halt = None            # reason string once an abort_wrench trip halts motion; see clear_halt()
        self.scripted_gains = {'K': self.fdcc_cfg['scripted']['stiffness']}   # see set_gains()
        self._gain_request = None        # applied by the control loop (imp is not thread-safe)
        self._reanchor_request = False   # see reanchor()
        zf = self.fdcc_cfg['zforce']
        self.zforce_gain, self.zforce_max_speed = zf['gain'], zf['max_speed']
        self._zf_target = None           # adaptive z-force target, see zforce_target()
        self._stats_lock = threading.Lock()
        self._stats = self._new_stats()
        self._check_robot_config()

        print("Initializing environment...")
        print(f"Robot IP:   {robot_ip}")
        print(f"Gripper IP: {gripper_ip}")
        print(f"Servo  {self.home_pose} frequency: {servo_frequency} Hz")

        # ----------------------------
        # threading
        # ----------------------------
        self.stop_flag = False
        self.threads = []
        self.obs_mode = obs_mode

        # ----------------------------
        # observation buffer
        # ----------------------------
        self.dataset_path = dataset_path
        self.save_interval = save_interval  # save thread loop interval in seconds
        self.robot_obs: list[RobotObs] = []
        self.gripper_obs: list[GripperObs] = []
        self.camera_obs: list[CameraObs] = []
        self.commands: list[Command] = []
        self.save_eps = save_eps
        self.image_idx = 0
        self.metadata = metadata

    def wait_for_obs(self):
        while len(self.camera_obs) == 0 or len(self.robot_obs) == 0 or len(self.gripper_obs) == 0:
            time.sleep(0.01)

    def get_obs(self):
        # Assume obs is populated with at least one entry
        if self.obs_mode == 'latest':
            obs = {
                'image': self.camera_obs[-1].image,
                'state': {
                    'actual_pose': self.robot_obs[-1].actual_pose,
                    'actual_force': self.robot_obs[-1].actual_force,
                    'filtered_force': self.robot_obs[-1].filtered_force,
                    'gripper_width': self.gripper_obs[-1].gripper_width,
                    'gripper_force': self.gripper_obs[-1].gripper_force,
                },
                'network_status': self.network_status
            }
            return obs

        elif self.obs_mode == 'mean':
            raise NotImplementedError("Mean obs mode not implemented yet")

    def step(self, des_pose, des_gripper_state, des_zforce=0., adaptive_mode=False, dualsense=None):
        """Args:
            des_pose: URPose
            des_gripper_state: int (GRIP_OPEN=0, GRIP_CLOSED=1)
            des_zforce: float (desired z-force in N)
            adaptive_mode: bool (whether to use adaptive z-force control)
        """
        log_cmd = Command(time=time.time() - self.t0,
                          des_pose=des_pose, des_gripper=des_gripper_state,
                          adaptive_mode=adaptive_mode, des_zforce=des_zforce,
                          controller_state=dualsense)
        self.commands.append(log_cmd)

        if self.adaptive_mode and not adaptive_mode:
            # Transitioning adaptive -> position
            self.last_step_t = time.perf_counter()
            self.last_step_end = des_pose
        else:
            self.last_step_t = time.perf_counter()
            self.last_step_end = self.des_pose

        self.des_pose = des_pose
        self.des_gripper_state = des_gripper_state
        self.des_zforce = des_zforce
        self.adaptive_mode = adaptive_mode
        return self.get_obs()

    def start(self):
        if self.dataset_path is not None:
            prefix = 'episode'
            ix = episode_index(self.dataset_path, prefix=prefix)
            self.epi_path = pathlib.Path(self.dataset_path) / f'{prefix}{ix:06d}'
            self.epi_path.mkdir(parents=True, exist_ok=True)
            os.makedirs(self.epi_path / 'images', exist_ok=True)

        self.stop_flag = False
        self.threads = [
            threading.Thread(target=self._control_loop, daemon=True),
            threading.Thread(target=self._camera_loop, daemon=True),
            threading.Thread(target=self._gripper_loop, daemon=True),
            threading.Thread(target=self._network_loop, daemon=True),
        ]
        if self.dataset_path is not None:
            self.threads.append(threading.Thread(target=self._logger_loop, daemon=True,))

        self.t0 = time.time()
        for thread in self.threads:
            thread.start()
        self.wait_for_obs()

    def reset(self, home_pose):
        """
        Reset environment:
            1. Open/home the gripper
            2. Move robot to home pose
            3. Set gripper to default width

        Args:
            home_pose: URPose
        """
        if home_pose is None:
            home_pose = self.home_pose

        print('Resetting environment...')
        if len(self.threads) > 0:
            self.stop_flag = True
            for thr in self.threads:
                thr.join()
            if self.dataset_path is not None:
                self.save_data()
        self.camera = Camera(sid="843212070496", crop_mode=self.camera_crop_mode)

        # ============================================================
        # Home / open gripper
        # ============================================================
        if self.gripper.gripstate().value != wsg.GripperState.IDLE.value:
            self.gripper.stop().wait()
            while self.gripper.gripstate().value != wsg.GripperState.IDLE.value:
                time.sleep(.1)

        g = self.gripper.home()
        g.ack.wait()

        # ============================================================
        # Move robot home (blocking)
        # ============================================================
        # _control_loop stops the RTDE script on exit, which also clears its
        # watchdog -- otherwise this blocking moveL would trip it (C207A0).
        if not self.ctrl.isProgramRunning():
            self.ctrl.reuploadScript()
        self.ctrl.moveL(home_pose, 0.1, 0.1)
        self.des_pose = home_pose  # Ensure robot doesn't move after homing
        self.last_step_t = -1

        # Wait for gripper homing to finish
        g.finished.wait()
        g = self.gripper.home()
        g.wait()

        # ============================================================
        # Move gripper to default open width
        # ============================================================
        g = self.gripper.move(position=self.open_width, speed=self.g_speed)
        g.finished.wait()
        self.gripper_state = GRIP_OPEN

        self.gripper.set_pwt(20)

        # ============================================================
        # Reset observations
        # ============================================================
        self.robot_obs: list[RobotObs] = []
        self.gripper_obs: list[GripperObs] = []
        self.camera_obs: list[CameraObs] = []
        self.commands: list[Command] = []
        self.ctrl.zeroFtSensor()

        print('payload kg', self.recv.getPayload())
        print('payload cog', self.recv.getPayloadCog())

        print('Environment reset complete.')

    def close(self):
        self.stop_flag = True
        # Bounded joins: a worker can be parked in a no-timeout blocking call
        # (gripper query .wait(), or ctrl.servoL after a protective stop) and
        # never re-check stop_flag. Since these are daemon threads, don't wait
        # on them forever -- that wedges the main thread and forces a ctrl-Z.
        for thr in self.threads:
            thr.join(timeout=2.0)
            if thr.is_alive():
                print(f'Warning: thread {thr.name} did not exit; leaving it to daemon cleanup.')
        self.camera.close()
        if self.dataset_path is not None:
            self.save_data()

    def init_period(self):
        self.period_init = time.perf_counter()

    def wait_period(self):
        """
        Waits for a time corresponding to `input_frequency`. Expects `init_period()` to be called at the top of the loop.
        """
        delta = time.perf_counter() - self.period_init
        sleep_time = max(0, 1 / self.input_frequency - delta)
        time.sleep(sleep_time)

    def interpolate(self):
        t = time.perf_counter() - self.last_step_t
        perc = min(1, t * self.input_frequency)
        return interpolate(self.last_step_end, self.des_pose, perc)

    force_alpha = 0.03
    _force_filtered = np.zeros(6)

    def filter_force(self, force):
        self._force_filtered = self.force_alpha * np.array(force) + (1 - self.force_alpha) * self._force_filtered
        return self._force_filtered

    def zforce_target(self, target_z, filtered_force):
        """
        Adaptive z-force: integrate the force error into the z TARGET, rate-limited.
        The old servoL PID returned actual z + kp*err + kd*d(err)/dt: under the admittance a
        target built from the actual pose moves with the arm, its velocity feeds forward and
        cancels the damping, and the kd term turned force noise into target jumps -- the arm
        went wild on Triangle (logs-debug-fdcc/episode000003, 22.9 s).
        """
        if self._zf_target is None:
            self._zf_target = target_z                  # continue from where the target was
        err = filtered_force.z - self.des_zforce
        rate = np.clip(self.zforce_gain * err, -self.zforce_max_speed, self.zforce_max_speed)
        self._zf_target += rate * self.dt
        return self._zf_target
    
    def _set_gripstate(self, gs):
        self.gripper_state = gs

    def request_zero_ft(self):
        """Ask the control loop to zero the F/T sensor at a safe point.

        self.ctrl (RTDE control) is not thread-safe, so it must only be touched
        from _control_loop. Callers on other threads (e.g. an interrupt-sequence
        .then callback) set this flag instead of calling ctrl.zeroFtSensor()
        directly, which would race with servoL() and can hang the RTDE handshake.
        """
        self._zero_ft_request = True

    # ================================================================
    # FDCC helpers
    # ================================================================
    def _check_robot_config(self):
        """Warn if the controller's payload / TCP differ from fdcc.toml (an unsaved pendant edit once did)."""
        r = self.fdcc_cfg['robot']
        mass, tcp = self.recv.getPayload(), np.array(self.ctrl.getTCPOffset())
        if abs(mass - r['payload_kg']) > r['payload_tolerance_kg']:
            print(f'WARNING: payload {mass:.3f} kg, fdcc.toml says {r["payload_kg"]:.3f}')
        if np.linalg.norm(tcp - np.array(r['tcp_offset'])) > r['tcp_tolerance_m']:
            print(f'WARNING: TCP offset {np.round(tcp, 4)}, fdcc.toml says {r["tcp_offset"]}')

    def _new_stats(self):
        return {'n': 0, 't0': None, 't1': None, 'dt_max': 0.0, 'work_max': 0.0, 'slow': 0,
                'f_max': 0.0, 't_max': 0.0, 'leash': 0, 'e_max': np.zeros(2), 'vsat': 0,
                'v_max': 0.0, 'ff_min': 1.0, 'last': {}}

    def fdcc_stats(self):
        """
        Control-loop diagnostics since the last call (then reset), for printing.
          hz, dt_max_ms, work_max_ms, slow  : loop rate, worst period, worst compute, cycles > 2 dt
          F, tau / F_max, tau_max           : raw getActualTCPForce now / window max
          F_c                               : processed wrench at c (after filter/deadband/clamp)
          e_mm, e_deg / leash_pct           : error of the target FDCC tracks; % of cycles the leash held it back
          v_mm_s, v_max_mm_s / vsat_pct     : commanded speed now / max; % of cycles at the speed clamp
          ff_min                            : lowest feedforward fade (1 = full feedforward)
          state                             : 'ok', 'PSTOP', or 'HALT: <reason>'
        """
        with self._stats_lock:
            s, self._stats = self._stats, self._new_stats()
        n, last = max(s['n'], 1), s['last']
        span = (s['t1'] - s['t0']) if s['n'] > 1 else float('nan')
        return {'hz': (s['n'] - 1) / span if s['n'] > 1 else float('nan'),
                'dt_max_ms': 1e3 * s['dt_max'], 'work_max_ms': 1e3 * s['work_max'], 'slow': s['slow'],
                'F': last.get('F', np.nan), 'tau': last.get('tau', np.nan),
                'F_max': s['f_max'], 'tau_max': s['t_max'], 'F_c': last.get('F_c', np.nan),
                'e_mm': last.get('e_mm', np.nan), 'e_deg': last.get('e_deg', np.nan),
                'e_max_mm': 1e3 * s['e_max'][0], 'e_max_deg': np.degrees(s['e_max'][1]),
                'leash_pct': 100 * s['leash'] / n, 'v_mm_s': last.get('v_mm_s', np.nan),
                'v_max_mm_s': 1e3 * s['v_max'], 'vsat_pct': 100 * s['vsat'] / n, 'ff_min': s['ff_min'],
                'state': last.get('state', '?')}

    def set_gains(self, K=None, D=None, M=None, sel=None):
        """
        Change the admittance gains from the next control cycle (thread-safe; fdcc.Impedance
        is only touched by _control_loop). Same forms as Impedance.set_gains; None keeps.
        """
        self._gain_request = {'K': K, 'D': D, 'M': M, 'sel': sel}

    def reanchor(self):
        """
        Restart the leashed target at the arm's pose on the next cycle (thread-safe).
        For scripted starts: MotionStep begins at the arm, and walking the stale teleop
        target back to it at the speed limit was fed forward as a lurch (episode000005, 25.3 s).
        """
        self._reanchor_request = True

    def restore_gains(self):
        """Back to the fdcc.toml gains."""
        p = self.imp.p
        self.set_gains(K=p.K, D=p.D, M=p.M, sel=p.sel)

    def _reset_ctrl(self):
        """Controller from rest; the leashed target restarts from the arm."""
        self.imp.reset()
        self._leashed = None

    def _leash(self, actual_pose, des_pose):
        """
        Non-dragging leash in NEWTONS: the target chases des_pose (at most the speed limit
        per cycle) but may not get further from the arm than leash_N / K -- the spring then
        pushes at most leash_N, the only push left in contact once the feedforward fades.
        Uses the current K (scripted gains shrink the distance), the largest K of each half
        so no axis exceeds the limit. It never drags the target after the arm, so pushing
        the arm by hand does not move the equilibrium. See fdcc.leash_step.
        """
        K = self.imp.K
        radius = self.leash_N / np.array([K[:3].max(), K[3:].max()])
        prev = np.asarray(actual_pose if self._leashed is None else self._leashed, float)
        self._leashed, held = fdcc.leash_step(prev, des_pose, actual_pose, radius, self.imp.p.speed * self.dt)
        self._leash_held = any(held)
        return URPose(*self._leashed)

    def clear_halt(self):
        """Resume after an abort_wrench halt (the arm restarts tracking the current target)."""
        self.fdcc_halt = None

    def _ramp_down(self, v):
        """
        Bring the arm to rest through speedL, feeding the watchdog every cycle.
        speedStop()/moveL from speed block the host long enough to trip the watchdog.
        """
        dt, a = self.imp.p.dt, self.imp.p.accel
        v = np.array(v, float)
        t_end = time.perf_counter() + self.ramp_timeout
        while time.perf_counter() < t_end:
            t_start = self.ctrl.initPeriod()
            if self.recv.isProtectiveStopped() or self.recv.isEmergencyStopped():
                break
            for sl, am in ((slice(0, 3), a[0]), (slice(3, 6), a[1])):
                n = np.linalg.norm(v[sl])
                v[sl] *= max(n - am * dt, 0.0) / n if n > 0 else 0.0
            if self.ctrl.speedL(v.tolist(), a[0], self.speedl_time) is False:
                break
            self.ctrl.waitPeriod(t_start)
            tw = np.array(self.recv.getActualTCPSpeed())
            if not v.any() and np.linalg.norm(tw[:3]) < self.rest_speed[0] and np.linalg.norm(tw[3:]) < self.rest_speed[1]:
                break
        return np.zeros(6)

    def _control_loop(self):
        imp = self.imp
        self._reset_ctrl()
        wd = self.watchdog_hz > 0
        if wd:
            self.ctrl.setWatchdog(self.watchdog_hz)     # speedL keeps the last velocity if this thread stalls
        self._v_last = np.zeros(6)
        try:
            self._servo_loop(imp, wd)
        finally:
            try:
                self._ramp_down(self._v_last)
                self.ctrl.speedStop(self.stop_decel)
            finally:
                self.ctrl.stopScript()                  # also clears the watchdog; reset() re-uploads

    def _servo_loop(self, imp, wd):
        v_last, t_prev, was_stopped = np.zeros(6), None, False
        self.restore_gains()                           # a previous run may have ended stiff
        while not self.stop_flag:
            t_start = self.ctrl.initPeriod()
            now = time.perf_counter()
            state = 'ok'
            if self._zero_ft_request:
                v_last = self._v_last = self._ramp_down(v_last)   # zero only at rest
                self.ctrl.zeroFtSensor()
                self._reset_ctrl()
                self._zero_ft_request = False
                t_prev = None                                  # the pause is not a loop stall
            actual_pose = URPose(*self.recv.getActualTCPPose())
            if self._reanchor_request:
                self._reanchor_request = False
                self._leashed = np.asarray(actual_pose, float)
                imp.note_target_jump(self._leashed)
            req, self._gain_request = self._gain_request, None
            if req is not None:
                # Bumpless: rescale the leashed target's offset so K*xi, the spring force
                # the arm is balancing, is unchanged by the new K -- a bare K step moved
                # the arm at scripted-move starts (logs-debug-fdcc/episode000004).
                try:
                    new = (imp.bumpless_target(actual_pose, self._leashed, req['K'])
                           if req.get('K') is not None and self._leashed is not None else None)
                    imp.set_gains(**req)
                    if new is not None:
                        self._leashed = new
                        imp.note_target_jump(new)       # a rescale, not a velocity
                except ValueError as ex:
                    print(f'\nset_gains ignored: {ex}')
            actual_force = URPose(*self.recv.getActualTCPForce())
            filtered_force = URPose(*self.filter_force(actual_force))
            self.robot_obs.append(RobotObs(time=time.time() - self.t0,
                                  actual_pose=actual_pose, actual_force=actual_force,
                                  filtered_force=filtered_force))

            des_pose = self.des_pose
            des_gripper_state = self.des_gripper_state
            gripper_state = self.gripper_state

            # ----------------------------
            # gripper logic (non-blocking preferred)
            # ----------------------------
            if gripper_state != GRIP_MOVING and gripper_state != des_gripper_state:
                self.gripper_state = GRIP_MOVING
                if gripper_state == GRIP_OPEN:
                    self.gripper.grip(force=self.g_force, width=self.g_width, speed=self.g_speed) \
                        .finished.then(lambda _: self._set_gripstate(GRIP_CLOSED)) \
                        .catch(lambda e: (
                            self._set_gripstate(GRIP_CLOSED),
                            print(f'Gripper GRIP failed: {e}'),
                            print(f'Last 5 gripper commands: ', [c.des_gripper for c in self.commands[-5:]])
                            ))
                else:
                    cur_width = self.gripper_obs[-1].gripper_width
                    self.gripper.release(pullback=(self.open_width - cur_width) / 2, speed=self.g_speed) \
                        .finished.then(lambda _: self._set_gripstate(GRIP_OPEN)) \
                        .catch(lambda e: (
                            self._set_gripstate(GRIP_OPEN),
                            print(f'Gripper RELEASE failed: {e}'),
                            print(f'Last 5 gripper commands: ', [c.des_gripper for c in self.commands[-5:]])
                        ))

            # ----------------------------
            # blend -> admittance target
            # ----------------------------
            if self.last_step_t > 0:
                # Received at least 1 input
                des_pose = self.interpolate()
            target = self._leash(actual_pose, des_pose)                         # FDCC: non-dragging leash, in newtons
            # target = clamp(actual_pose, des_pose, self.max_position_step, self.max_orientation_step)  # old servoL leash: DRAGS the target after the arm

            # ----------------------------
            # adaptive z-force control
            # ----------------------------
            if self.adaptive_mode:
                target = target._replace(z=self.zforce_target(target.z, filtered_force))
            else:
                self._zf_target = None

            # ----------------------------
            # safety, then admittance -> speedL
            # ----------------------------
            W = np.asarray(actual_force, float)
            f, tq = np.linalg.norm(W[:3]), np.linalg.norm(W[3:])
            stopped = self.recv.isProtectiveStopped() or self.recv.isEmergencyStopped()
            if self.fdcc_halt is None and (f > self.abort_wrench[0] or tq > self.abort_wrench[1]):
                self.fdcc_halt = f'|F| {f:.1f} N, |tau| {tq:.2f} Nm over abort_wrench'
                print(f'\nFDCC HALT: {self.fdcc_halt} -- env.clear_halt() to resume')
                v_last = self._ramp_down(v_last)
            if stopped:
                was_stopped = True
                state = 'PSTOP'
            elif was_stopped:
                # Cleared on the pendant: the RTDE script died with the stop. Start it again
                # (a fresh script has no watchdog) and restart from rest.
                if not self.ctrl.isProgramRunning():
                    self.ctrl.reuploadScript()
                if wd:
                    self.ctrl.setWatchdog(self.watchdog_hz)
                self._reset_ctrl()
                was_stopped, state = False, 'resumed after pstop'
            if stopped or self.fdcc_halt is not None:
                self._reset_ctrl()
                v_last = np.zeros(6)
                if not stopped:
                    state = f'HALT: {self.fdcc_halt}'
                    if wd:
                        self.ctrl.kickWatchdog()        # holding still, not stalled
            else:
                v_last = imp.step(actual_pose, W, target)
                self.ctrl.speedL(v_last.tolist(), imp.p.accel[0], self.speedl_time)

            # ----------------------------
            # diagnostics (read with fdcc_stats())
            # ----------------------------
            xi = imp.last.get('xi', np.zeros(6))
            e_lin, e_ang = np.linalg.norm(xi[:3]), np.linalg.norm(xi[3:])
            v_lin = np.linalg.norm(v_last[:3])
            work = time.perf_counter() - now
            with self._stats_lock:
                s = self._stats
                if s['t0'] is None:
                    s['t0'] = now
                s['t1'] = now
                s['n'] += 1
                if t_prev is not None:
                    s['dt_max'] = max(s['dt_max'], now - t_prev)
                    s['slow'] += (now - t_prev) > 2 * self.dt
                s['work_max'] = max(s['work_max'], work)
                s['f_max'], s['t_max'] = max(s['f_max'], f), max(s['t_max'], tq)
                s['leash'] += self._leash_held
                s['e_max'] = np.maximum(s['e_max'], [e_lin, e_ang])
                s['vsat'] += v_lin > 0.999 * imp.p.speed[0]
                s['v_max'] = max(s['v_max'], v_lin)
                s['ff_min'] = min(s['ff_min'], float(np.min(imp.g)))
                s['last'] = {'F': f, 'tau': tq, 'F_c': float(np.linalg.norm(imp.last.get('F_c', np.zeros(6))[:3])),
                             'e_mm': 1e3 * e_lin, 'e_deg': np.degrees(e_ang), 'v_mm_s': 1e3 * v_lin,
                             'state': state}
            t_prev = now
            self._v_last = v_last

            self.ctrl.waitPeriod(t_start)

    def _camera_loop(self):
        while not self.stop_flag:
            image = self.camera.get_image().copy()
            self.camera_obs.append(CameraObs(time=time.time() - self.t0, image=image))

    def _gripper_loop(self):
        while not self.stop_flag:
            t0 = time.perf_counter()
            # print('============ SENDING QUERIES ================')
            force = self.gripper.force()
            pos = self.gripper.position()

            self.gripper_obs.append(GripperObs(time=time.time() - self.t0,
                                               gripper_width=pos.value,
                                               gripper_force=force.value))
            sleep_dur = max(0, 1.0 / self.gripper_query_frequency - (time.perf_counter() - t0))
            # print('============ QUERY RESOLVED ================')
            time.sleep(sleep_dur)

    def _network_loop(self):
        while not self.stop_flag:
            t0 = time.perf_counter()
            status = is_network_up(self.network_iface)
            self.network_status = status
            sleep_dur = max(0, 1.0 / self.network_query_frequency - (time.perf_counter() - t0))
            time.sleep(sleep_dur)

    def _logger_loop(self):
        image_path = self.epi_path / 'images'
        time_list = []
        pose_list = []
        force_list = []
        filt_force_list = []
        gpos_list = []
        gforce_list = []
        self.wait_for_obs()  # Ensure we have at least one obs before starting logging

        last_pose = None

        image_idx = 0
        tinit = time.time()
        while not self.stop_flag:
            t0 = time.perf_counter()
            obs = self.get_obs()

            # Do not log if stationary
            # cur_pose = np.r_[obs['state']['actual_pose'], obs['state']['gripper_width']]
            # delta = max(abs(cur_pose - last_pose)) if last_pose is not None else float('inf')
            # if delta < self.save_eps:
            #     sleep_time = max(0, self.save_interval - (time.perf_counter() - t0))
            #     time.sleep(sleep_time)
            #     continue
            # last_pose = cur_pose

            im_path = pathlib.Path(image_path) / f'{image_idx:06d}.png'
            cv2.imwrite(im_path, obs['image'])
            time_list.append(time.time() - tinit)
            pose_list.append(obs['state']['actual_pose'])
            force_list.append(obs['state']['actual_force'])
            filt_force_list.append(obs['state']['filtered_force'])
            gpos_list.append(obs['state']['gripper_width'])
            gforce_list.append(obs['state']['gripper_force'])
            image_idx += 1

            sleep_time = max(0, self.save_interval - (time.perf_counter() - t0))
            time.sleep(sleep_time)

        # IMPORTANT: logger must exit via stop_flag for data to be saved
        np.savez_compressed(
            self.epi_path / 'states.npz',
            time=np.array(time_list),
            pose=np.array(pose_list),
            force=np.array(filt_force_list),
            force_raw=np.array(force_list),
            gripper_width=np.array(gpos_list),
            gripper_force=np.array(gforce_list),
            metadata=self.metadata,
            allow_pickle=True
        )

    def save_data(self):
        # Save collected RAW data to HDF5.
        print(f'Saving data to {self.epi_path}...')
        path = self.epi_path / 'rawdata.h5'
        with h5py.File(path, 'w') as f:
            f.create_dataset('robot_obs/time', data=[obs.time for obs in self.robot_obs])
            f.create_dataset('robot_obs/actual_pose', data=[obs.actual_pose for obs in self.robot_obs])
            f.create_dataset('robot_obs/actual_force', data=[obs.actual_force for obs in self.robot_obs])

            f.create_dataset('gripper_obs/time', data=[obs.time for obs in self.gripper_obs])
            f.create_dataset('gripper_obs/gripper_width', data=[obs.gripper_width for obs in self.gripper_obs])
            f.create_dataset('gripper_obs/gripper_force', data=[obs.gripper_force for obs in self.gripper_obs])

            f.create_dataset('camera_obs/time', data=[obs.time for obs in self.camera_obs])
            f.create_dataset('camera_obs/image_bgr', data=[obs.image for obs in self.camera_obs])

            f.create_dataset('commands/time', data=[cmd.time for cmd in self.commands])
            f.create_dataset('commands/des_pose', data=[cmd.des_pose for cmd in self.commands])
            f.create_dataset('commands/des_gripper', data=[cmd.des_gripper for cmd in self.commands])
            f.create_dataset('commands/adaptive_mode', data=[cmd.adaptive_mode for cmd in self.commands])
            f.create_dataset('commands/des_zforce', data=[cmd.des_zforce for cmd in self.commands])

            control = [vars(cmd.controller_state) for cmd in self.commands]
            f.create_dataset('dualsense/time', data=[cmd.time for cmd in self.commands])
            for key in control[0].keys():
                f.create_dataset(f'dualsense/{key}', data=[item[key] for item in control])

            m = f.create_group('metadata')
            dict2hdf5(m, self.metadata)

        print(f'Data saved to {path}')
