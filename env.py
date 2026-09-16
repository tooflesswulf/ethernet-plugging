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
from kinematics import URKin
from impedance import CartesianImpedance, SafetyMonitor, IDLE, IMPEDANCE, FAULT
from camera import Camera
import wsg

# Gripper command states (des_gripper_state / gripper_state)
GRIP_OPEN = 0
GRIP_CLOSED = 1
GRIP_MOVING = -1


class RobotObs(namedtuple('RobotObs', ('time', 'actual_pose', 'actual_force', 'filtered_force',
                                       'actual_q', 'actual_qd', 'tau_cmd', 'cmd_wrench'))):
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
        max_position_step=(0.03, 0.03, 0.03),
        max_orientation_step=0.05,
        coulomb_friction=None,
        workspace=None,
        watchdog_hz=10.0,
        gain_ramp_time=0.3,
        mode_blend_time=0.2,
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
        # NOTE: do not pass FLAG_UPPER_RANGE_REGISTERS -- it hangs construction on
        # this controller. It is only needed for getJacobian()/getMassMatrix(),
        # and the Jacobian is computed locally (kinematics.URKin, ~16 us).
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
        # Under impedance control these are the equilibrium-error LEASH, not a
        # speed limit: the spring force is bounded by K * max_position_step, so
        # size them per gain set as F_sat / K.
        self.max_position_step = np.array(max_position_step)
        self.max_orientation_step = max_orientation_step

        # ============================================================
        # Impedance control
        # ============================================================
        step = self.ctrl.getStepTime()
        if step <= 0:
            # Observed on this controller. Don't divide by it.
            print(f'Warning: getStepTime() returned {step}; assuming {self.dt} s.')
            step = self.dt
        self.dt = step
        self.tcp_offset = self.ctrl.getTCPOffset()
        self.kin = URKin(self.tcp_offset)
        self.imp = CartesianImpedance(f_c=coulomb_friction,
                                      tau_rated=self.kin.tau_rated)
        self.safety = SafetyMonitor(workspace=workspace, dt=self.dt)
        self.watchdog_hz = watchdog_hz
        self.gain_ramp_time = gain_ramp_time
        self.mode_blend_time = mode_blend_time
        self._mode = IDLE
        self._fault = None
        self._mode_blend = 0.0      # 0 = free-space gains, 1 = contact gains
        self._loop_hz = 0.0
        self._t_start_ctrl = time.perf_counter()
        self._last_tau = np.zeros(6)
        self._last_wrench = np.zeros(6)

        print("Initializing environment...")
        print(f"Robot IP:   {robot_ip}")
        print(f"Gripper IP: {gripper_ip}")
        print(f"Impedance {self.home_pose} at {1/self.dt:.0f} Hz")
        print(f"TCP offset: {np.round(self.tcp_offset, 5)}")

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

        # The old adaptive/position branch here existed only because zforce_pid
        # left des_pose.z stale. Under impedance, z is always tracked from the
        # setpoint, so there is one case.
        self.last_step_t = time.perf_counter()
        self.last_step_end = self.des_pose

        if self._fault is not None:
            # Latched fault: keep serving observations, ignore new setpoints.
            return self.get_obs()

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
            # Bounded, for the same reason close() is: a worker can park forever
            # in a blocking RTDE call after a protective stop. An unbounded join
            # here wedges the main thread.
            for thr in self.threads:
                thr.join(timeout=2.0)
                if thr.is_alive():
                    print(f'Warning: thread {thr.name} did not exit during reset.')
            if self.dataset_path is not None:
                self.save_data()
        self._fault = None
        self._mode = IDLE
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
        wedged = False
        for thr in self.threads:
            thr.join(timeout=2.0)
            if thr.is_alive():
                wedged = True
                print(f'Warning: thread {thr.name} did not exit; leaving it to daemon cleanup.')
        if wedged:
            # Under servoL a wedged control thread was survivable. Under torque
            # control it means returning from close() with the arm still holding
            # a stale torque, so escalate. This knowingly reaches for self.ctrl
            # from outside _control_loop (see request_zero_ft) -- it is the
            # nuclear option, and stopping the script is always safe for the arm.
            print('Control thread wedged; calling stopScript() to drop torque.')
            try:
                self.ctrl.stopScript()
            except Exception as e:
                print(f'stopScript() failed: {e}')
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

    def _trip(self, reason):
        """
        Freeze: stop commanding a wrench, latch the fault, ignore new setpoints.
        Zero torque is gravity-compensated float, so the arm holds against gravity
        and does not lurch. Sticky on purpose -- a controller that re-arms itself
        after a divergence trip will diverge again immediately. The loop stays
        alive so camera/gripper/logging continue for post-mortem.
        """
        if self._fault is None:
            self._fault = reason
            self._mode = FAULT
            print(f'\n!!! IMPEDANCE TRIP: {reason}')

    def clear_fault(self):
        """Explicit operator action. Re-arms at the current pose."""
        self._fault = None
        self.des_pose = URPose(*self.recv.getActualTCPPose())
        self.last_step_end = self.des_pose
        self.last_step_t = -1
        self.imp.reset()
        self._mode = IDLE
        print('Fault cleared.')

    def _safe_stop_torque(self):
        """
        directTorque is re-applied by the controller every cycle while the command
        register still holds cmd 66 (the dispatch loop skips sync() for it), so a
        stale torque does NOT decay on its own. Zero it, then stopJ -- which is
        non-realtime, so it changes the register away from 66 and decelerates
        under the controller's own position control.
        """
        try:
            for _ in range(5):
                self.ctrl.directTorque([0.0] * 6)
            self.ctrl.stopJ(2.0)
        except Exception as e:
            print(f'safe stop failed: {e}')
        finally:
            self._mode = IDLE


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

    def _control_loop(self):
        # Arm the robot-side watchdog last, immediately before streaming starts:
        # once armed, any pause longer than 1/watchdog_hz stops the robot with
        # "fieldbus interrupted". 10 Hz, not 50 -- 20 ms is inside what Python GC
        # and terminal I/O can take.
        self.ctrl.setWatchdog(self.watchdog_hz)
        self._mode = IMPEDANCE
        self.imp.reset()
        self._t_start_ctrl = time.perf_counter()
        t_prev = self._t_start_ctrl
        ticks = 0
        try:
            while not self.stop_flag:
                t_start = self.ctrl.initPeriod()
                now = time.perf_counter()
                dt_actual = now - t_prev
                t_prev = now
                ticks += 1
                self._loop_hz = ticks / max(now - self._t_start_ctrl, 1e-9)
                try:
                    self._control_tick(t_start, dt_actual, ticks)
                except Exception as e:
                    self._trip(f'exception in control tick: {e!r}')
                    # The tick may have raised BEFORE commanding anything, and a
                    # tick that sends nothing leaves the controller re-applying
                    # the previous torque forever. Always command zero here.
                    try:
                        self.ctrl.directTorque([0.0] * 6)
                    except Exception:
                        pass
                self.ctrl.waitPeriod(t_start)
        finally:
            self._safe_stop_torque()

    def _control_tick(self, t_start, dt_actual, ticks):
        if self._zero_ft_request:
            self.ctrl.zeroFtSensor()
            self._zero_ft_request = False
        actual_pose = URPose(*self.recv.getActualTCPPose())
        actual_force = URPose(*self.recv.getActualTCPForce())
        filtered_force = URPose(*self.filter_force(actual_force))
        q = np.array(self.recv.getActualQ())
        qd = np.array(self.recv.getActualQd())
        twist = np.array(self.recv.getActualTCPSpeed())

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
        # equilibrium pose
        # ----------------------------
        # interpolate() smooths the 20-100 Hz setpoint stream up to the servo
        # rate; clamp() is now the equilibrium-error leash, which bounds the
        # spring force at K * max_position_step.
        if self.last_step_t > 0:
            des_pose = self.interpolate()
        eq_pose = clamp(
            actual_pose,
            des_pose,
            self.max_position_step,
            self.max_orientation_step,
        )

        # ----------------------------
        # impedance
        # ----------------------------
        # adaptive_mode is a gain schedule, blended rather than stepped: a step
        # change in K with a nonzero error is an instantaneous torque
        # discontinuity, which reads as a jolt.
        target_blend = 1.0 if self.adaptive_mode else 0.0
        rate = self.dt / max(self.mode_blend_time, self.dt)
        self._mode_blend += np.clip(target_blend - self._mode_blend, -rate, rate)

        if self._mode == FAULT:
            tau = np.zeros(6)
            F = np.zeros(6)
        else:
            elapsed = time.perf_counter() - self._t_start_ctrl
            ramp = min(1.0, elapsed / self.gain_ramp_time) if self.gain_ramp_time > 0 else 1.0
            J = self.kin.jacobian(q)
            tau, F, _ = self.imp.compute(
                q, qd, np.asarray(actual_pose, float), np.asarray(eq_pose, float),
                twist, J, blend=self._mode_blend, ramp=ramp,
            )
            reason = self.safety.check(q, qd, np.asarray(actual_pose, float),
                                       twist, np.asarray(actual_force, float),
                                       tau, dt_actual if ticks > 5 else None)
            if reason is None and ticks % 50 == 0:
                reason = self._check_robot_state()
            if reason is not None:
                self._trip(reason)
                tau = np.zeros(6)

        self._last_tau = tau
        self._last_wrench = F
        # Logged after the law runs so tau/wrench belong to the same tick.
        self.robot_obs.append(RobotObs(time=time.time() - self.t0,
                              actual_pose=actual_pose, actual_force=actual_force,
                              filtered_force=filtered_force,
                              actual_q=q, actual_qd=qd,
                              tau_cmd=tau, cmd_wrench=F))
        if self.ctrl.directTorque(tau.tolist()) is False:
            self._trip('directTorque() returned False')

    def _check_robot_state(self):
        """
        Slow checks, polled from self.recv only -- receive reads are cached RTDE
        state and are cheap, while ctrl.* goes through the control script and
        would block the loop.
        """
        if self.recv.isProtectiveStopped():
            return 'protective stop'
        if self.recv.isEmergencyStopped():
            return 'emergency stop'
        if self.recv.getSafetyMode() != 1:
            return f'safety mode {self.recv.getSafetyMode()}'
        return None

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
            # Added for impedance tuning. Existing keys are untouched --
            # scripts/rawdata_to_dataset.py reads by explicit key, so extras are
            # ignored and old checkpoints keep working.
            f.create_dataset('robot_obs/actual_q', data=[obs.actual_q for obs in self.robot_obs])
            f.create_dataset('robot_obs/actual_qd', data=[obs.actual_qd for obs in self.robot_obs])
            f.create_dataset('robot_obs/tau_cmd', data=[obs.tau_cmd for obs in self.robot_obs])
            f.create_dataset('robot_obs/cmd_wrench', data=[obs.cmd_wrench for obs in self.robot_obs])

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
