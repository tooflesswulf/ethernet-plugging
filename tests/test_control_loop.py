"""
Smoke test for Env._control_tick with RTDE mocked.

Env.__init__ opens sockets, so the instance is built with __new__ and the few
attributes the tick actually touches. Crude, but it exercises the real tick --
the gain blend, the fault latch, the torque path -- which is otherwise only
reachable on hardware.
"""
import sys
import types
import numpy as np
import pytest

for _n in ('rtde_control', 'rtde_receive', 'cv2', 'wsg', 'camera', 'pyrealsense2', 'h5py'):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules['rtde_control'].RTDEControlInterface = object
sys.modules['rtde_receive'].RTDEReceiveInterface = object
sys.modules['camera'].Camera = object
sys.modules['wsg'].WSG = object
sys.modules['wsg'].GripperState = types.SimpleNamespace(IDLE=types.SimpleNamespace(value=0))

import env as env_mod                                        # noqa: E402
from env import Env, URPose, GRIP_OPEN                       # noqa: E402
from impedance import CartesianImpedance, SafetyMonitor, IMPEDANCE, FAULT  # noqa: E402
from kinematics import URKin                                 # noqa: E402

Q = [0.0, -1.4, 1.4, -1.5, -1.5, 0.0]
TCP = [0, 0, 0.1537, 0, 0, 0]


class FakeCtrl:
    def __init__(self):
        self.torques = []
        self.scales = []
        self.ret = True
        self.stopped = False

    def initPeriod(self):
        return 0.0

    def waitPeriod(self, t):
        pass

    def directTorque(self, tau, viscous=None, coulomb=None):
        self.torques.append(np.array(tau))
        # Scales are load-bearing: a stop must zero them, a control tick must
        # send the configured ones. Record so tests can assert on it.
        self.scales.append((None if viscous is None else np.array(viscous),
                            None if coulomb is None else np.array(coulomb)))
        return self.ret

    def zeroFtSensor(self):
        pass

    def stopJ(self, a):
        self.stopped = True

    def stopScript(self):
        self.stopped = True

    def setWatchdog(self, hz):
        pass


class FakeRecv:
    def __init__(self, kin):
        self.q = np.array(Q)
        self.qd = np.zeros(6)
        self.twist = np.zeros(6)
        self.force = np.zeros(6)
        self.pose = kin.fk(self.q)
        self.protective = False

    def getActualTCPPose(self):
        return list(self.pose)

    def getActualTCPForce(self):
        return list(self.force)

    def getActualQ(self):
        return list(self.q)

    def getActualQd(self):
        return list(self.qd)

    def getActualTCPSpeed(self):
        return list(self.twist)

    def isProtectiveStopped(self):
        return self.protective

    def isEmergencyStopped(self):
        return False

    def getSafetyMode(self):
        return 1


@pytest.fixture
def e():
    kin = URKin(TCP)
    o = Env.__new__(Env)
    o.kin = kin
    o.ctrl, o.recv = FakeCtrl(), FakeRecv(kin)
    o.imp = CartesianImpedance(f_c=np.zeros(6), tau_rated=kin.tau_rated)
    o.safety = SafetyMonitor(dt=0.002)
    o.dt = 0.002
    o.t0 = 0.0
    o.robot_obs, o.commands, o.gripper_obs = [], [], []
    o.des_pose = URPose(*o.recv.pose)
    o.last_step_end = o.des_pose
    o.last_step_t = -1
    o.des_gripper_state = o.gripper_state = GRIP_OPEN
    o.adaptive_mode = False
    o.max_position_step = np.array([0.03, 0.03, 0.03])
    o.max_orientation_step = 0.05
    o.mode_blend_time = 0.2
    o.gain_ramp_time = 0.0
    o._zero_ft_request = False
    o._mode = IMPEDANCE
    o._fault = None
    o._mode_blend = 0.0
    o._t_start_ctrl = -10.0          # past the ramp
    o._last_tau = o._last_wrench = np.zeros(6)
    o._force_filtered = np.zeros(6)
    o.watchdog_hz = 10.0
    o._late = __import__('collections').deque()
    o.late_window = 5.0
    o.late_max = 25
    o._late_worst = 0.0
    o.stop_flag = True
    o._loop_hz = 0.0
    # Firmware friction-compensation scales, as Env.__init__ sets them.
    o.viscous_scale = [0.9, 0.9, 0.8, 0.9, 0.9, 0.9]
    o.coulomb_scale = [0.9, 0.8, 0.8, 0.7, 0.8, 1.0]
    o._scale_off = [0.0] * 6
    return o


def test_tick_runs_and_commands_torque(e):
    e._control_tick(0.0, 0.002, 10)
    assert len(e.ctrl.torques) == 1
    assert np.all(np.isfinite(e.ctrl.torques[0]))
    assert e._fault is None


def test_at_equilibrium_torque_is_negligible(e):
    """Bumpless: sitting exactly on the setpoint must not push."""
    e._control_tick(0.0, 0.002, 10)
    assert np.abs(e.ctrl.torques[0]).max() < 1e-9


def test_displacement_produces_restoring_torque(e):
    e.des_pose = URPose(*(np.array(e.recv.pose) + [0.01, 0, 0, 0, 0, 0]))
    e.last_step_end = e.des_pose
    e._control_tick(0.0, 0.002, 10)
    assert np.abs(e.ctrl.torques[0]).max() > 0.1


def test_obs_logged_with_tau_and_wrench(e):
    e._control_tick(0.0, 0.002, 10)
    obs = e.robot_obs[-1]
    assert obs.tau_cmd is not None and obs.cmd_wrench is not None
    assert len(obs.actual_q) == 6 and len(obs.actual_qd) == 6


def test_overspeed_trips_and_zeroes_torque(e):
    e.recv.twist = np.array([9.0, 0, 0, 0, 0, 0])
    e._control_tick(0.0, 0.002, 10)
    assert e._fault is not None
    assert e._mode == FAULT
    assert np.abs(e.ctrl.torques[-1]).max() == 0.0


def test_fault_is_sticky_and_keeps_zeroing(e):
    e.recv.twist = np.array([9.0, 0, 0, 0, 0, 0])
    e._control_tick(0.0, 0.002, 10)
    first = e._fault
    e.recv.twist = np.zeros(6)                 # condition clears
    e.des_pose = URPose(*(np.array(e.recv.pose) + [0.05, 0, 0, 0, 0, 0]))
    e._control_tick(0.0, 0.002, 11)
    assert e._fault == first                   # no auto-recovery
    assert np.abs(e.ctrl.torques[-1]).max() == 0.0


def test_directTorque_false_trips(e):
    e.ctrl.ret = False
    e._control_tick(0.0, 0.002, 10)
    assert e._fault is not None


def test_single_late_tick_zeroes_torque_but_does_not_trip(e):
    """
    A late tick means the loop was preempted (the logger PNG-encodes at 20 Hz and
    the camera buffer grows unbounded). Killing the run for that is wrong; zero
    torque for the tick is safe and recoverable.
    """
    e.des_pose = URPose(*(np.array(e.recv.pose) + [0.02, 0, 0, 0, 0, 0]))
    e.last_step_end = e.des_pose
    e._control_tick(0.0, 0.5, 10)
    assert e._fault is None
    assert np.abs(e.ctrl.torques[-1]).max() == 0.0


def test_sustained_lateness_trips(e):
    for i in range(e.late_max + 5):
        e._control_tick(0.0, 0.5, 10 + i)
    assert e._fault is not None and 'late' in e._fault


def test_recovers_after_isolated_late_ticks(e):
    """Occasional preemption must not accumulate into a trip."""
    e.des_pose = URPose(*(np.array(e.recv.pose) + [0.02, 0, 0, 0, 0, 0]))
    e.last_step_end = e.des_pose
    for i in range(10):
        e._control_tick(0.0, 0.5, 10 + i)          # late
        for j in range(5):
            e._control_tick(0.0, 0.002, 100 + i * 10 + j)   # on time
    assert e._fault is None
    assert np.abs(e.ctrl.torques[-1]).max() > 0.1  # still controlling


def test_early_ticks_exempt_from_lateness(e):
    """Startup jitter must not even count as late before the loop has settled."""
    e._control_tick(0.0, 0.5, 2)
    assert e._fault is None
    assert len(e._late) == 0


def test_adaptive_mode_blends_gradually(e):
    """A stepped gain change with nonzero error is a torque discontinuity."""
    e.adaptive_mode = True
    e._control_tick(0.0, 0.002, 10)
    after_one = e._mode_blend
    assert 0 < after_one < 1                    # ramps, does not jump
    for i in range(500):
        e._control_tick(0.0, 0.002, 11 + i)
    assert e._mode_blend == pytest.approx(1.0, abs=1e-6)


def test_blend_returns_to_free_gains(e):
    e.adaptive_mode = True
    for i in range(500):
        e._control_tick(0.0, 0.002, 10 + i)
    e.adaptive_mode = False
    for i in range(500):
        e._control_tick(0.0, 0.002, 600 + i)
    assert e._mode_blend == pytest.approx(0.0, abs=1e-6)


def test_leash_bounds_wrench_for_far_setpoint(e):
    """clamp() is the error leash: a distant setpoint must not command huge force."""
    e.des_pose = URPose(*(np.array(e.recv.pose) + [10.0, 0, 0, 0, 0, 0]))
    e.last_step_end = e.des_pose
    e._control_tick(0.0, 0.002, 10)
    assert np.all(np.abs(e._last_wrench) <= e.imp.F_sat + 1e-9)
    assert np.all(np.abs(e.ctrl.torques[-1]) <= e.imp.tau_sat + 1e-9)


def test_protective_stop_detected_on_slow_poll(e):
    e.recv.protective = True
    e._control_tick(0.0, 0.002, 50)             # slow checks run every 50 ticks
    assert e._fault is not None and 'protective' in e._fault


def test_exception_in_tick_still_commands_zero(e):
    """
    A tick that raises before commanding leaves the controller re-applying the
    previous torque indefinitely -- the stale-torque runaway. The loop must
    always send zero on the exception path.
    """
    e.des_pose = URPose(*(np.array(e.recv.pose) + [0.02, 0, 0, 0, 0, 0]))
    e.last_step_end = e.des_pose
    e._control_tick(0.0, 0.002, 10)
    assert np.abs(e.ctrl.torques[-1]).max() > 0.1      # a real torque is pending

    def boom(*a, **k):
        raise RuntimeError('RTDE read failed')
    e.recv.getActualTCPPose = boom
    e.stop_flag = False

    calls = {'n': 0}
    orig = e.ctrl.waitPeriod

    def stop_after_one(t):
        calls['n'] += 1
        if calls['n'] >= 1:
            e.stop_flag = True
        orig(t)
    e.ctrl.waitPeriod = stop_after_one

    e._control_loop()
    assert e._fault is not None and 'exception' in e._fault
    assert np.abs(e.ctrl.torques[-1]).max() == 0.0


def test_safe_stop_kills_the_friction_scales_not_just_the_torque(e):
    """
    Zero torque is NOT a stop while firmware friction compensation is live -- the
    controller keeps pushing a MOVING joint whatever we command. The scales have
    to go to zero too, which restores natural stiction and helps arrest it. This
    is what `identify` got wrong when it drove joint 1 into a wall.
    """
    e._safe_stop_torque()
    vis, cou = e.ctrl.scales[-1]
    assert vis is not None and cou is not None, 'stop sent bare directTorque'
    assert np.all(vis == 0) and np.all(cou == 0)
    assert np.all(e.ctrl.torques[-1] == 0)


def test_control_tick_sends_the_configured_scales(e):
    """A bare call inherits pybind's non-zero defaults; always be explicit."""
    e._control_tick(0.0, 0.002, 1)
    vis, cou = e.ctrl.scales[-1]
    assert np.allclose(vis, e.viscous_scale)
    assert np.allclose(cou, e.coulomb_scale)


def test_friction_feedforward_is_off_by_default(e):
    """
    The firmware compensates now. Stacking our own feedforward on top is the
    over-compensation that turns stiction into a limit cycle -- see
    RECALIBRATION-PLAN.md step 2 before switching any of it back on.
    """
    assert np.all(e.imp.f_c == 0)


def test_safe_stop_zeroes_then_stops(e):
    e._safe_stop_torque()
    assert e.ctrl.stopped
    assert np.abs(e.ctrl.torques[-1]).max() == 0.0
