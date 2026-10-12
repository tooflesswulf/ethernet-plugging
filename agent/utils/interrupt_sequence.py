"""
Interrupt sequences: temporarily take over RobotExecution.get_action() and
play queued commands (motions, gripper moves, waits, teleop) one action per
control tick, with Promise chaining for follow-up steps and custom logic.

Usage from within a RobotExecution subclass's get_action():

    if reset_condition:
        seq = interrupt(self)
        seq.move_relative([0, 0, 0.05, 0, 0, 0])   # retract 5 cm up
        seq.gripper(GRIP_OPEN)
        seq.move_to(self.home_pose) \\
           .then(lambda _: self.buffer.clear())    # custom logic via Promise
        return self.get_action()  # now taken over: plays the first tick
"""
from scipy.spatial.transform import Rotation as R
import numpy as np
import time

from util import URPose, interpolate
from promise import Promise


MIN_JERK_PEAK = 1.875          # peak / mean speed of the minimum-jerk profile
MOTION_PROFILE = 'trapezoid'   # MotionStep time scaling: 'linear', 'trapezoid' or 'min_jerk'
TRAPEZOID_ACCEL_S = 0.2        # trapezoid: time to reach the peak speed (and to stop from it)


def min_jerk(tau):
    """
    Minimum-jerk time scaling s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 on [0, 1]: zero speed
    and acceleration at both ends. A constant-speed line starts and stops the target at
    full speed -- a step in the admittance's feedforward (jerk at scripted starts/ends).
    """
    tau = min(max(tau, 0.0), 1.0)
    return tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)


def trapezoid(b):
    """
    Trapezoidal time scaling on [0, 1]: constant acceleration over the first fraction
    `b` (<= 0.5), constant speed, constant deceleration over the last `b`. Peak speed
    1 / (1 - b) of the mean; b = 0.5 is a triangle.
    """
    vp = 1.0 / (1.0 - b)

    def s(tau):
        tau = min(max(tau, 0.0), 1.0)
        if tau < b:
            return vp * tau * tau / (2 * b)
        if tau > 1 - b:
            return 1 - vp * (1 - tau) ** 2 / (2 * b)
        return vp * (tau - b / 2)
    return s


def pose_error(p, q):
    """(position error [m], orientation error [rad]) between two 6-poses."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    pos_err = np.linalg.norm(p[:3] - q[:3])
    rot_err = (R.from_rotvec(p[3:]) * R.from_rotvec(q[3:]).inv()).magnitude()
    return pos_err, rot_err


class InterruptSequence:
    """
    Takes over `rexec.get_action()` and plays a queue of steps, one action per
    control tick, until the queue drains — then restores normal control.

    Queue commands with the move_to / move_relative / gripper / wait / teleop
    methods; each returns a Promise resolved when that step finishes. The
    `.then()` callbacks run on the control thread *before* the next step
    ticks, so they can queue follow-up steps or run custom logic (e.g.
    clearing a policy action buffer) before control is handed back. Plain
    sequential calls queue in call order on the same sequence, so both styles
    work. NOTE: .then() needs a *callable* — .then(seq.gripper(0)) queues in
    the right order by accident but rejects the chain; use a lambda.

    While active, the run() loop keeps ticking normally (camera display,
    logging, square-button exit) but joystick/policy input is ignored. On
    completion the interface targets are overwritten to match wherever the
    sequence left the robot, so control resumes from there instead of the
    stale pre-interrupt target. Queue steps only while the sequence is active
    (before returning from get_action, or from .then callbacks); a finished
    sequence is never ticked again.
    """

    def __init__(self, rexec):
        self.rexec = rexec
        self.queue = []  # list of (step, Promise)
        obs = rexec.env.get_obs()
        # the leashed target, not the arm: an empty queue must not move the target
        self.last_action = (URPose(*obs['state']['target_pose']),
                            rexec.env.des_gripper_state, False, 0.)
        rexec.get_action = self._tick  # shadows the class method until the queue drains
        rexec._interrupt_sequence = self

    @classmethod
    def current(cls, rexec):
        """The active sequence on `rexec`, installing a new one if needed."""
        seq = getattr(rexec, '_interrupt_sequence', None)
        if seq is None:
            seq = cls(rexec)
        return seq

    # ============================================================
    # Queueing commands (each returns a Promise)
    # ============================================================
    def move_to(self, target_pose, gripper_state=None, relative=False,
                speed=0.08, rot_speed=0.5, pos_tol=2e-3, rot_tol=0.02, timeout=15.0):
        """
        Queue a motion to `target_pose`.

        Args:
            target_pose: URPose (or any 6-sequence [x, y, z, rx, ry, rz])
            gripper_state: gripper command held during the motion (GRIP_OPEN,
                GRIP_CLOSED); None holds the last commanded state
            relative: treat `target_pose` as a delta from wherever the robot
                is when the step starts (see move_relative)
            speed: translation speed in m/s
            rot_speed: rotation speed in rad/s (slowest of the two sets duration)
            pos_tol: position convergence tolerance in meters
            rot_tol: orientation convergence tolerance in radians
            timeout: give up (but continue the sequence) after this many seconds
        """
        return self.add(MotionStep(
            self.rexec, target_pose, gripper_state=gripper_state, relative=relative,
            speed=speed, rot_speed=rot_speed, pos_tol=pos_tol, rot_tol=rot_tol,
            timeout=timeout))

    def move_relative(self, delta_pose, gripper_state=None,
                      speed=0.08, rot_speed=0.5, pos_tol=2e-3, rot_tol=0.02, timeout=15.0):
        """
        Queue a relative motion: `delta_pose` is [dx, dy, dz, drx, dry, drz]
        (base-frame translation, tool-frame rotation) applied to wherever the
        robot is when the step starts, e.g. [0, 0, 0.05, 0, 0, 0] moves up
        5 cm. Same arguments as move_to.
        """
        return self.move_to(delta_pose, gripper_state=gripper_state, relative=True,
                            speed=speed, rot_speed=rot_speed, pos_tol=pos_tol,
                            rot_tol=rot_tol, timeout=timeout)

    def gripper(self, gripper_state, settle_time=0.3, width_eps=0.1, timeout=5.0):
        """
        Queue a gripper move, holding the current pose while it executes.

        Args:
            gripper_state: GRIP_OPEN (0) or GRIP_CLOSED (1)
            settle_time: done once the observed width has been stable this long (s)
            width_eps: width change (mm) below which the gripper counts as stable
            timeout: give up (but continue the sequence) after this many seconds
        """
        return self.add(GripperStep(
            self.rexec, gripper_state, settle_time=settle_time,
            width_eps=width_eps, timeout=timeout))

    def wait(self, t):
        """Queue a wait: hold the current pose and gripper state for `t` seconds."""
        return self.add(WaitStep(self.rexec, t))

    def teleop(self, until=None):
        """
        Queue joystick teleop. With until=None this is terminal: the sequence
        stays in control and the joystick keeps driving (the subclass's policy
        get_action never resumes). Pass a zero-arg predicate to hand control
        back when it fires, e.g.

            seq.teleop(until=lambda: self.iface.dualsense.state.DpadRight)

        Returns a Promise resolved when `until` fires (never, if until is None).
        """
        return self.add(TeleopStep(self.rexec, until))

    def add(self, step):
        """Queue a raw step: a callable returning action tuples, then None when done."""
        promise = Promise()
        self.queue.append((step, promise))
        return promise

    # ============================================================
    # Execution (installed as rexec.get_action)
    # ============================================================
    def _tick(self):
        while self.queue:
            step, promise = self.queue[0]
            action = step()
            if action is not None:
                self.last_action = action
                return action
            self.queue.pop(0)
            promise.resolve()  # .then() callbacks may queue more steps here
        self._uninstall()
        return self.last_action

    def _uninstall(self):
        rexec, iface = self.rexec, self.rexec.iface
        rexec.env.restore_gains()          # in case a MotionStep was cut short
        rexec.env.set_feedforward(True)    # likewise, mid-release
        del rexec.get_action
        rexec._interrupt_sequence = None

        # Discard stale joystick/adaptive state so control resumes from
        # wherever the sequence left the robot.
        des_pose, des_gripper, _, _ = self.last_action
        iface.targ_pose = np.array(des_pose)
        iface.targ_zforce = 0.
        iface.adaptive_mode = False
        iface.gripper_state = des_gripper


class Step:
    """
    Base class for per-tick interrupt steps, for InterruptSequence.add().

    Subclasses implement tick(t) — called once per control tick with `t`
    seconds since the step started — returning an action tuple
    (des_pose, des_gripper, adaptive_mode, des_zforce) or None when complete.
    on_start() runs on the first tick, so a step queued behind others plans
    from the robot's state when it becomes active, not when it was queued.
    """

    def __init__(self, rexec):
        self.rexec, self.env, self.iface = rexec, rexec.env, rexec.iface
        self._start_t = None

    def __call__(self):
        if self._start_t is None:
            self._start_t = time.perf_counter()
            self.on_start()
        return self.tick(time.perf_counter() - self._start_t)

    def on_start(self):
        pass

    def actual_pose(self):
        return URPose(*self.env.get_obs()['state']['actual_pose'])


class MotionStep(Step):
    """
    Drive the robot to a target pose, in two phases.

    Release: the commanded target walks back, at `release_speed`, to the one whose held
    pose (fdcc hold offset included) is the arm, at the current gains and without
    feedforward -- whatever the spring was pressing with drops to zero while the arm stays
    put. Done once within `release_tol`; immediate in free space. This replaces the old
    reanchor(), which snapped the target to the arm in one cycle: the same release, but a
    jump in the logged target that a policy cloning it cannot reproduce.

    Move: stiffen (fdcc.toml [scripted]), then interpolate the command (linear position,
    slerp orientation) from the leashed target to the goal with the MOTION_PROFILE time
    scaling (constant speed, trapezoid with TRAPEZOID_ACCEL_S ramps, or minimum-jerk), whose
    PEAK speeds are `speed` (m/s) and `rot_speed` (rad/s), whichever takes longer (min-jerk
    takes about 1.9x as long as linear). The goal is where the ARM should end up: the
    command aims at fdcc's unheld(goal), so a hold offset left by an earlier gain change
    under load does not shift it. Gains are restored when it finishes. Finishes when the
    measured pose converges (pos_tol meters, rot_tol radians) or `timeout` seconds elapse.

    With `relative=True`, `target_pose` is a delta [dx, dy, dz, drx, dry, drz]
    (base-frame translation, tool-frame rotation) applied to wherever the
    robot is when the move starts (after the release). `gripper_state` of None holds the
    last commanded gripper state.
    """

    def __init__(self, rexec, target_pose, gripper_state=None, relative=False,
                 speed=0.08, rot_speed=0.5, pos_tol=2e-3, rot_tol=0.02, timeout=15.0,
                 release_speed=(0.05, 0.5), release_tol=(1e-3, 0.01), release_timeout=3.0):
        super().__init__(rexec)
        self.target = URPose(*target_pose)
        self.gripper_state = gripper_state
        self.relative = relative
        self.speed, self.rot_speed = speed, rot_speed
        self.pos_tol, self.rot_tol = pos_tol, rot_tol
        self.timeout = timeout
        self.release_speed, self.release_tol = release_speed, release_tol
        self.release_timeout = release_timeout

    def on_start(self):
        # Continue from the leashed target, not the arm: no jump, the hold is kept.
        self.cmd = np.array(self.env.restart_at_target(), float)
        self.t_last = 0.
        self.t_move = None                  # set when the move phase starts
        self.env.set_feedforward(False)     # release: see env.set_feedforward()

    def _grip(self):
        return self.env.des_gripper_state if self.gripper_state is None else self.gripper_state

    def _release(self, t):
        """One release tick: the action, or None once released."""
        imp, arm = self.env.imp, np.array(self.actual_pose(), float)
        pos_err, rot_err = pose_error(imp.held(self.cmd), arm)
        if (pos_err < self.release_tol[0] and rot_err < self.release_tol[1]) or t > self.release_timeout:
            if t > self.release_timeout:
                print(f'release timed out (pos_err={pos_err:.4f} m, rot_err={rot_err:.4f} rad)')
            return None
        # Step toward the target that makes the spring pull toward the arm. The arm unloads
        # a little as the force drops, so the goal is re-read every tick.
        goal, dt = imp.unheld(arm), t - self.t_last
        self.t_last = t
        dp, dr = goal[:3] - self.cmd[:3], (R.from_rotvec(goal[3:]) * R.from_rotvec(self.cmd[3:]).inv()).as_rotvec()
        lp, la = np.linalg.norm(dp), np.linalg.norm(dr)
        sp = min(1.0, self.release_speed[0] * dt / lp) if lp > 0 else 1.0
        sa = min(1.0, self.release_speed[1] * dt / la) if la > 0 else 1.0
        self.cmd = np.r_[self.cmd[:3] + sp * dp, (R.from_rotvec(sa * dr) * R.from_rotvec(self.cmd[3:])).as_rotvec()]
        return URPose(*map(float, self.cmd)), self._grip(), False, 0.

    def _start_move(self, t):
        self.env.set_feedforward(True)      # the move starts from rest: nothing to kick
        self.t_move = t
        self.start_pose = URPose(*map(float, self.cmd))
        arm = self.actual_pose()
        if self.relative:
            self.goal = URPose(*map(float, np.r_[
                np.array(arm[:3]) + np.array(self.target[:3]),
                (R.from_rotvec(arm[3:]) * R.from_rotvec(self.target[3:])).as_rotvec(),
            ]))
        else:
            self.goal = self.target
        self.cmd_goal = URPose(*map(float, self.env.imp.unheld(self.goal)))
        print(f'Moving robot to {self.goal} ...')
        dist, ang = pose_error(self.start_pose, self.cmd_goal)
        # `speed` / `rot_speed` are PEAK speeds (and stay under fdcc's speed clamp), so
        # stretch the duration by each profile's peak / mean speed.
        cruise = max(dist / self.speed, ang / self.rot_speed, 1e-6)   # at peak speed throughout
        if MOTION_PROFILE == 'min_jerk':
            self.profile, self.duration = min_jerk, MIN_JERK_PEAK * cruise
        elif MOTION_PROFILE == 'trapezoid':
            ta = TRAPEZOID_ACCEL_S
            # same acceleration (peak / ta) either way; too short to reach the peak -> triangle
            self.duration = cruise + ta if cruise >= ta else 2 * np.sqrt(cruise * ta)
            self.profile = trapezoid(min(ta / self.duration, 0.5))
        else:
            self.profile, self.duration = (lambda tau: min(max(tau, 0.0), 1.0)), cruise
        # Stiffen -- at the teleop gains the last mm crawl in with D/K = 3.3 s against
        # pos_tol (fdcc.toml [scripted]). After the release nothing is pressed, so the gain
        # change leaves the hold as it is (fdcc.py step 2a only rescales a loaded spring).
        self.env.set_gains(**self.env.scripted_gains)

    def tick(self, t):
        if self.t_move is None:
            action = self._release(t)
            if action is not None:
                return action
            self._start_move(t)
        t -= self.t_move
        pos_err, rot_err = pose_error(self.actual_pose(), self.goal)
        # Only after the profile has finished: converging inside the tolerance while the
        # target is still decelerating stopped it in one tick -- a clunk every transition.
        if t >= self.duration and pos_err < self.pos_tol and rot_err < self.rot_tol:
            self.env.restore_gains()
            return None
        if t > self.timeout:
            print(f'motion to {self.goal} timed out (pos_err={pos_err:.4f} m, rot_err={rot_err:.4f} rad)')
            self.env.restore_gains()
            return None
        return interpolate(self.start_pose, self.cmd_goal, self.profile(t / self.duration)), self._grip(), False, 0.


class GripperStep(Step):
    """
    Hold the current commanded pose and command the gripper to
    `gripper_state` (GRIP_OPEN, GRIP_CLOSED). Finishes once the observed width has
    been stable (within `width_eps` mm) for `settle_time` seconds, or after
    `timeout`.
    """

    def __init__(self, rexec, gripper_state, settle_time=0.3, width_eps=0.1, timeout=5.0):
        super().__init__(rexec)
        self.gripper_state = gripper_state
        self.settle_time, self.width_eps, self.timeout = settle_time, width_eps, timeout
        self.last_width = None
        self.stable_t = 0.

    def tick(self, t):
        width = self.env.gripper_obs[-1].gripper_width
        if self.last_width is None or abs(width - self.last_width) > self.width_eps:
            self.stable_t = t
        self.last_width = width

        settled = t - self.stable_t > self.settle_time and t > self.settle_time
        if settled and self.env.gripper_state == self.gripper_state:
            return None
        if t > self.timeout:
            print(f'gripper move to state {self.gripper_state} timed out (width={width:.1f} mm)')
            return None
        return URPose(*self.env.des_pose), self.gripper_state, False, 0.


class WaitStep(Step):
    """Hold the current commanded pose and gripper state for `duration` seconds."""

    def __init__(self, rexec, duration):
        super().__init__(rexec)
        self.duration = duration

    def tick(self, t):
        if t > self.duration:
            return None
        return URPose(*self.env.des_pose), self.env.des_gripper_state, False, 0.


class TeleopStep(Step):
    """
    Joystick teleop (the base RobotExecution.get_action() behavior). On start,
    re-syncs the interface targets to the current commanded pose so teleop
    continues from wherever the sequence left the robot. Runs until the
    zero-arg `until` predicate returns truthy (checked once per tick); with
    until=None it never finishes and the sequence stays in teleop.
    """

    def __init__(self, rexec, until=None):
        super().__init__(rexec)
        self.until = until

    def on_start(self):
        print('Entering joystick teleop...')
        self.iface.targ_pose = np.array(self.env.des_pose)
        self.iface.targ_zforce = 0.
        self.iface.adaptive_mode = False
        self.iface.gripper_state = self.env.des_gripper_state

    def tick(self, t):
        if self.until is not None and self.until():
            return None
        return (URPose(*self.iface.target_pose), self.iface.gripper_state,
                self.iface.adaptive_mode, self.iface.target_zforce)
