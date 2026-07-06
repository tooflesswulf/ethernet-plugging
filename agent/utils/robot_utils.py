from scipy.spatial.transform import Rotation as R
import numpy as np
import time

from env import URPose, Env
from util import blend
from promise import Promise
from interface import DualSenseInterface
from agent.model.policy import DiffusionPolicy


def get_actions(policy: DiffusionPolicy, nimages, nagent_poses, curr_pose, curr_gripper_width):
    """
    nimages:      (T, C, H, W) in [0, 255]
    nagent_poses: (T, state_dim) raw/unnormalized
    Returns (des_poses (H, 6) absolute [trans, rotvec], des_widths (H,)) ready to execute.
    """

    conditions = {
        'rgb': (nimages / 255.0).unsqueeze(0),  # (1, T, C, H, W)
        'state': nagent_poses.unsqueeze(0),     # (1, T, state_dim); policy normalizes internally
    }
    naction = policy.predict_action(conditions)
    naction = naction.detach().to('cpu').numpy()[0]

    # integrate deltas (per the policy's action_mode) into absolute poses + widths
    return policy.integrate_actions(naction, curr_pose, curr_gripper_width)


class ResetSequence:
    """
    Hijacks `rexec.get_action()` and plays a queue of reset steps, one action
    per control tick, until the queue drains — then restores control.

    A step is a callable invoked once per tick that returns an action tuple
    (des_pose, des_gripper, adaptive_mode, des_zforce) to execute, or None
    when the step is complete. `add(step)` queues a step and returns a Promise
    that resolves when that step finishes; its `.then()` callbacks run on the
    control thread *before* the next step ticks, so they can queue follow-up
    steps or run custom logic (e.g. clearing a policy action buffer) before
    control is handed back.

    While hijacked, the run() loop keeps ticking normally (camera display,
    logging, square-button exit) but joystick/policy input is ignored. On
    completion the interface targets are overwritten to match wherever the
    sequence left the robot, so control resumes from there instead of the
    stale pre-reset target.
    """

    def __init__(self, rexec):
        self.rexec = rexec
        self.queue = []  # list of (step_fn, Promise)
        obs = rexec.env.get_obs()
        self.last_action = (URPose(*obs['state']['actual_pose']),
                            rexec.env.des_gripper_state, False, 0.)
        rexec.get_action = self._tick  # shadows the class method until the queue drains
        rexec._reset_sequence = self

    @classmethod
    def current(cls, rexec):
        """The active sequence on `rexec`, installing a new one if needed."""
        seq = getattr(rexec, '_reset_sequence', None)
        if seq is None:
            seq = cls(rexec)
        return seq

    def add(self, step):
        promise = Promise()
        self.queue.append((step, promise))
        return promise

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
        del rexec.get_action
        rexec._reset_sequence = None

        # Discard stale joystick/adaptive state so control resumes from
        # wherever the sequence left the robot.
        des_pose, des_gripper, _, _ = self.last_action
        iface.targ_pose = np.array(des_pose)
        iface.targ_zforce = 0.
        iface.adaptive_mode = False
        iface.gripper_state = des_gripper


def motion_step(rexec, target_pose, gripper_state=None,
                pos_tol=2e-3, rot_tol=0.02, timeout=15.0):
    """
    Per-tick step driving the robot to `target_pose`, for ResetSequence.add().

    Blends the commanded pose toward `target_pose` within the env's step
    limits, finishing when the measured pose converges (pos_tol meters,
    rot_tol radians) or `timeout` seconds elapse. `gripper_state` of None
    holds the last commanded gripper state. The clock starts on the step's
    first tick, not when it is queued.
    """
    env = rexec.env
    target_pose = URPose(*target_pose)
    cmd_pose = None
    deadline = None

    def step():
        nonlocal cmd_pose, deadline
        if cmd_pose is None:  # first tick: start from wherever the robot is now
            print(f'Resetting robot to {target_pose} ...')
            cmd_pose = URPose(*env.get_obs()['state']['actual_pose'])
            deadline = time.perf_counter() + timeout
        grip = env.des_gripper_state if gripper_state is None else gripper_state

        actual = np.array(env.get_obs()['state']['actual_pose'])
        pos_err = np.linalg.norm(actual[:3] - np.array(target_pose[:3]))
        rot_err = (R.from_rotvec(actual[3:]) * R.from_rotvec(target_pose[3:]).inv()).magnitude()
        if pos_err < pos_tol and rot_err < rot_tol:
            return None
        if time.perf_counter() > deadline:
            print(f'motion to {target_pose} timed out (pos_err={pos_err:.4f} m, rot_err={rot_err:.4f} rad)')
            return None

        cmd_pose = blend(cmd_pose, target_pose,
                         max_position_step=env.max_position_step,
                         max_orientation_step=env.max_orientation_step)
        return cmd_pose, grip, False, 0.

    return step


def gripper_step(rexec, gripper_state, settle_time=0.3, width_eps=0.1, timeout=5.0):
    """
    Per-tick step that holds the current commanded pose and commands the
    gripper to `gripper_state` (0=open, 1=closed), for ResetSequence.add().

    Finishes once the observed gripper width has been stable (within
    `width_eps` mm) for `settle_time` seconds, or after `timeout`.
    """
    env = rexec.env
    start_t = None
    stable_since = None
    last_width = None

    def step():
        nonlocal start_t, stable_since, last_width
        now = time.perf_counter()
        if start_t is None:
            start_t = now
        hold_pose = URPose(*env.des_pose)

        width = env.gripper_obs[-1].gripper_width
        if last_width is None or abs(width - last_width) > width_eps:
            stable_since = now
        last_width = width

        settled = now - stable_since > settle_time and now - start_t > settle_time
        if settled and env.gripper_state == gripper_state:
            return None
        if now - start_t > timeout:
            print(f'gripper move to state {gripper_state} timed out (width={width:.1f} mm)')
            return None
        return hold_pose, gripper_state, False, 0.

    return step


def reset_to_position(rexec, reset_pose, gripper_state=None,
                      pos_tol=2e-3, rot_tol=0.02, timeout=15.0):
    """
    Queue a motion back to `reset_pose` on `rexec`'s active ResetSequence
    (hijacking get_action() if not already hijacked), ignoring joystick/policy
    input until the whole sequence completes.

    Returns a Promise resolved when this motion finishes. Chain `.then()` to
    queue further steps or run custom logic before control is handed back:

        def get_action(self):
            if reset_condition:
                (reset_to_position(self, PRE_RESET_POSE)
                    .then(lambda _: reset_gripper(self, 0))
                    .then(lambda _: reset_to_position(self, self.home_pose))
                    .then(lambda _: self.action_chunk.clear()))
                return self.get_action()  # now hijacked: plays the first reset tick
            ...

    Calling it (or reset_gripper) several times in a row queues the steps in
    call order on the same sequence, so plain sequential style works too.
    """
    seq = ResetSequence.current(rexec)
    return seq.add(motion_step(rexec, reset_pose, gripper_state, pos_tol, rot_tol, timeout))


def reset_gripper(rexec, gripper_state, settle_time=0.3, width_eps=0.1, timeout=5.0):
    """
    Queue a gripper move (0=open, 1=closed) on `rexec`'s active ResetSequence,
    holding the current pose while it executes. Returns a Promise resolved
    when the gripper settles; see reset_to_position for chaining usage.
    """
    seq = ResetSequence.current(rexec)
    return seq.add(gripper_step(rexec, gripper_state, settle_time, width_eps, timeout))


def wait_for_circle(env: Env, iface: DualSenseInterface, close_gripper=False):
    freq = 250
    print('Waiting the circle...')
    while True:
        flag = iface.update(1 / freq)
        if flag == -1:
            raise RuntimeError('Square pressed, exiting.')

        des_pose = URPose(*iface.target_pose)
        des_gripper = iface.gripper_state
        if close_gripper:
            obs = env.step(
                des_pose=des_pose,
                des_gripper_state=des_gripper,
                des_zforce=iface.target_zforce,
                adaptive_mode=iface.adaptive_mode,
            )
        if des_gripper == 1:
            break
        time.sleep(1 / 250)

    time.sleep(0.1)
    env.gripper.wait_idle()
    time.sleep(1)
