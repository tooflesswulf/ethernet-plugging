import time

from env import URPose, Env, GRIP_CLOSED
from interface import DualSenseInterface
from agent.model.policy import DiffusionPolicy
from agent.utils.interrupt_sequence import InterruptSequence


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


def interrupt(rexec):
    """The active InterruptSequence on `rexec`, installing one if needed."""
    return InterruptSequence.current(rexec)


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
        if des_gripper == GRIP_CLOSED:
            break
        time.sleep(1 / 250)

    time.sleep(0.1)
    env.gripper.wait_idle()
    time.sleep(1)
