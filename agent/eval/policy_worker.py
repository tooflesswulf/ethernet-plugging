"""
Policy inference in a child process.

In the eval process, a prediction thread shares one GIL with the env's 500 Hz control
loop and its camera / gripper / logger threads. A diffusion sample is a few hundred torch
calls, each of which has to win the GIL back, so on the robot the chunk rate fell from
~30 Hz (inference alone) to a few Hz -- and the control loop ran late in turn. With only
a few live chunks to average, every chunk that arrived or expired stepped the commanded
target: the 4-10 Hz vibration of logs-policy-imp (2026-10-07). A child process has its
own interpreter and GIL; the parent's thread only waits on a pipe, which releases it.
"""
import multiprocessing as mp


def _serve(conn, ckpt, device):
    import torch
    from agent.model.policy import DiffusionPolicy
    from agent.utils.robot_utils import get_actions

    policy = DiffusionPolicy.from_checkpoint(ckpt, device)
    policy.eval()
    conn.send('ready')
    while True:
        obs_seq = conn.recv()
        if obs_seq is None:
            break
        with torch.no_grad():
            conn.send(get_actions(policy, obs_seq, device, return_gains=True))


class PolicyWorker:
    """
    get_actions(policy, obs_deque, device, return_gains=True), computed by a child process
    that holds the policy. Requests are synchronous: one in flight at a time.
    """

    def __init__(self, ckpt, device='cuda'):
        ctx = mp.get_context('spawn')   # a forked child cannot use CUDA
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_serve, args=(child, ckpt, device), daemon=True)
        self._proc.start()
        child.close()
        self._ready = False

    def wait_ready(self):
        """Block until the child has loaded the policy (it starts loading at construction)."""
        if not self._ready:
            msg = self._conn.recv()
            assert msg == 'ready', msg
            self._ready = True

    def get_actions(self, obs_deque):
        """obs_deque: env obs dicts, as for robot_utils.get_actions. Only 'image' and 'state' are sent."""
        self.wait_ready()
        self._conn.send([{'image': o['image'], 'state': o['state']} for o in obs_deque])
        return self._conn.recv()

    def close(self):
        if self._proc.is_alive():
            try:
                self._conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.terminate()
