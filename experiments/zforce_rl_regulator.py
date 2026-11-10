"""
Hand-written residual for the insert policy: keep the z-force low while it searches for
the port, without stopping the insertion push.

    python experiments/zforce_regulator.py --insert_ckpt <ckpt> --log_dir <dir>
"""
from agent.eval.eval_residual import ResidualEnv, ResidualActor, Correction
import os, hydra, argparse, json, numpy as np, shutil, torch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from agent.dataset.sequence import GripperStats
from agent.utils.utils import init_buffer
from agent.finetuning.config.residual_td3 import ResidualTD3DexmgConfig
from agent.finetuning.off_policy.common_utils import utils
from agent.finetuning.off_policy.rl.q_agent import QAgent
from agent.finetuning.off_policy.common_utils.train_helpers import _train_agent
from agent.utils.dtype import to_uint8
from agent.utils.offline_dataset_to_buffer import parse_offline_dataset, populate_offline_buffer
from agent.utils.rb_transforms import MultiStepTransform
from agent.utils.checkpoint import save_replay_buffers, load_replay_buffers
from agent.utils.utils import add_episode_to_buffer, _compute_dataset_norm_stats

def prepare_rl_experiment(cfg, dataset_path, lowdim_keys, actor_fields, base_policy, img_h=128, img_w = 128, device='cuda'):
    rl_scratch_dir = "./../../rl_online_buffer"
    rl_scratch_dir2 = "./../../rl_offline_buffer"
    rl_buffer_dir = "./../../rl_dump_buffer"
    offline_episodes, total_transitions, rl_input_dim = parse_offline_dataset(dataset_path, lowdim_keys, actor_fields)

    print('Remove all previous content in the directory')
    [ shutil.rmtree(scratch_dir ) for scratch_dir in [rl_scratch_dir, rl_scratch_dir2]  ]
    [ os.makedirs(scratch_dir) for scratch_dir in [rl_scratch_dir, rl_scratch_dir2]  ]
    print('Debug with SMAL ONLINE BATCH SIZE')
    online_batch_size = 2 #  int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
    offline_batch_size = 2 #  int(cfg.algo.batch_size * cfg.algo.offline_fraction)

    offline_rb, online_rb = init_buffer(total_transitions, rl_scratch_dir2, rl_scratch_dir, cfg, online_batch_size, offline_batch_size)
    offline_rb, flag = load_replay_buffers(offline_rb, 'offline_rb', rl_buffer_dir)
    if not flag:
        added = populate_offline_buffer(
            dataset_path,
            offline_episodes,
            rb=offline_rb,
            policy_base_actions=cfg.offline_data.use_base_policy_for_base_actions,
            base_policy=base_policy.to(device) if cfg.offline_data.use_base_policy_for_base_actions else None,
            img_h=img_h, img_w=img_w,
            device=device,
        )
        print(f"Added {added} offline transitions to buffer (size={len(offline_rb)})")
        save_replay_buffers(offline_rb, 'offline_rb', rl_buffer_dir)
    
    print('\nDebug, SKIP POPULATING OFFLINE\n')

    agent = QAgent(
        condition_length = cfg.algo.condition_length,
        prop_dim=rl_input_dim['state'],
        action_dim=rl_input_dim['action'],
        cfg=cfg.agent,
        residual_actor=True,  # Enable residual actor mode
    ).float().to(device)
    print('\nDebug, SKIP INIT STATS FOR AGENT\n')
    norm_stats = _compute_dataset_norm_stats(online_rb=online_rb, offline_rb=offline_rb if cfg.algo.offline_fraction > 0.0 else None, action_dim=rl_input_dim['action'])
    # From here on the agent owns normalization: act()/update() take raw measurements
    # and these stats ride along in the checkpoint as buffers.
    agent.set_norm_stats(
        norm_stats,
        min_action_range=cfg.offline_data.min_action_range,
        min_state_std=cfg.offline_data.min_state_std,
    )
    return offline_rb, online_rb, agent, online_batch_size, offline_batch_size
    

class ZForceRegulator(ResidualActor):
    def __init__(self, f_floor=5.0, gain=3e-3, max_speed=0.015, horizon=0.4, neural_actor=None, global_step=None, cfg=None):
        self.f_floor = f_floor      # [N] no correction below this
        self.gain = gain            # [m/s per N] upward velocity per N over f_floor
        self.max_speed = max_speed  # [m/s] fastest upward velocity: slower than an insertion push 
        self.horizon = horizon      # [s] the time over which the policy absorbs an offset
        # neural network and its parameters 
        self.neural_actor= neural_actor 
        self.global_step = global_step; self.cfg = cfg
    def act(self, step):
        if self.neural_actor is None:
            fz = float(step.state['filtered_force'][2])
            v = min(self.gain * max(0., fz - self.f_floor), self.max_speed)
            return Correction(dpose=np.r_[0., 0., v * self.horizon, 0., 0., 0.], info={'fz': fz, 'v': v})
        else:
            # Modify residualactor: get a list of step instead of step
            # concatenate a window of state and force 
            # return corrections, only on z-axis 
            pass
            with torch.no_grad():
                obs = None
                stddev = utils.schedule(self.cfg.algo.stddev_schedule, self.global_step)
                residual_action = self.neural_actor.act(obs, eval_mode=False, stddev=stddev, cpu=False)
            return Correction(dpose=np.r_[0., 0., residual_action[2], 0., 0., 0.], info={'fz': fz, 'v': v})

def summarize(ep, f_floor):
    """Per-episode numbers: outcome, and how the z-force, the offset and the target behaved."""
    s = {'episode': ep.index, 'success': ep.success, 'end': ep.end, 'aborted': ep.aborted,
         'duration': round(ep.duration, 2), 'path': ep.path, 'timing': ep.timing}
    if ep.ticks:
        t, fz, dz = ep.ticks['t'], ep.ticks['filtered_force'][:, 2], ep.ticks['dpose'][:, 2]
        dt = np.diff(t, append=t[-1])
        over = fz > f_floor
        s.update(fz_max=round(float(fz.max()), 1), fz_p95=round(float(np.percentile(fz, 95)), 1),
                 s_over_floor=round(float(dt[over].sum()), 2),
                 fz_mean_over_floor=round(float(fz[over].mean()), 1) if over.any() else None,
                 dz_max_mm=round(1000 * float(dz.max()), 1))
    return s

def deploy(episodes, env, args, actor_fields, online_rb, collect_data=True):
    """Deploy the actor for some episodes for evaluation or data collection."""
    successes, total_transitions, last_successful = [], 0, False
    for _ in range(episodes):
        env.register(ZForceRegulator(args.f_floor, args.gain, args.max_speed, args.horizon), period=args.period)
        env.wait_circle = not last_successful; ep = env.run()

        insert_episode = None if len(ep.steps) == 0 else ep.steps # a dictionary with 15 keys
        last_successful = ep.success
        if len(ep.steps) > 0:
            total_transitions += len(insert_episode['base_pose'])
        if insert_episode is not None and collect_data:
            add_episode_to_buffer(insert_episode, actor_fields, last_successful, online_rb)
        successes.append(last_successful)
    return successes, total_transitions
    
def main(args, cfg: ResidualTD3DexmgConfig):
    
    global_step = 0
    with ResidualEnv.from_args(args) as env:
        insert_policy = env._runner.policies['insert'][-1]
        ################# Prepare experiment: replay buffer, neural actor
        # fields for the base policy, use for base actiom computation; fields for online actor
        obs_fields = insert_policy.obs_fields; actor_fields = args.actor_fields
        online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction)); offline_batch_size = int(cfg.algo.batch_size * cfg.algo.offline_fraction)
        offline_rb, online_rb, agent, online_batch_size, offline_batch_size = prepare_rl_experiment(
                cfg,
                dataset_path=args.offline_dataset_path,
                lowdim_keys=obs_fields,
                actor_fields=actor_fields,
                base_policy=insert_policy
        )
        # modify actor_fields to fit env naming convention
        rename_actor_fields = [ 'filtered_force' if name == 'force' else ('actual_pose' if name == 'pose' else name) for name in actor_fields]
        actor_fields = rename_actor_fields

        ################ RL-warmup: fill the replay buffer with some episodes, train the critic
        warmup_episodes = 2 # hardcoded right now
        _, total_warmup_steps = deploy(warmup_episodes, env, args, actor_fields, online_rb); global_step += total_warmup_steps
        # critic_warmup(total_warmup_steps) # warmup for the same number of steps
        _train_agent( agent, total_warmup_steps, global_step, online_rb, offline_rb, cfg, online_batch_size, offline_batch_size, warmup=True); global_step += total_warmup_steps
        ################ RL-training: alternating between collect 1 episode, train for certain steps
        train_episodes = 2 # hardcoded right now 
        for ep_id in range(train_episodes):
            ep_success, train_steps = deploy(1, env, args, actor_fields, online_rb)
            _train_agent( agent, train_steps, global_step, online_rb, offline_rb, cfg, online_batch_size, offline_batch_size, warmup=False); global_step += train_steps


def hydra_entry():
    # 1. Define your argparse arguments exactly as before
    parser = argparse.ArgumentParser(description='Insert with a z-force-regulating residual, episode after episode.')
    ResidualEnv.add_args(parser)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--f_floor', type=float, default=5.0)
    parser.add_argument('--gain', type=float, default=3e-3)
    parser.add_argument('--max_speed', type=float, default=0.015)
    parser.add_argument('--horizon', type=float, default=0.4)
    parser.add_argument('--period', type=float, default=0.02)
    parser.add_argument('--offline_dataset_path', type=str)
    parser.add_argument('--actor_fields', nargs='+', type=str, default=[ 'pose', 'force'])
    
    # 2. Split args: 'args' goes to argparse, 'hydra_overrides' captures everything else
    args, hydra_overrides = parser.parse_known_args()

    # 3. Manually initialize and run Hydra with the leftover overrides
    # Note: Replace 'conf' with the actual relative path to your config directory
    config_dir = '/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ethernet-plugging/agent/finetuning/config'
    
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        # Hydra processes its config overrides here (e.g., agent.actor_lr=1e-5)
        cfg_conf = compose(config_name="residual_td3_dexmg_config", overrides=hydra_overrides)
    
    # 4. Hand off both clean configurations to your main execution function
    main(args, cfg_conf)

if __name__ == "__main__":
    # Call the function directly without the @hydra.main decorator
    hydra_entry()