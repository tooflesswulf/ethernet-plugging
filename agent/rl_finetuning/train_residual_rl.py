# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations
from PIL import Image
from einops import rearrange
import os, pprint, random, shutil, time, torch, numpy as np, wandb, hydra, interface

from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictPrioritizedReplayBuffer, LazyMemmapStorage
from tqdm import tqdm
from env import Env, URPose, GRIP_OPEN
from agent.model.policy import DiffusionPolicy
from agent.dataset.sequence import GripperStats
from agent.utils.utils import get_dataset
from agent.rl_finetuning.config.residual_td3 import ResidualTD3DexmgConfig
from agent.rl_finetuning.off_policy.common_utils import utils
from agent.rl_finetuning.off_policy.rl.q_agent import QAgent
from agent.rl_finetuning.utils.dtype import to_uint8
from agent.rl_finetuning.utils.rb_transforms import MultiStepTransform
from agent.rl_finetuning.wrappers.rl_env import BasePolicyVecEnvWrapper
from agent.rl_finetuning.utils.checkkpoint import save_replay_buffers, load_replay_buffers

rl_scratch_dir = "./../../rl_online_buffer"
rl_buffer_dir = "./../../rl_dump_buffer"

def _add_transitions_to_buffer(
    *,
    obs: dict,
    next_obs: dict,
    actions: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    info: dict,
    device: torch.device,
   
    lowdim_keys: list[str],
    num_envs: int,
    online_rb: TensorDictPrioritizedReplayBuffer,
) -> None:
    """Helper function to create transitions and add them to the replay buffer.

    Handles terminal observations correctly and convert images to uint8 for storage.
    """
    obs_keys_set = ['observation.base_action', 'observation.rgb', 'observation.state']

    # Keep only relevant keys & convert images to uint8 for storage
    curr_obs_i = {k: v for k, v in obs.items() if k in obs_keys_set}
    next_obs_i = {k: v  for k, v in next_obs.items() if k in obs_keys_set}
    to_uint8(curr_obs_i, ['observation.rgb'])
    to_uint8(next_obs_i, ['observation.rgb'])
    for k in obs_keys_set:
        curr_v, next_v = curr_obs_i[k], next_obs_i[k]

        if not isinstance(curr_v, torch.Tensor):
            curr_obs_i[k] =  torch.tensor(curr_v)
        if not isinstance(next_v, torch.Tensor):
            next_obs_i[k] =  torch.tensor(next_v)

    td = TensorDict(
        {
            "obs": TensorDict(curr_obs_i, batch_size=[]),
            "next": TensorDict(
                {
                    "obs": TensorDict(next_obs_i, batch_size=[]),
                    "done": torch.tensor(done, dtype=torch.bool),
                    "reward": torch.tensor(reward, dtype=torch.float32),
                },
                batch_size=[],
            ),
            "action": actions.squeeze(0),
            "_priority": torch.tensor(10.0, dtype=torch.float32),  # High initial priority for new samples
        },
        batch_size=[],
    ).unsqueeze(0)

    online_rb.add(td)


# -----------------------------------------------------------------------------
# Main training loop -----------------------------------------------------------
# -----------------------------------------------------------------------------
def main(cfg: ResidualTD3DexmgConfig):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    # Enable performance optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---------------------------------------------------------------------
    # Load the behaviour-cloning policy that will serve as the "base" policy
    # for residual learning.
    # ---------------------------------------------------------------------
    assert "base_policy" in cfg, "Base policy configuration is required"
    base_policy = DiffusionPolicy.from_checkpoint(cfg.base_policy.ckpt, device)
    base_policy.to(device)
    base_policy.eval()
    lowdim_dim, action_dim, img_c, img_h, img_w =base_policy.state_min.shape[-1],  base_policy.action_min.shape[-1], 3, base_policy.img_size, base_policy.img_size
    if action_dim == 8:
        action_dim -= 1 # drop done dimension
    lowdim_keys = base_policy.obs_fields 
   
    # Load dataset and get normalization functions early
    print("Loading dataset and setting up normalization...")
    print("#"*20)
    print('No Normalization is done to the dataset!!!!!')
    print("#"*20)
    offline_dataset_path = os.path.join(cfg.offline_data.dir_path, cfg.offline_data.name)
    offline_episodes, ep_states, ep_actions, ep_rewards, total_transitions = get_dataset(offline_dataset_path, lowdim_keys, base_policy.action_mode, cfg.offline_data.num_episodes)
    grip = GripperStats(*base_policy.grip_stats)
    def get_envs(
        base_policy,
    ):
       
        # Create the vectorized environment
        
        env = Env(
            robot_ip="192.168.0.100",
            gripper_ip="192.168.0.20",
            camera_crop_mode=1,
            dataset_path=None,
            control_frequency=20,
            save_interval=1.0 / 20,
            gwidth=grip.grip_width_mm,
            gforce=grip.grip_force_n,
            gspeed=grip.grip_speed_mmps,
            gpullback=grip.grip_pullback_mm,
        )
        
        # Wrap it with the base policy wrapper
        return BasePolicyVecEnvWrapper(env=env,  base_policy=base_policy, image_size = (img_h, img_w), lowdim_keys=lowdim_keys, device=device)

    # ---------------------------------------------------------------------
    # Seeding (must be done before environment creation) ------------------
    # ---------------------------------------------------------------------
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    # Comprehensive seeding for reproducibility
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    # CUDA seeding for multi-GPU reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    print(f"Set random seed to {cfg.seed}")

    # ---------------------------------------------------------------------
    # Environment setup ----------------------------------------------------
    # ---------------------------------------------------------------------
    assert cfg.num_envs == 1, "Only support 1 environment for now because of how n_step is implemented"
    env = get_envs(base_policy=base_policy); cfg.eval_num_envs = 1
 
    # ---------------------------------------------------------------------
    # Networks ------------------------------------------------------------
    # ---------------------------------------------------------------------
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=['observation.rgb'],
        cfg=cfg.agent,
        residual_actor=True,  # Enable residual actor mode
    ).float()
   
    # Set up actor learning rate warmup
    actor_updates = 0
    if cfg.algo.actor_lr_warmup_steps > 0:
        print(
            f"Actor LR warmup enabled: 0.0 -> {cfg.agent.actor_lr:.2e} "
            f"over {cfg.algo.actor_lr_warmup_steps} actor updates"
        )

    # ---------------------------------------------------------------------
    # Replay buffers -------------------------------------------------------
    # ---------------------------------------------------------------------
    # -----------------------------------------------------------------
    # Use TensorDictPrioritizedReplayBuffer for unified PER support
    # For uniform sampling, we'll use alpha=0 and beta=0, and never update priorities
    # -----------------------------------------------------------------
    alpha = cfg.algo.priority_alpha if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0
    beta = cfg.algo.priority_beta if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0

    online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
    offline_batch_size = int(cfg.algo.batch_size * cfg.algo.offline_fraction)

    if cfg.algo.offline_fraction == 0.0:
        print("Online-only training mode: offline_fraction=0.0")

    # Use TensorDictPrioritizedReplayBuffer with optimized prefetching
    if os.path.isdir(rl_scratch_dir):
        shutil.rmtree(rl_scratch_dir)
    storage =  LazyMemmapStorage(cfg.algo.buffer_size, scratch_dir=rl_scratch_dir)
    online_rb = TensorDictPrioritizedReplayBuffer(
        storage=storage, # on disk storage
        alpha=alpha,
        beta=beta,
        eps=1e-6,  # Small epsilon added to priorities to prevent zero values
        priority_key="_priority",
        transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,  # Add prefetching
        batch_size=online_batch_size,
    )

    # Calculate buffer size for simplified approach (1 transition per frame pair)
    max_offline_transitions = total_transitions
    offline_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=max_offline_transitions, device="cpu"), # keep in RAM
        alpha=alpha,
        beta=beta,
        eps=1e-6,  # Small epsilon added to priorities to prevent zero values
        priority_key="_priority",
        transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,  # Add prefetching
        batch_size=max(offline_batch_size, 1),  # Ensure batch_size is at least 1
    )

 
    # ------------------------------------------------------------------
    # Convert offline dataset episodes into transitions and fill buffer
    # ------------------------------------------------------------------
    def _populate_offline_buffer(
        ep_paths,
        ep_states,
        ep_actions,
        ep_rewards,
        rb: ReplayBuffer,
        use_base_policy_for_base_actions: bool = False,
        base_policy = None,
        img_h=128, img_w=128
    ) -> int:
        """
        Iterates through *dataset* sequentially, converts consecutive frames
        into residual RL transitions and pushes them into *rb*.

        Two modes:
        1. GT-as-base (use_base_policy_for_base_actions=False):
           Uses GT actions as both the base action (in observations) and the target action
           (in transitions). Teaches residual policy to output zero: residual = GT - GT = 0

        2. Base-policy-as-base (use_base_policy_for_base_actions=True):
           Uses base policy to generate base actions and GT actions as targets.
           More consistent with online training: residual = GT - base_policy_action

        Returns the number of transitions added.
        """
        if use_base_policy_for_base_actions and base_policy is None:
            raise ValueError("base_policy must be provided when use_base_policy_for_base_actions=True")

        # Populate buffer from pre-loaded dataset
        print("Populating offline buffer from dataset...")

        episode_cache: dict[int, dict] = {}; transitions = 0; step_id = 0
        
        for ep_idx, ep_path in enumerate(tqdm(ep_paths, desc="Processing offline dataset") ):
            ep_image_dir = os.path.join(ep_path, 'images'); ep_state = torch.tensor( ep_states[ep_idx]).float(); ep_action = torch.tensor( ep_actions[ep_idx] ).float(); ep_reward =  torch.tensor( ep_rewards[ep_idx] ).float()
            images = torch.tensor( np.array( [ Image.open( os.path.join(ep_image_dir, n) ).resize((img_h, img_w)) for n in sorted(os.listdir(ep_image_dir), key = lambda x: int( x.replace('.png', '')) ) ] ) ).float()
            assert len(images) == len(ep_state)
            base_actions = None
            for step, (image, state, action, reward) in enumerate(zip(images, ep_state, ep_action, ep_reward)):
               
                # ------------------------------------------------------------------
                # Build observation and action directly for replay buffer ----------
                # ------------------------------------------------------------------
                # Extract data and keep on CPU (replay buffer uses CPU storage)
                done_flag = step == len(images)-1

                # Generate base action based on the selected mode
                if use_base_policy_for_base_actions:
                    # Use base policy to generate base action from current observation
                    # Build raw observation first for base policy inference
                    raw_obs = {'rgb': rearrange( image.unsqueeze(0).unsqueeze(0).to(device) / 255.0, 'B T H W C -> B T C H W'), 'state': state.unsqueeze(0).unsqueeze(0).to(device),}

                    # Get base action from base policy
                    if base_actions is None:
                        with torch.no_grad():
                            base_actions = base_policy.predict_action(raw_obs).squeeze(0).to(device)[:, :7]
                    base_action = base_actions[0]
           
                    base_actions =  base_actions[1:] if len(base_actions) > 1 else None
                else:
                   assert False, f"Not Implemented"
                
                # Build observation dict directly in target format
                curr_obs = { "observation.state": state.cpu(), "observation.base_action": base_action.cpu(), "observation.rgb": image.cpu() }
               
                # Convert images to uint8 for memory-efficient storage
                to_uint8(curr_obs, ["observation.rgb"])

                # ------------------------------------------------------------------
                # If we already cached the *previous* frame for this episode we can
                # create transitions now.
                # ------------------------------------------------------------------
                if ep_idx in episode_cache:
                    # Create transitions for each combination of prev and current variants
                    prev_obs = episode_cache[ep_idx]["obs"]
                    prev_action_scaled = episode_cache[ep_idx]["action"]
                   
                    transition = TensorDict(
                        {
                            "obs": TensorDict(prev_obs, batch_size=[]),
                            "action": prev_action_scaled,
                            "next": TensorDict(
                                {
                                    "obs": TensorDict(curr_obs, batch_size=[]),
                                    "done": torch.tensor(done_flag, dtype=torch.bool),
                                    "reward": torch.tensor(reward, dtype=torch.float32),
                                },
                                batch_size=[],
                            ),
                            "_priority": torch.tensor(10.0, dtype=torch.float32),  # High initial priority for new samples
                        },
                        batch_size=[],
                    ).unsqueeze(0)

                    rb.add(transition)
                    transitions += 1

                    step_id += 1
                else:
                    step_id = 0

                # Cache current frame for pairing with the next one ---------------
                episode_cache[ep_idx] = {
                    "obs": curr_obs,
                    "action": action,
                    "done": done_flag,
                    "step_id": step_id,
                }

        # Log final statistics
        print(f"Added {transitions} transitions")

        return transitions

    added = 0
    if cfg.algo.offline_fraction > 0.0:
        offline_rb, flag = load_replay_buffers(offline_rb, 'offline_rb', rl_buffer_dir )
        if not flag:
            added = _populate_offline_buffer(
                offline_episodes, ep_states, ep_actions, ep_rewards,
                rb=offline_rb,
                use_base_policy_for_base_actions=cfg.offline_data.use_base_policy_for_base_actions,
                base_policy=base_policy if cfg.offline_data.use_base_policy_for_base_actions else None,
            )
            print(f"Added {added} offline transitions to buffer (size={len(offline_rb)})")
            dump_replay_buffer(offline_rb, 'offline_rb', rl_buffer_dir)
    
    # ------------------------------------------------------------------
    # Warm-up phase (random policy) --------------------------------------
    # ------------------------------------------------------------------
    online_rb, flag = load_replay_buffers(online_rb, 'warmup_rb', rl_buffer_dir )
    cfg.algo.learning_starts = cfg.algo.learning_starts//2
    if len(online_rb) < cfg.algo.learning_starts :
        print(f"Warm-up: filling online buffer with {cfg.algo.learning_starts - len(online_rb)} random steps…")
        
        # --------------------------------------------------------------
        # Logging helper: print progress every 1 000 collected transitions
        # --------------------------------------------------------------
        next_log_threshold = 100  # first threshold for progress message

        reward_sum = 0
        episode_count = 0
        obs, _ = env.reset()
        while len(online_rb) < cfg.algo.learning_starts:
            
            if cfg.algo.use_base_policy_for_warmup:
                # Use base policy action + noise (residual exploration)
                # Since the environment wrapper always adds base_action to residual_action,
                # we just need to provide the noise as the residual action
                rand_actions = (
                    torch.rand((cfg.num_envs, action_dim), device=device) * 2 - 1
                ) * cfg.algo.random_action_noise_scale
            else:
                # Pure uniform random actions - need to cancel out the base policy action
                # Since env does: combined = base_action + residual_action
                # To get pure random: residual_action = random - base_action
                base_action = obs["observation.base_action"]  # Already normalized to [-1, 1]
                pure_random = (
                    torch.rand((cfg.num_envs, action_dim), device=device) * 2 - 1
                ) * cfg.algo.random_action_noise_scale
                rand_actions = pure_random - base_action
            
            next_obs, reward, terminated, info = env.step(rand_actions)
            
            done = terminated 
            reward_sum += reward
            episode_count += int( done )

            # Use the executed combined action returned by the environment
            combined_action = info["scaled_action"]
            _add_transitions_to_buffer(
                obs=obs,
                next_obs=next_obs,
                actions=combined_action,
                reward=reward,
                done=done,
                info=info,
                device=device,
                lowdim_keys=lowdim_keys,
                num_envs=cfg.num_envs,
                online_rb=online_rb,
            )

            # ----------------------------------------------------------
            # Progress logging (every ~1 000 transitions) --------------
            # ----------------------------------------------------------
            if len(online_rb) >= next_log_threshold:
                success_rate = reward_sum / episode_count if episode_count > 0 else 0.0
                print(
                    f"[Warm-up] {len(online_rb)} / {cfg.algo.learning_starts} "
                    f"transitions collected, reward_sum={reward_sum:.2f}, "
                    f"success_rate={success_rate:.3f} ({reward_sum}/{episode_count})"
                )
                next_log_threshold += 100

            obs = next_obs  # roll state
            if terminated or (not len(online_rb) < cfg.algo.learning_starts):
                # env.env.des_gripper_state = 0
                # env.env._homing()
                print('\t\tGet reward:', reward)
                print('Gripper before reset:', env.env.des_gripper_state, env.env.gripper_state)
                env.env.reset(env.env.home_pose) 
                # hardcode to cancelout previous behavior
                env.env.des_gripper_state = 0
                env.env.start()
                time.sleep(2)
                obs, _ = env.reset()
                print('Gripper after reset:', env.env.des_gripper_state, env.env.gripper_state)
    # save warmup data 
    if not flag:
        dump_replay_buffer(online_rb, 'warmup_rb', rl_buffer_dir)

    run_name = f"seed{cfg.seed}"
    if cfg.wandb.name is not None:
        run_name = f"{cfg.wandb.name}__{run_name}"

    _wandb_config = OmegaConf.to_container(cfg, resolve=True)
    # Remove notes from config if present
    assert isinstance(_wandb_config, dict)
    _wandb_config["wandb"].pop("notes", None)

    # Print a nice summary of the config
    print("Launching run with the following config:")
    pprint.pprint(_wandb_config)

    wandb.init(
        id=cfg.wandb.continue_run_id,
      
        project=cfg.wandb.project,
        config=_wandb_config,
        name=run_name,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
        notes=cfg.wandb.notes,
        group=cfg.wandb.group,
    )

    global_step = 0; training_cum_time = 0.0; episode_count = 0

    def _run_critic_warmup(
        agent, online_rb, offline_rb, cfg, device, online_batch_size, offline_batch_size
    ):
        """Run critic-only updates for warmup phase."""
        for i in range(cfg.algo.critic_warmup_steps):
            # Sample mixed online/offline batch
            # Sample batches from replay buffers
            online_batch = online_rb.sample(online_batch_size)
            online_batch = online_batch.to(device, non_blocking=True)

            if cfg.algo.offline_fraction > 0.0:
                # Mixed online/offline training
                offline_batch = offline_rb.sample(offline_batch_size)
                offline_batch = offline_batch.to(device, non_blocking=True)
                batch = torch.cat([online_batch, offline_batch], dim=0)
            else:
                # Online-only training
                batch = online_batch

            # Only update critic during warmup (update_actor=False)
           
            metrics = agent.update(batch, stddev=0.0, update_actor=False, )

            # Update priorities for prioritized experience replay
            if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                # Update priorities in the batch for priority updates
                td_errors = metrics["_td_errors"]
                batch["_priority"] = td_errors

                if cfg.algo.offline_fraction > 0.0:
                    # Mixed online/offline training - update both buffers
                    online_batch_size_actual = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))

                    # Update online buffer priorities
                    if online_batch_size_actual > 0:
                        online_batch_subset = batch[:online_batch_size_actual]
                        online_rb.update_tensordict_priority(online_batch_subset)

                    # Update offline buffer priorities
                    if online_batch_size_actual < len(batch):
                        offline_batch_subset = batch[online_batch_size_actual:]
                        offline_rb.update_tensordict_priority(offline_batch_subset)
                else:
                    # Online-only training - update only online buffer
                    online_rb.update_tensordict_priority(batch)

            # Progress logging
            if i % 100 == 0:
                print(
                    f"Critic warmup: {i} / {cfg.algo.critic_warmup_steps}, "
                    f"train/critic_qt={metrics['train/critic_qt']:.4f} "
                    f"train/critic_loss={metrics['train/critic_loss']:.4f}"
                )

    # ------------------------------------------------------------------
    # Critic warmup phase ----------------------------------------------
    # ------------------------------------------------------------------
    cfg.algo.critic_warmup_steps = 4000
    if cfg.algo.critic_warmup_steps > 0:
        print(f"Critic warmup: running {cfg.algo.critic_warmup_steps} critic-only updates...")
        _run_critic_warmup(
            agent=agent,
            online_rb=online_rb,
            offline_rb=offline_rb,
            cfg=cfg,
            device=device,
            online_batch_size=online_batch_size,
            offline_batch_size=offline_batch_size,
        )
        print("Critic warmup completed.")
  
    while global_step <= cfg.algo.total_timesteps:
        iter_start = time.time()
        # ------------------------------------------------------------------
        # (1) Collect action + Environment step for an episode ---------------------------
        # ------------------------------------------------------------------
        print("Collect 1 episode ...")
  
        obs, _ = env.reset()
        done = False; episode_length = 0
        while not done:
            with torch.no_grad(), utils.eval_mode(agent):
                stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
                action = agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)
            
            if cfg.algo.progressive_clipping_steps > 0:
                clip_factor = min(1.0, global_step / cfg.algo.progressive_clipping_steps)
                action = action * clip_factor
            
            next_obs, reward, terminated, info = env.step(action)
            done = terminated

            # Add to online replay buffer --------------------------------------
            # Use the executed combined action returned by the environment
            combined_action = info["scaled_action"]
            _add_transitions_to_buffer(
                obs=obs,
                next_obs=next_obs,
                actions=combined_action,
                reward=reward,
                done=done,
                info=info,
                device=device,
                lowdim_keys=lowdim_keys,
                num_envs=cfg.num_envs,
                online_rb=online_rb,
            ); obs = next_obs; global_step += cfg.num_envs
            episode_length += 1

            if done:
                print('\t\tGet reward:', reward)
                print('Gripper before reset:', env.env.des_gripper_state, env.env.gripper_state)
                env.env.reset(env.env.home_pose) 
                env.env.des_gripper_state = 0
                env.env.start()
                time.sleep(2)
                obs, _ = env.reset()
                print('Gripper after reset:', env.env.des_gripper_state, env.env.gripper_state)
                episode_count += int(done)
            
                wandb.log(
                    {
                        "training/episode_reward": reward, # track the last reward
                        "training/episode_count": episode_count,
                    },
                    step=global_step,
                )

        # ------------------------------------------------------------------
        # (4) Updates -------------------------------------------------------
        # ------------------------------------------------------------------
        if global_step % cfg.algo.update_every_n_steps == 0 or global_step == cfg.num_envs:
            
            actor_update_cadence = cfg.algo.num_updates_per_iteration // cfg.algo.actor_updates_per_iteration
            # Normal training loop - critic is already warmed up
            for i in tqdm(range(episode_length), desc='training actor/critic'):
                # --------------------------------------------------------------
                # Sample mixed online/offline batch
                # --------------------------------------------------------------
                global_step += 1
                # Sample batches from replay buffers
                online_batch = online_rb.sample(online_batch_size)
                online_batch = online_batch.to(device, non_blocking=True)

                if cfg.algo.offline_fraction > 0.0:
                    # Mixed online/offline training
                    offline_batch = offline_rb.sample(offline_batch_size)
                    offline_batch = offline_batch.to(device, non_blocking=True)
                    batch = torch.cat([online_batch, offline_batch], dim=0)
                else:
                    # Online-only training
                    batch = online_batch

                # Update actor on the last iteration of each update cycle
                update_actor = (i + 1) % actor_update_cadence == 0

                # Apply actor learning rate warmup
                if update_actor:
                    if cfg.algo.actor_lr_warmup_steps > 0:
                        # Calculate current LR with linear warmup from 0 to target
                        warmup_progress = min(1.0, actor_updates / cfg.algo.actor_lr_warmup_steps)
                        current_lr = cfg.agent.actor_lr * warmup_progress
                        for param_group in agent.actor_opt.param_groups:
                            param_group["lr"] = current_lr

                    actor_updates += 1

                metrics = agent.update(batch, stddev, update_actor, )

                # Update priorities for prioritized experience replay
                if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                    # Update priorities in the batch for priority updates
                    td_errors = metrics["_td_errors"]
                    batch["_priority"] = td_errors

                    if cfg.algo.offline_fraction > 0.0:
                        # Mixed online/offline training - update both buffers
                        online_batch_size_actual = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))

                        # Update online buffer priorities
                        if online_batch_size_actual > 0:
                            online_batch_subset = batch[:online_batch_size_actual]
                            online_rb.update_tensordict_priority(online_batch_subset)

                        # Update offline buffer priorities
                        if online_batch_size_actual < len(batch):
                            offline_batch_subset = batch[online_batch_size_actual:]
                            offline_rb.update_tensordict_priority(offline_batch_subset)
                    else:
                        # Online-only training - update only online buffer
                        online_rb.update_tensordict_priority(batch)

                metrics["data/batch_terminal_R"] = batch["next"]["reward"][~batch["nonterminal"]].mean(); metrics["data/terminal_share"] = (~batch["nonterminal"]).float().mean()


                # Prepare base logging dict
                if global_step % cfg.log_freq == 0:
                    log_dict = {
         
                        "training/global_step": global_step,
                        "buffer/online_size": len(online_rb),
                        "buffer/offline_size": len(offline_rb) if offline_rb else 0,
                        "training/actor_lr": agent.actor_opt.param_groups[0]["lr"],
                    }

                    # Add metrics, filtering out internal data
                    filtered_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
                    log_dict.update(filtered_metrics)

                    # Compute residual action statistics only when logging
                    if "_actions" in metrics:
                        actions = metrics["_actions"]
                        # Compute L1/L2 magnitudes (only during logging to save computation)
                        residual_l1_magnitude = torch.mean(torch.abs(actions)).item()
                        residual_l2_magnitude = torch.mean(torch.square(actions)).item()

                        log_dict["train/residual_l1_magnitude"] = residual_l1_magnitude
                        log_dict["train/residual_l2_magnitude"] = residual_l2_magnitude
                        log_dict["histograms/residual_actions"] = wandb.Histogram(actions.numpy().reshape(-1))
                    else:
                        residual_l1_magnitude = None
                        residual_l2_magnitude = None

                    # Add Q values histogram when available
                    if "_target_q" in metrics:
                        target_q = metrics["_target_q"]
                        log_dict["histograms/critic_qt"] = wandb.Histogram(target_q.numpy().reshape(-1))

                    if cfg.algo.progressive_clipping_steps > 0:
                        log_dict["training/progressive_clipping_factor"] = clip_factor

                    wandb.log(log_dict, step=global_step)

                    # Enhanced print statement with residual action magnitudes, gradient norms, and actor LR
                    current_actor_lr = agent.actor_opt.param_groups[0]["lr"]

                    if "train/actor_loss_base" in metrics:
                        actor_loss_str = f"actor_loss_base={metrics['train/actor_loss_base']:.4f}"
                        print_msg = (
                            f"[{global_step}] {actor_loss_str} "
                            f"critic_loss={metrics['train/critic_loss']:.4f} "
                            f"actor_lr={current_actor_lr:.2e}"
                        )
                    else:
                        # During critic warmup, actor might not be updated
                        print_msg = (
                            f"[{global_step}] critic_loss={metrics['train/critic_loss']:.4f} "
                            f"actor_lr={current_actor_lr:.2e} (actor not updated)"
                        )
                    if residual_l1_magnitude is not None and residual_l2_magnitude is not None:
                        print_msg += f" residual_l1={residual_l1_magnitude:.4f} residual_l2={residual_l2_magnitude:.4f}"

                    # Add gradient norms to print statement
                    if "train/actor_grad_norm" in metrics:
                        print_msg += f" actor_grad_norm={metrics['train/actor_grad_norm']:.4f}"

                    # Add L2 penalty if active
                    if "train/actor_l2_penalty" in metrics:
                        print_msg += f" l2_penalty={metrics['train/actor_l2_penalty']:.4f}"

                    # print(print_msg)

        training_cum_time += time.time() - iter_start

        # ------------------------------------------------------------------
        # (6) Logging -------------------------------------------------------
        # ------------------------------------------------------------------
        # if global_step % cfg.log_freq == 0:
        #     sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

          



# -----------------------------------------------------------------------------
# Hydra entry point -----------------------------------------------------------
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_name="residual_td3_dexmg_config")
def hydra_entry(cfg: ResidualTD3DexmgConfig):
    cfg_conf = OmegaConf.structured(cfg)
    main(cfg_conf)


if __name__ == "__main__":
    hydra_entry()