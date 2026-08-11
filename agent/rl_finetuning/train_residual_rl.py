# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import os, pprint, random, shutil, time, torch, numpy as np, wandb

from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictPrioritizedReplayBuffer, LazyMemmapStorage
from tqdm import tqdm
from env import Env, URPose, GRIP_OPEN
from agent.model.policy import DiffusionPolicy
from agent.dataset.sequence import GripperStats

from agent.rl_finetuning.config.residual_td3 import ResidualTD3DexmgConfig
from agent.rl_finetuning.off_policy.common_utils import utils
from agent.rl_finetuning.off_policy.rl.q_agent import QAgent
from agent.rl_finetuning.utils.dtype import to_uint8
from agent.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from agent.rl_finetuning.utils.rb_transforms import MultiStepTransform
from agent.rl_finetuning.wrappers.rl_env import BasePolicyVecEnvWrapper

rl_scratch_dir = "./../../rl_online_buffer"

def _add_transitions_to_buffer(
    *,
    obs: dict,
    next_obs: dict,
    actions: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    info: dict,
    device: torch.device,
    image_keys: list[str],
    lowdim_keys: list[str],
    num_envs: int,
    online_rb: TensorDictPrioritizedReplayBuffer,
) -> None:
    """Helper function to create transitions and add them to the replay buffer.

    Handles terminal observations correctly and convert images to uint8 for storage.
    """
    obs_keys_set = set(image_keys) | set(lowdim_keys)
    for i in range(num_envs):
        # Handle terminal observation (same logic as main loop)
        if done[i] and "final_obs" in info and info["final_obs"][i] is not None:
            final_obs_dict = info["final_obs"][i]
            next_obs_i = {k: torch.as_tensor(v, device=device) for k, v in final_obs_dict.items()}
        else:
            next_obs_i = {k: v[i] for k, v in next_obs.items()}

        curr_obs_i = {k: v[i] for k, v in obs.items()}

        # Keep only relevant keys & convert images to uint8 for storage
        curr_obs_i = {k: v for k, v in curr_obs_i.items() if k in obs_keys_set}
        next_obs_i = {k: v for k, v in next_obs_i.items() if k in obs_keys_set}
        to_uint8(curr_obs_i, image_keys)
        to_uint8(next_obs_i, image_keys)

        td = TensorDict(
            {
                "obs": TensorDict(curr_obs_i, batch_size=[]),
                "next": TensorDict(
                    {
                        "obs": TensorDict(next_obs_i, batch_size=[]),
                        "done": done[i],
                        "reward": reward[i],
                    },
                    batch_size=[],
                ),
                "action": actions[i],
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

    base_policy = DiffusionPolicy.from_checkpoint(cfg.ckpt, device)
    base_policy.to(device)
    base_policy.eval()

    # Extract the configuration from base policy
    base_cfg = base_policy.config

    # Load dataset and get normalization functions early
    print("Loading dataset and setting up normalization...")
    dataset = None

    # Create action scaler from dataset statistics
    action_scaler = ActionScaler.from_dataset_stats(
        action_stats=dataset.meta.stats["action"],
        action_scale=cfg.agent.actor.action_scale,
        min_range_per_dim=cfg.offline_data.min_action_range,
        device=device,
    )

    # Create state standardizer from dataset statistics
    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=dataset.meta.stats["observation.state"],
        min_std=cfg.offline_data.min_state_std,
        device=device,
    )

    def get_envs(
        env_name: str,
        num_envs: int,
        base_policy,
        device: str,
        video_key: str,
        debug: bool,
        action_scaler: ActionScaler,
        state_standardizer: StateStandardizer,
    ):
        assert action_scaler is not None, "action_scaler must be provided for consistent normalization"
        assert state_standardizer is not None, "state_standardizer must be provided for consistent normalization"

        # Create the vectorized environment
        grip = GripperStats(*base_policy.grip_stats)
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
        return BasePolicyVecEnvWrapper(
            vec_env=env,
            base_policy=base_policy,
            action_scaler=action_scaler,
            state_standardizer=state_standardizer,
        )

    # ---------------------------------------------------------------------
    # Seeding (must be done before environment creation) ------------------
    # ---------------------------------------------------------------------
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    # Comprehensive seeding for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

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
    env = get_envs(
        env_name=cfg.task,
        num_envs=cfg.num_envs,
        base_policy=base_policy,
        device=device_str,
        video_key=cfg.video_key,
        debug=cfg.debug,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
    ); cfg.eval_num_envs = 1
    
    # ---------------------------------------------------------------------
    # Observation / action dimensions -------------------------------------
    # ---------------------------------------------------------------------
    # Determine which image keys (camera observations) will be used. The
    # configuration can specify either a single camera name (str) or a list of
    # names.
    if isinstance(cfg.rl_camera, str):
        image_keys: list[str] = [cfg.rl_camera]
    else:
        image_keys = list(cfg.rl_camera)
    assert isinstance(image_keys, list)
    lowdim_dim = None # TODO
    img_c, img_h, img_w = None # TODO
    action_dim = 7
    lowdim_keys = ["observation.state", "observation.base_action"]

    # ---------------------------------------------------------------------
    # Networks ------------------------------------------------------------
    # ---------------------------------------------------------------------
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,  # Enable residual actor mode
    )
   
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
    max_offline_transitions = None 

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

    # Normalization functions already defined above - use them
    # ------------------------------------------------------------------
    # Convert offline dataset episodes into transitions and fill buffer
    # ------------------------------------------------------------------
    def _populate_offline_buffer(
        dataset,
        rb: ReplayBuffer,
        image_keys: list[str],
        num_episodes: int | None = None,
        use_base_policy_for_base_actions: bool = False,
        base_policy = None,
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
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        episode_cache: dict[int, dict] = {}
        transitions = 0
        step_id = 0

        for sample in tqdm(loader, desc="Processing offline dataset"):
            ep_idx = int(sample["episode_index"].item())
            if num_episodes is not None and ep_idx == num_episodes:
                break

            # ------------------------------------------------------------------
            # Build observation and action directly for replay buffer ----------
            # ------------------------------------------------------------------
            # Extract data and keep on CPU (replay buffer uses CPU storage)
            _gt_action: torch.Tensor = sample["action"].float().squeeze(0)
            gt_action_scaled = action_scaler.scale(_gt_action)
            done_flag = bool(sample["next.done"].item())

            # Generate base action based on the selected mode
            if use_base_policy_for_base_actions:
                # Use base policy to generate base action from current observation
                # Build raw observation first for base policy inference
                raw_obs = {}
                for k in sample:
                    if "observation" in k:
                        raw_obs[k] = sample[k].to(device)  # Keep batch dimension for base policy

                # Get base action from base policy
                with torch.no_grad():
                    base_action = base_policy.select_action(raw_obs)
                base_action_scaled = action_scaler.scale(base_action.squeeze(0).cpu())
            else:
                # Use GT action as base action (original behavior)
                base_action_scaled = gt_action_scaled

            # Build observation dict directly in target format
            curr_obs = {
                "observation.state": state_standardizer.standardize(sample["observation.state"].float().squeeze(0)),
                "observation.base_action": base_action_scaled,
            }
            for k in image_keys:
                curr_obs[k] = sample[k].squeeze(0)

            # Convert images to uint8 for memory-efficient storage
            to_uint8(curr_obs, image_keys)

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
                                "reward": torch.tensor(float(done_flag), dtype=torch.float32),
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
                "action": gt_action_scaled,
                "done": done_flag,
                "step_id": step_id,
            }

        # Log final statistics
        print(f"Added {transitions} transitions")

        return transitions

    added = 0
    if cfg.algo.offline_fraction > 0.0:
        added = _populate_offline_buffer(
            dataset=dataset,
            rb=offline_rb,
            image_keys=image_keys,
            num_episodes=cfg.offline_data.num_episodes,
            use_base_policy_for_base_actions=cfg.offline_data.use_base_policy_for_base_actions,
            base_policy=base_policy if cfg.offline_data.use_base_policy_for_base_actions else None,
        )

        print(f"Added {added} offline transitions to buffer (size={len(offline_rb)})")


    # ------------------------------------------------------------------
    # Warm-up phase (random policy) --------------------------------------
    # ------------------------------------------------------------------

    if len(online_rb) < cfg.algo.learning_starts :
        print(f"Warm-up: filling online buffer with {cfg.algo.learning_starts - len(online_rb)} random steps…")
        obs, _ = env.reset()
        # --------------------------------------------------------------
        # Logging helper: print progress every 1 000 collected transitions
        # --------------------------------------------------------------
        next_log_threshold = 1000  # first threshold for progress message

        reward_sum = 0
        episode_count = 0

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

            reward_sum += reward.sum().item()
            episode_count += done.float().sum().item()

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
                image_keys=image_keys,
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
                next_log_threshold += 1000

            obs = next_obs  # roll state

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
        resume=None if cfg.wandb.continue_run_id is None else "allow",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=_wandb_config,
        name=run_name,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
        notes=cfg.wandb.notes,
        group=cfg.wandb.group,
    )

    obs, _ = env.reset()

    global_step = 0
    best_eval_success_rate = 0.0
    training_cum_time = 0.0
    episode_count = 0

    def _run_critic_warmup(
        agent, online_rb, offline_rb, cfg, device, training_timer, online_batch_size, offline_batch_size
    ):
        """Run critic-only updates for warmup phase."""
        for i in range(cfg.algo.critic_warmup_steps):
            # Sample mixed online/offline batch
            with training_timer.time("batch_sampling"):
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
            with training_timer.time("gradient_update"):
                metrics = agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=agent)

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
        # (1) Collect action + Environment step ---------------------------
        # ------------------------------------------------------------------
       
        with torch.no_grad(), utils.eval_mode(agent):
            stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
            action = agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)

        if cfg.algo.progressive_clipping_steps > 0:
            clip_factor = min(1.0, global_step / cfg.algo.progressive_clipping_steps)
            action = action * clip_factor

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
        if done.any():
            episode_count += done.float().sum().item()
            # Extract episode information from final_info
            final_info = info["final_info"]
            episode_steps = final_info["episode_steps"]
            episode_indices = final_info["_episode_steps"]

            # Calculate discounted episode return
            discount_factor = cfg.algo.gamma ** episode_steps[episode_indices]
            episode_rewards = reward.cpu().numpy()[episode_indices]
            episode_return = np.mean(discount_factor * episode_rewards)

            wandb.log(
                {
                    "training/episode_return": episode_return,
                    "training/episode_steps": episode_steps,
                    "training/episode_count": episode_count,
                },
                step=global_step,
            )

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
            image_keys=image_keys,
            lowdim_keys=lowdim_keys,
            num_envs=cfg.num_envs,
            online_rb=online_rb,
        )

        obs = next_obs  # roll

        # ------------------------------------------------------------------
        # (3) Periodic evaluation ------------------------------------------
        # ------------------------------------------------------------------
        if global_step % cfg.eval_interval_every_steps == 0 and (cfg.eval_first or global_step > 0):
            eval_metrics = None

            # Handle model saving when success rate improves
            current_success_rate = eval_metrics["eval/success_rate"]
            if current_success_rate > best_eval_success_rate:
                print(f"🎉 New best success rate: {current_success_rate:.4f} (prev: {best_eval_success_rate:.4f})")
                best_eval_success_rate = current_success_rate

        global_step += cfg.num_envs

        # ------------------------------------------------------------------
        # (4) Updates -------------------------------------------------------
        # ------------------------------------------------------------------
        if global_step % cfg.algo.update_every_n_steps == 0 or global_step == cfg.num_envs:
            i = 0
            actor_update_cadence = cfg.algo.num_updates_per_iteration // cfg.algo.actor_updates_per_iteration
            # Normal training loop - critic is already warmed up
            while i < cfg.algo.num_updates_per_iteration:
                # --------------------------------------------------------------
                # Sample mixed online/offline batch
                # --------------------------------------------------------------
               
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

                
                metrics = agent.update(batch, stddev, update_actor, bc_batch=None, ref_agent=agent)

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

                metrics["data/batch_terminal_R"] = batch["next"]["reward"][~batch["nonterminal"]].mean()
                metrics["data/terminal_share"] = (~batch["nonterminal"]).float().mean()

                i += 1

        training_cum_time += time.time() - iter_start

        # ------------------------------------------------------------------
        # (6) Logging -------------------------------------------------------
        # ------------------------------------------------------------------
        if global_step % cfg.log_freq == 0:
            sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

            # Prepare base logging dict
            log_dict = {
                "training/SPS": sps,
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

            print(print_msg)



# -----------------------------------------------------------------------------
# Hydra entry point -----------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cfg_conf = None # instance of a config
    main(cfg_conf)
