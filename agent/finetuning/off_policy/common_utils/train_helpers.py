import torch, numpy as np
from tqdm import tqdm  
from agent.finetuning.off_policy.common_utils import utils

def _train_agent(
        agent, train_steps, global_step, online_rb, offline_rb, cfg,  online_batch_size, offline_batch_size, warmup=False, device='cuda'
    ):
        """Run critic-only updates for warmup phase."""
        actor_update_cadence = cfg.algo.num_updates_per_iteration // cfg.algo.actor_updates_per_iteration
        stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
        collect_metrics = {}
        for i in tqdm( range(train_steps) ):
            # Sample mixed online/offline batch
            # Sample batches from replay buffers
            online_batch = online_rb.sample(online_batch_size)
            online_batch = online_batch.to(device, non_blocking=True)

            if cfg.algo.offline_fraction > 0.0:
                # Mixed online/offline training
                offline_batch = offline_rb.sample(offline_batch_size); offline_batch = offline_batch.to(device, non_blocking=True)
                batch = torch.cat([online_batch, offline_batch], dim=0)
            else:
                # Online-only training
                batch = online_batch
            
            # Update actor on the last iteration of each update cycle
            std = 0.0 if warmup else stddev
            update_actor = (not warmup) and (i + 1) % actor_update_cadence == 0
            metrics = agent.update(batch, stddev=std, update_actor=update_actor)
                
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

            # logging
            for k,v in metrics.items():
                if k not in collect_metrics:
                    collect_metrics[k] = []
                collect_metrics[k].append(v)
        # averaging metrics 
        for k, v_list in collect_metrics.items():
            v_arr = np.array(v_list)
            collect_metrics[k] = np.mean(v_arr)
        return collect_metrics