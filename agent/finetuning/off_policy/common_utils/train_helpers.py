import torch

def _train_agent(
        agent, train_steps, global_step, online_rb, offline_rb, cfg,  online_batch_size, offline_batch_size, warmup=False, device='cuda'
    ):
        """Run critic-only updates for warmup phase."""
        for i in range(train_steps):
            # Sample mixed online/offline batch
            # Sample batches from replay buffers
            online_batch = online_rb.sample(online_batch_size)
            online_batch = online_batch.to(device, non_blocking=True)

            # if cfg.algo.offline_fraction > 0.0:
            #     # Mixed online/offline training
            #     offline_batch = offline_rb.sample(offline_batch_size)
            #     offline_batch = offline_batch.to(device, non_blocking=True)
            #     batch = torch.cat([online_batch, offline_batch], dim=0)
            # else:
            #     # Online-only training
            #     batch = online_batch

            # Only update critic during warmup (update_actor=False)
            if warmup:
                pass
                metrics = agent.update(online_batch, stddev=0.0, update_actor=False, )
                print('Critic metrics', metrics)
                print()
                metrics = agent.update(online_batch, stddev=0.0, update_actor=True )
                print('All metrics', metrics)
                print('\t\tHard Code with Online Batch ONLY!!!!!')
                return 
            # # Update priorities for prioritized experience replay
            # if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
            #     # Update priorities in the batch for priority updates
            #     td_errors = metrics["_td_errors"]
            #     batch["_priority"] = td_errors

            #     if cfg.algo.offline_fraction > 0.0:
            #         # Mixed online/offline training - update both buffers
            #         online_batch_size_actual = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))

            #         # Update online buffer priorities
            #         if online_batch_size_actual > 0:
            #             online_batch_subset = batch[:online_batch_size_actual]
            #             online_rb.update_tensordict_priority(online_batch_subset)

            #         # Update offline buffer priorities
            #         if online_batch_size_actual < len(batch):
            #             offline_batch_subset = batch[online_batch_size_actual:]
            #             offline_rb.update_tensordict_priority(offline_batch_subset)
            #     else:
            #         # Online-only training - update only online buffer
            #         online_rb.update_tensordict_priority(batch)

            # Progress logging
            pass
