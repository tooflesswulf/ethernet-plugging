import numpy as np
from tqdm import tqdm
import argparse
import os
import wandb
import torch
import torch.nn as nn

from agent.utils.utils import save_checkpoint
from agent.utils.logging import NoOpLogger, setup_logger
from agent.model.diffusion import build_diffusion_policy
from agent.dataset.sequence import StitchedSequenceDataset

DEVICE = "cuda:0"


def to_device(x, device=DEVICE):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f"Unrecognized type in `to_device`: {type(x)}")


def batch_to_device(batch, device="cuda:0"):
    vals = [to_device(getattr(batch, field), device) for field in batch._fields]
    return type(batch)(*vals)


def train(task, dataset_path, ckpt_dir, epochs=100, use_wandb=False, log_interval=10, save_interval=10, device='cuda:0'):
    logger = setup_logger(use_wandb=use_wandb, project="realrobot-learning", name=f"pretrain-{task}-relact")
    dataset = StitchedSequenceDataset(dataset_path, horizon_steps=16, device=device)
    val_dataset = StitchedSequenceDataset(dataset_path, horizon_steps=16, max_n_episodes=1, device=device)
    val_dataset.state_max = dataset.state_max
    val_dataset.state_min = dataset.state_min
    val_dataset.action_max = dataset.action_max
    val_dataset.action_min = dataset.action_min
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,  # since all data are in ram, worker=0 is fine. multi-worker causing issue.
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64)
    nets, ema, opt, lr_scheduler, noise_scheduler = build_diffusion_policy(
        num_training_steps=len(dataloader) * epochs,
        num_warmup_steps=len(dataloader),
        device=device)
    pbar = tqdm(range(epochs))
    step = 0
    for epoch in pbar:
        epoch_losses = []
        for i, batch in enumerate(dataloader):
            batch = batch_to_device(batch, device)
            images, states, actions, B = batch.conditions['rgb'], batch.conditions['state'], batch.actions, len(
                batch.actions)
            # BxTxCxHxW -> (B T)xCxHxW -> (B T) x d -> BxTxd
            image_features = nets['vision_encoder'](images.flatten(end_dim=1)).reshape(*images.shape[:2], -1)
            obs_features = torch.cat([image_features, states], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)
            noise = torch.randn(actions.shape, device=device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

            # add noise to the clean images according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

            # predict the noise residual
            noise_pred = nets['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

            # L2 loss
            loss = nn.functional.mse_loss(noise_pred, noise)

            # optimize
            loss.backward()
            opt.step()
            opt.zero_grad()
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            ema.step(nets.parameters())

            # logging
            step += 1
            loss_cpu = loss.item()
            epoch_losses.append(loss_cpu)
            if i % log_interval == 0:
                logger.log({"train/loss": loss, "train/epoch": epoch}, step=step)

        avg_loss = round(np.mean(epoch_losses), 4)
        pbar.set_postfix({"loss": avg_loss})
        if epoch % save_interval == 0:
            save_checkpoint(nets, ema, ckpt_dir, epoch=epoch)
        val_mses, gripper_correctness = [], []
        for i, batch in enumerate(val_dataloader):
            with torch.no_grad():
                batch = batch_to_device(batch, device)
                images, states, actions, B = batch.conditions['rgb'], batch.conditions['state'], batch.actions, len(
                    batch.actions)
                # BxTxCxHxW -> (B T)xCxHxW -> (B T) x d -> BxTxd
                image_features = nets['vision_encoder'](images.flatten(end_dim=1)).reshape(*images.shape[:2], -1)
                obs_features = torch.cat([image_features, states], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)
                naction = torch.randn(actions.shape, device=device)

                # init scheduler
                noise_scheduler.set_timesteps(100)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = nets['noise_pred_net'](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
                val_mses.append(nn.functional.mse_loss(naction, actions).mean().item())
                tgt_gripper = actions[:, :, -1].long()  # it should be binary already
                tgt_mask = tgt_gripper <= 0
                tgt_gripper[tgt_mask] = -1
                tgt_gripper[~tgt_mask] = 1
                pred_gripper = naction[:, :, -1]
                mask = pred_gripper <= 0
                pred_gripper[mask] = -1
                pred_gripper[~mask] = 1
                gripper_correctness.append((pred_gripper == tgt_gripper).float().mean().item())

        logger.log({"val/mse_loss": np.mean(val_mses),
                   "val/gripper_correctness": np.mean(gripper_correctness), "val/epoch": epoch}, step=step)

    # save the lastest model
    # ema_nets = nets
    # ema.copy_to(ema_nets.parameters())
    save_checkpoint(nets, ema, ckpt_dir, epoch=None)


def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Policy Training')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--task', type=str, default='ethernet_plug_v2_dataset')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--data_dir', type=str, default='/zfsauton/scratch/yiqiw2/100%/datasets')
    parser.add_argument('--ckpt_dir', type=str, default='/zfsauton/scratch/yiqiw2/100%/ckpts')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_path, ckpt_dir = os.path.join(args.data_dir, args.task), os.path.join(args.ckpt_dir, args.task)
    train(args.task, dataset_path, ckpt_dir, args.epochs, use_wandb=args.use_wandb, device=args.device)
