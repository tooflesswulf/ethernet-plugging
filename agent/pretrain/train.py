import numpy as np
from tqdm import tqdm
import argparse
import os
import wandb
import torch
import torch.nn as nn

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from agent.utils.utils import save_checkpoint, compute_norm_stats
from agent.utils.logging import NoOpLogger, setup_logger
from agent.model.policy import DiffusionPolicy
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
    # obs_fields = ['pose', 'gripper_width', 'force', 'gripper_force', 'targ_ixs']
    obs_fields = ['pose', 'gripper_width', 'targ_ixs']
    dataset = StitchedSequenceDataset(dataset_path, obs_fields=obs_fields, horizon_steps=16, device=device)
    val_dataset = StitchedSequenceDataset(dataset_path, obs_fields=obs_fields,
                                          horizon_steps=16, max_n_episodes=1, device=device)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,  # since all data are in ram, worker=0 is fine. multi-worker causing issue.
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64)

    # Normalization stats from the training set; stored as buffers inside the
    # policy so they're saved in the checkpoint for un-normalizing at eval time.
    norm_stats = compute_norm_stats(dataset)
    policy = DiffusionPolicy(action_horizon=16, norm_stats=norm_stats,
                             state_dim=dataset.obs_dim, action_dim=dataset.act_dim).to(device)
    ema = EMAModel(parameters=policy.parameters(), power=0.75)
    opt = torch.optim.AdamW(params=policy.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=opt,
        num_warmup_steps=len(dataloader),
        num_training_steps=len(dataloader) * epochs,
    )
    pbar = tqdm(range(epochs))
    step = 0
    for epoch in pbar:
        epoch_losses = []
        for i, batch in enumerate(dataloader):
            batch = batch_to_device(batch, device)
            loss = policy.compute_loss(batch.actions, batch.conditions)

            # optimize
            loss.backward()
            opt.step()
            opt.zero_grad()
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            ema.step(policy.parameters())

            # logging
            step += 1
            loss_cpu = loss.item()
            epoch_losses.append(loss_cpu)
            if i % log_interval == 0:
                logger.log({"train/loss": loss, "train/epoch": epoch}, step=step)

        avg_loss = round(np.mean(epoch_losses), 4)
        pbar.set_postfix({"loss": avg_loss})
        if epoch % save_interval == 0:
            save_checkpoint(policy, ema, ckpt_dir, epoch=epoch)

        val_mses, gripper_correctness = [], []
        for i, batch in enumerate(val_dataloader):
            with torch.no_grad():
                batch = batch_to_device(batch, device)
                actions = batch.actions.float()
                # predict_action returns unnormalized actions, comparable to raw dataset actions
                naction = policy.predict_action(batch.conditions)

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

    # save the lastest model (with EMA weights applied)
    save_checkpoint(policy, ema, ckpt_dir, epoch=None)


def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Policy Training')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--task', type=str, default='ethernet_plug_v2_dataset')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--data_dir', type=str, default='/zfsauton/scratch/yiqiw2/100%/datasets')
    parser.add_argument('--ckpt_dir', type=str, default='/zfsauton/scratch/yiqiw2/100%/ckpts')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_path, ckpt_dir = os.path.join(args.data_dir, args.task), os.path.join(args.ckpt_dir, args.task)
    train(args.task, dataset_path, ckpt_dir, args.epochs, use_wandb=args.use_wandb, device=args.device)
