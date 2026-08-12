"""
Train the image-only grasp classifier on a dataset built by
`scripts/build_grasp_dataset.py`.

    python -m agent.pretrain.train_grasp --data_dir ../data/grasp_cls_dataset \\
        --ckpt_dir logs --name grasp-cls

Validation holds out whole episodes (see `episode_split`), so the reported numbers
are not inflated by near-duplicate neighbouring frames.
"""
from tqdm import tqdm
import argparse
import pathlib
import numpy as np
import torch
import torch.nn as nn

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from agent.dataset.grasp_images import (GraspImageDataset, episode_split, expand_roi,
                                        source_image_shape, IMAGE_SIZE)
from agent.model.grasp import GRASP_ROI

# GRASP_ROI is quoted in pixels of the 480x480 wrist frame (agent/model/grasp.py).
GRASP_ROI_FRAME = 480
from agent.model.grasp_cls import GraspClassifier
from agent.utils.logging import setup_logger
from agent.utils.utils import save_checkpoint


def binary_metrics(logits, targets, threshold=0.0):
    """Accuracy / precision / recall / F1 for one epoch of validation logits."""
    pred = (logits > threshold).float()
    tp = float((pred * targets).sum())
    fp = float((pred * (1 - targets)).sum())
    fn = float(((1 - pred) * targets).sum())
    tn = float(((1 - pred) * (1 - targets)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        'accuracy': (tp + tn) / max(len(targets), 1),
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        # Mean of per-class recall -- the honest number if the split is lopsided.
        'balanced_accuracy': 0.5 * (recall + (tn / (tn + fp) if tn + fp else 0.0)),
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    all_logits, all_targets, losses = [], [], []
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        losses.append(criterion(logits, labels).item())
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())
    model.train()

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    metrics = binary_metrics(logits, targets)
    metrics['loss'] = float(np.mean(losses)) if losses else float('nan')
    return metrics


def train(name, dataset_path, ckpt_dir, epochs=30, batch_size=128, lr=1e-4, weight_decay=1e-6,
          image_size=IMAGE_SIZE, val_frac=0.2, seed=0, num_workers=8, backbone='resnet18',
          pretrained=True, augment=True, use_roi=False, use_wandb=False, log_interval=10,
          save_interval=10, device='cuda:0'):
    torch.manual_seed(seed)

    split = episode_split(dataset_path, val_frac=val_frac, seed=seed)
    roi = None
    if use_roi:
        height, width = source_image_shape(dataset_path)[:2]
        # GRASP_ROI is quoted for the 480x480 wrist frame; rescale it to whatever these
        # frames are, then pad it so the crop survives small camera shifts.
        roi = expand_roi(GRASP_ROI, 0.4, width=width, height=height, roi_frame_size=GRASP_ROI_FRAME)
        print(f'Cropping to ROI {roi} of {width}x{height} frames')

    train_set = GraspImageDataset(dataset_path, indices=split.train_indices, image_size=image_size,
                                  augment=augment, roi=roi)
    val_set = GraspImageDataset(dataset_path, indices=split.val_indices, image_size=image_size,
                                augment=False, roi=roi)
    train_neg, train_pos = train_set.class_counts
    val_neg, val_pos = val_set.class_counts
    print(f'train: {len(train_set)} frames from {len(split.train_episodes)} episodes '
          f'({train_pos} in-gripper, {train_neg} not)')
    print(f'val:   {len(val_set)} frames from {len(split.val_episodes)} episodes '
          f'({val_pos} in-gripper, {val_neg} not); episodes {split.val_episodes.tolist()}')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, drop_last=True,
                                               persistent_workers=num_workers > 0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=num_workers,
                                             persistent_workers=num_workers > 0)

    model = GraspClassifier(backbone=backbone, pretrained=pretrained, image_size=image_size,
                            roi=roi).to(device)
    ema = EMAModel(parameters=model.parameters(), power=0.75)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = get_scheduler(name='cosine', optimizer=opt, num_warmup_steps=len(train_loader),
                                 num_training_steps=len(train_loader) * epochs)
    # Re-weight the positive class if the aggregate is lopsided; ~1.0 when it is balanced.
    pos_weight = torch.tensor(train_neg / max(train_pos, 1), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    logger = setup_logger(use_wandb=use_wandb, project='realrobot-learning', name=name)
    logger.log_config(model.config)

    best_f1, step = -1.0, 0
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        epoch_losses = []
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)

            loss.backward()
            opt.step()
            opt.zero_grad()
            lr_scheduler.step()
            ema.step(model.parameters())

            step += 1
            epoch_losses.append(loss.item())
            if i % log_interval == 0:
                logger.log({'train/loss': loss.item(), 'train/lr': lr_scheduler.get_last_lr()[0],
                            'train/epoch': epoch}, step=step)

        # Validate the weights that actually get saved, i.e. the EMA ones.
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        metrics = evaluate(model, val_loader, criterion, device)
        ema.restore(model.parameters())

        logger.log({f'val/{k}': v for k, v in metrics.items()} | {'val/epoch': epoch}, step=step)
        pbar.set_postfix({'loss': round(float(np.mean(epoch_losses)), 4),
                          'val_acc': round(metrics['accuracy'], 4),
                          'val_f1': round(metrics['f1'], 4)})

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            save_checkpoint(model, ema, ckpt_dir, epoch='best')
        if epoch % save_interval == 0:
            save_checkpoint(model, ema, ckpt_dir, epoch=epoch)

    save_checkpoint(model, ema, ckpt_dir, epoch=None)
    print(f'Best val F1: {best_f1:.4f}')
    return best_f1


def parse_args():
    parser = argparse.ArgumentParser(description='Train the image-only cable-in-gripper classifier')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Dataset directory (or dataset.h5) from scripts/build_grasp_dataset.py')
    parser.add_argument('--ckpt_dir', type=str, default='logs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--image_size', type=int, default=IMAGE_SIZE)
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--val_frac', type=float, default=0.2, help='Fraction of EPISODES held out')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--roi', action='store_true',
                        help='Crop to the wrist-camera finger-gap ROI (agent/model/grasp.GRASP_ROI, '
                             'padded 40%%) instead of using the full frame')
    parser.add_argument('--no_pretrained', action='store_true', help='Train the trunk from scratch')
    parser.add_argument('--no_augment', action='store_true', help='Disable train-time augmentation')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.name is None:
        args.name = 'grasp-cls'
        print('Name not given. Assuming name for logging and wandb:', args.name)

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print(f'{device} unavailable, falling back to CPU')
        device = 'cpu'

    ckpt_path = pathlib.Path(args.ckpt_dir) / args.name
    if ckpt_path.exists():
        print(f'Checkpoint directory {ckpt_path} already exists. OVERWRITE.')
    print('Saving checkpoints to:', ckpt_path)

    train(name=args.name, dataset_path=args.data_dir, ckpt_dir=ckpt_path, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
          image_size=args.image_size, val_frac=args.val_frac, seed=args.seed,
          num_workers=args.num_workers, backbone=args.backbone, pretrained=not args.no_pretrained,
          augment=not args.no_augment, use_roi=args.roi, use_wandb=args.use_wandb, device=device)
