import os
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agent.dataset.sequence import FailureDetectDataset
from agent.model.networks import MultiEncoder


def build_dataloader(dataset_path, label_file, batch_size, shuffle):
    dataset = FailureDetectDataset(dataset_path, label_file)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    use_wandb=False,
    log_interval=10,
    global_step=0,
):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, leave=False)

    for i, (data, label) in enumerate(pbar):
        image = data["image"].to(device); force = data["force"].to(device); label = label.float().to(device)
        gripper = data["gripper_width"].unsqueeze(-1)
        pose = torch.cat([data["pose"], gripper], dim=-1).to(device)
        optimizer.zero_grad()
        out = model(image, pose, force)
        loss = criterion(out["logits"], label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_description(f"loss={loss.item():.4f}")

        ##################################################
        # WandB logging
        ##################################################
        if use_wandb and (global_step % log_interval == 0):
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "global_step": global_step,
                },
                step=global_step,
            )

        global_step += 1

    avg_loss = total_loss / len(dataloader)

    return avg_loss, global_step


@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0; total_correct = 0; total_samples = 0
    class_correct = [0, 0]; class_total = [0, 0]

    for data, label in tqdm(dataloader, leave=False):
        label = label.float().to(device)
        image = data["image"].to(device); gripper = data["gripper_width"].unsqueeze(-1)
        pose = torch.cat([data["pose"], gripper], dim=-1).to(device); force = data["force"].to(device)

        out = model(image, pose, force)
        logits = out["logits"]
        loss = criterion(logits, label)
        total_loss += loss.item()
        ###########################
        # Accuracy
        ###########################

        pred = (torch.sigmoid(logits) > 0.5).long()
        gt = label.long()
        correct = (pred == gt)

        total_correct += correct.sum().item()
        total_samples += gt.numel()

        for cls in [0, 1]:
            mask = (gt == cls)
            class_total[cls] += mask.sum().item()
            if mask.any():
                class_correct[cls] += correct[mask].sum().item()

    avg_loss = total_loss / len(dataloader)
    overall_acc = total_correct / max(total_samples, 1)
    class_acc = []
    for cls in [0, 1]:
        if class_total[cls] == 0:
            class_acc.append(0.0)
        else:
            class_acc.append(class_correct[cls] / class_total[cls])

    return {
        "validate/loss": avg_loss,
        "validate/overall_acc": overall_acc,
        "validate/acc_0": class_acc[0],
        "validate/acc_1": class_acc[1],
    }

def main():

    ##############################################
    # Config
    ##############################################
    use_wandb = True
    dataset_path = "/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ethernet-plugging/logs-collectfailures/75rl-rtc"
    train_label = os.path.join(dataset_path, "train_label.csv"); val_label = os.path.join(dataset_path, "val_label.csv")
    modality = ['image', 'pose', 'force']
    name = '_'.join([ m[:2] for m in modality])
    batch_size = 64
    lr = 5e-4
    epochs = 30
    global_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################################
    # WandB
    ##############################################

    if use_wandb:
        wandb.init(
            project="cable_failure_detection",
            name = name,
            config={
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
            },
        )

    ##############################################
    # Data
    ##############################################

    train_loader = build_dataloader(
        dataset_path,
        train_label,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = build_dataloader(
        dataset_path,
        val_label,
        batch_size=batch_size,
        shuffle=False,
    )
    ##############################################
    # Model
    ##############################################

    model = MultiEncoder(modality = modality).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    ##############################################
    # Training Loop
    ##############################################

    best_val_loss = float("inf")

    for epoch in range(epochs):

        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_wandb=use_wandb,
            global_step=global_step,
        )

        val_metrics  = validate(
            model,
            val_loader,
            criterion,
            device, )
        val_loss = val_metrics['validate/loss']
        print(
            f"[{epoch+1:03d}/{epochs}] "
            f"train={train_loss:.4f} "
            f"val={val_loss:.4f}"
        )

        if use_wandb:
            log_dict = {   "epoch": epoch,}
            log_dict.update(val_metrics)
            wandb.log( log_dict )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ckpts/failure_detect/best_model.pt",
            )

    print("Finished Training.")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()