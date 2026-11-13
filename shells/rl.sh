#!/bin/bash
pwd

python experiments/zforce_rl_regulator.py \
    --insert_ckpt ../ckpts/ethernet-impedance-v2v3dag-insert-unet1d-offset/ckpt_final.pth \
    --offline_dataset_path ../dataset/ethernet-impedance-v2v3dag-insert_dataset/dataset.h5 \
    --use_wandb