python agent/pretrain/train.py \
    --epochs 200 \
    --name rigid-follow-force-exp \
    --ckpt_dir ../ckpts \
    --data_dir ../dataset/rigid-follow-exp_dataset \
    --use_wandb

# Impedance policy (target pose + K/D/M) on FDCC data:
# python agent/pretrain/train.py \
#     --impedance \
#     --epochs 200 \
#     --name ethernet-impedance-force \
#     --ckpt_dir ../ckpts \
#     --data_dir ../data/ethernet-impedance_dataset
# Variants: --obs_fields pose gripper_width (no force), --obs_fields pose gripper_width log_stiffness (observe K)
