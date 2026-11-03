python agent/pretrain/train.py \
    --epochs 200 \
    --name plugin_unplug_75_local_delta \
    --ckpt_dir /zfsauton/scratch/yiqiw2/100%/ckpts \
    --data_dir /zfsauton/scratch/yiqiw2/100%/datasets/ethernet_plugin_unplug_dataset \
    --use_wandb

# # /zfsauton/scratch/yiqiw2/100%/datasets/ethernet_plugin_unplug_rl_dataset \

# Impedance policy (target pose + K/D/M) on FDCC data:
# python agent/pretrain/train.py \
#     --impedance \
#     --epochs 200 \
#     --name ethernet-impedance-force \
#     --ckpt_dir ../ckpts \
#     --data_dir ../data/ethernet-impedance_dataset
# Variants: --obs_fields pose gripper_width (no force), --obs_fields pose gripper_width log_stiffness (observe K)
