python agent/pretrain/train.py \
    --epochs 200 \
<<<<<<< HEAD
<<<<<<< HEAD
    --name plugin_unplug_75_local_delta \
    --ckpt_dir /zfsauton/scratch/yiqiw2/100%/ckpts \
    --data_dir /zfsauton/scratch/yiqiw2/100%/datasets/ethernet_plugin_unplug_dataset \
=======
    --name plugin-unplug-pos-local_delta-50 \
    --ckpt_dir ../ckpts \
    --data_dir ../data/ethernet_plugin_unplug_dataset \
>>>>>>> 0e1c919e0e8cb0e50b9ad45919e536b9f693f639
    --use_wandb

# # /zfsauton/scratch/yiqiw2/100%/datasets/ethernet_plugin_unplug_rl_dataset \
    
=======
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
>>>>>>> 80f13d40214f7c5ae29ea7a254702d8f9c27fc6d
