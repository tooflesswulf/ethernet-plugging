# python agent/pretrain/train_force.py \
#     --epochs 1000 \
#     --name forcepose-up-exp-h16 \
#     --ckpt_dir ../ckpts \
#     --data_dir ../dataset/force-up-exp \
#     --use_wandb

python agent/pretrain/train.py \
    --epochs 1000 \
    --name follow-line-exp-h16 \
    --ckpt_dir ../ckpts \
    --data_dir ../dataset/follow-line-exp_dataset \
    --use_wandb
    
