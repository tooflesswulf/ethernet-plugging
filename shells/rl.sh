# debug=true will stop wandb
python agent/rl_finetuning/train_residual_rl.py \
    --config-name=residual_td3_net_config \
    seed=0 \
    base_policy.ckpt='/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ckpts/rigid-follow-force-exp/ckpt_final.pth' \
    offline_data.num_episodes=50 \
    debug=true # true will disable wandb
