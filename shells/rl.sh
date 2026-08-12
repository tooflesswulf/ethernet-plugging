python agent/rl_finetuning/train_residual_rl.py \
    --config-name=residual_td3_net_config \
    seed=0 \
    base_policy.ckpt='/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ckpts/c2r2-50-pose/ckpt_final.pth' \
    offline_data.num_episodes=2 \
    debug=true

    
