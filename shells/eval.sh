#!/bin/bash
# python agent/evaluate/eval_realtime.py \
#     --ckpt '/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ckpts/ethernet_plugin_unplug_75/h16/ckpt_final.pth' \
#     --log_dir './logs-collectfailures/75rl-rtc' \
    # --mode realtime

# python agent/evaluate/eval_force.py \
#     --ckpt '/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ckpts/forcepose-up-exp-h16/ckpt_final.pth' \
#     --log_dir './logs-force-exp/test' \

python agent/evaluate/eval.py \
    --ckpt '/home/atkesonlab4/Desktop/YiqiProject/100%_Project/ckpts/follow-line-exp-h16/albert/ckpt_ep_280.pth' \
    --log_dir './logs-follow-line/v0' \