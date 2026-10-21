#!/bin/bash
# Actual-pose policies with explicit admittance targets on the insert phase, as the
# ethernet-impedance-v2v3-insert-{unet1d-noimg,cross}-{pose,offset} runs but with
#   _act=pvf: pose + twist + wrench + impedance gains
#   _act=pf:  pose + wrench + impedance gains (twist from the target's motion at eval)
# Dataset: scripts/split_dataset.py on v2 + v3 after rawdata_to_dataset.py --labels_only.
# unet1d = ResNet18 image features + FiLM unet1d; unet1d-noimg = state only.
# Run from the repo root; each run logs to ../ckpts/<name>.log.
DATA=../data/ethernet-impedance-v2v3fv-insert_dataset
PREFIX=ethernet-impedance-v2v3-insert
COMMON="--pose_target pose --action_anchor pose --impedance_fields stiffness damping mass
        --epochs 200 --max_episodes 1000 --ckpt_dir ../ckpts --data_dir $DATA --use_wandb"

declare -A ARCH=(
    [unet1d]="--encoder_type resnet --noise_net unet --obs_horizon 4"         # ResNet18 + FiLM unet1d
    [unet1d-noimg]="--encoder_type none --noise_net unet --obs_horizon 4"     # state only
    [cross]="--encoder_type vit"
)
declare -A OBS=(
    [pose]="pose gripper_width"
    [offset]="pose gripper_width target_offset log_stiffness"
    [force]="pose gripper_width force"
)
# Usage: bash shells/train_fv.sh [archs] [obs], e.g.
#   bash shells/train_fv.sh "unet1d unet1d-noimg cross" "pose offset"   (default)
#   bash shells/train_fv.sh "unet1d cross" force
ARCHS=${1:-"unet1d unet1d-noimg cross"}
OBSS=${2:-"pose offset"}
declare -A ACT=(
    [pvf]="twist wrench"
    [pf]="wrench"
)

for arch in $ARCHS; do
    for obs in $OBSS; do
        for act in pvf pf; do
            name="$PREFIX-$arch-${obs}_act=$act"
            nohup python agent/pretrain/train.py $COMMON ${ARCH[$arch]} \
                --obs_fields ${OBS[$obs]} --feedforward_fields ${ACT[$act]} \
                --name "$name" > "../ckpts/$name.log" 2>&1 &
            echo "started $name (pid $!)"
        done
    done
done
