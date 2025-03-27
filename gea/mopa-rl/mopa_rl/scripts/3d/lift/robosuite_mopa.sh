#!/bin/bash -x
gpu=$1
seed=$2
prefix="Lift-MoPA"
env="LiftMoPA"
algo='sac'
debug="False"
log_root_dir="./logs"
vis_replay="True"
plot_type='3d'
reward_scale="1.0" # doesn't matter since i'm getting the reward directly from robosuite env.
use_ik_target="False"
ik_target="grip_site"
action_range="0.001"
lr_actor="0.001"
lr_critic="0.0005"
invalid_target_handling="True"
wandb="True"
mopa="True"
omega="0.5"
entity="tchiruvolu"
project="moparl_tests"
reward_type="dense"
vis_replay="False"
num_record_samples="5"

python -m rl.main \
    --log_root_dir $log_root_dir \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --debug $debug \
    --algo $algo \
    --seed $seed \
    --vis_replay $vis_replay \
    --plot_type $plot_type \
    --reward_scale $reward_scale \
    --use_ik_target $use_ik_target \
    --ik_target $ik_target \
    --action_range $action_range \
    --lr_actor $lr_actor \
    --lr_critic $lr_critic \
    --wandb $wandb \
    --entity $entity \
    --project $project \
    --reward_type $reward_type \
    --vis_replay $vis_replay \
    --num_record_samples $num_record_samples \
    --omega $omega \
    --mopa $mopa \
    --invalid_target_handling $invalid_target_handling
