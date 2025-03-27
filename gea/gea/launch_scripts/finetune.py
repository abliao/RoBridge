import os
import random
import time

env_names = [
    "metaworld_bin-picking-v2",
]
num_seeds = 3
pretrain = "/path/to/checkpoint"
seed = random.randint(0, 100000000)
run_name = f"real_world_finetune"

cmd = f"python gea/finetune.py experiment_subdir=metaworld_gea lr=1e-4 pretrain.path={pretrain} num_seed_frames=4000 expert=drqv2 agent@_global_=IL expert_schedule=\"\'linear(1.0, 0.25, 1000000)\'\" use_wandb=True seed={seed} debug=True save_video=True num_train_frames=10000000 eval_every_frames=40000 num_eval_episodes=30 wandb.project_name=gea wandb.run_name={run_name} task@_global_={env_name} psl=True path_length=200 camera_name=corner2 action_repeat=2 experiment_id={run_name}  replay_buffer_size=750000  "    
os.system(cmd)