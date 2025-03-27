import os
import random
import time


env_names = [
    "metaworld_MT50",
    # "metaworld_MLV"
]
num_seeds = 3
pretrain = "/path/to/checkpoint"
for _ in range(num_seeds):
    for idx, env_name in enumerate(env_names):
        seed = random.randint(0, 100000000)
        run_name = f"metaworld_MT_eval_{env_name}"
        if "MT50" in env_name:
            cmd = f"python gea/eval.py experiment_subdir=metaworld_gea agent@_global_=IL pretrain.path={pretrain} use_wandb=False seed={seed} debug=True save_video=True num_train_frames=2000000 eval_every_frames=100000 num_eval_episodes=50 wandb.project_name=gea wandb.run_name={run_name} task@_global_={env_name} psl=True path_length=200 camera_name=corner2 action_repeat=2 experiment_id={run_name}  replay_buffer_size=750000 expert_schedule=\"\'linear(0.75, 0.25, 1000000)\'\" "
        elif "MLV" in env_name:
            cmd = f"python gea/eval.py experiment_subdir=metaworld_gea agent@_global_=IL pretrain.path={pretrain} use_wandb=False seed={seed} debug=True save_video=True num_train_frames=2000000 eval_every_frames=100000 num_eval_episodes=100 wandb.project_name=gea wandb.run_name={run_name} task@_global_={env_name} psl=True path_length=200 camera_name=corner2 action_repeat=2 experiment_id={run_name}  replay_buffer_size=750000 expert_schedule=\"\'linear(0.75, 0.25, 1000000)\'\" "
        else:
            cmd = f"python gea/eval.py experiment_subdir=metaworld_gea agent@_global_=IL pretrain.path={pretrain} use_wandb=False seed={seed} debug=True save_video=True num_train_frames=2000000 eval_every_frames=40000 num_eval_episodes=100 wandb.project_name=gea wandb.run_name={run_name} task@_global_={env_name} psl=True path_length=200 camera_name=corner2 action_repeat=2 experiment_id={run_name}  replay_buffer_size=750000"
        
        os.system(cmd)
    break
