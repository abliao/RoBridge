import os
import random
import time

env_names = [
    # "metaworld_bin-picking-v2",
    # "metaworld_assembly-v2",
    # "metaworld_disassemble-v2",
    # "metaworld_button-press-topdown-v2",
    # "metaworld_button-press-topdown-wall-v2",
    # "metaworld_button-press-v2",
    # "metaworld_button-press-wall-v2",
    # "metaworld_coffee-button-v2",
    # "metaworld_box-close-v2",
    # "metaworld_dial-turn-v2",
    # "metaworld_door-close-v2",
    # "metaworld_door-open-v2",
    # "metaworld_door-lock-v2",
    
    # "metaworld_coffee-push-v2",
    # "metaworld_coffee-pull-v2",
    # "metaworld_drawer-close-v2",
    # "metaworld_drawer-open-v2",
    # "metaworld_faucet-close-v2",
    # "metaworld_faucet-open-v2",
    # "metaworld_handle-press-side-v2",
    # "metaworld_handle-press-v2",
    # "metaworld_handle-pull-side-v2", 
    # "metaworld_lever-pull-v2",
    # "metaworld_peg-insert-side-v2",
    # "metaworld_hand-insert-v2",
    # "metaworld_handle-pull-v2",
    # "metaworld_pick-place-v2",
    # "metaworld_pick-place-wall-v2",
    # "metaworld_reach-v2",
    # "metaworld_plate-slide-v2",
    # "metaworld_plate-slide-side-v2",
    # "metaworld_plate-slide-back-v2",
    # "metaworld_plate-slide-back-side-v2",
    # "metaworld_push-v2",
    # "metaworld_push-back-v2",
    # "metaworld_peg-unplug-sssside-v2",
    # "metaworld_stick-pull-v2",
    # "metaworld_stick-push-v2",
    # "metaworld_reach-wall-v2",
    # "metaworld_soccer-v2",
    # "metaworld_sweep-into-v2",
    # "metaworld_push-wall-v2",
    # "metaworld_window-open-v2",
    # "metaworld_window-close-v2",
    # "metaworld_shelf-place-v2",
    # "metaworld_sweep-v2",
    # "metaworld_basketball-v2", 
    # "metaworld_pick-out-of-hole-v2",
    
    # "metaworld_hammer-v2", 
    # "metaworld_door-unlock-v2",

    "metaworld_MT50",
    "metaworld_MLV"
]
num_seeds = 3
for _ in range(num_seeds):
    for idx, env_name in enumerate(env_names):
        seed = random.randint(0, 100000000)
        run_name = f"metaworld_MT_test_{env_name}"
        if "MT50" in env_name:
            cmd = f"python gea/dagger_train.py experiment_subdir=metaworld_gea agent@_global_=IL use_wandb=True seed={seed} debug=True save_video=True num_train_frames=2000000 eval_every_frames=100000 num_eval_episodes=100 wandb.project_name=gea wandb.run_name={run_name} task@_global_={env_name} psl=True path_length=200 camera_name=corner2 action_repeat=2 experiment_id={run_name}  replay_buffer_size=750000 expert_schedule=\"\'linear(0.75, 0.25, 1000000)\'\" "
        elif "MLV" in env_name:
            cmd = f"python gea/dagger_train.py experiment_subdir=metaworld_gea lr=1e-4 num_seed_frames=100000 expert=drqv2 agent@_global_=IL expert_schedule=\"\'linear(1.0, 1.0, 10000000)\'\" use_wandb=True seed={seed} debug=True save_video=True num_train_frames=2000000 eval_every_frames=40000 num_eval_episodes=100 wandb.project_name=gea wandb.run_name={run_name} task@_global_={env_name} psl=True path_length=200 camera_name=corner2 action_repeat=2 experiment_id={run_name}  replay_buffer_size=750000 "
        else:
            cmd = f"python gea/dagger_train.py experiment_subdir=metaworld_gea lr=1e-4 num_seed_frames=4000 expert=drqv2 agent@_global_=IL expert_schedule=\"\'linear(1.0, 0.25, 1000000)\'\" use_wandb=True seed={seed} debug=True save_video=True num_train_frames=2000000 eval_every_frames=40000 num_eval_episodes=30 wandb.project_name=gea wandb.run_name={run_name} task@_global_={env_name} psl=True path_length=200 camera_name=corner2 action_repeat=2 experiment_id={run_name}  replay_buffer_size=750000  "
        os.system(cmd)
    break
