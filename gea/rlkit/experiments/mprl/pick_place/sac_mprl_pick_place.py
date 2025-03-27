import os

import numpy as np

from rlkit.mprl.experiment import experiment, preprocess_variant_mp
from rlkit.torch.model_based.dreamer.experiments.arguments import get_args
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    setup_sweep_and_launch_exp,
)

if __name__ == "__main__":
    # noinspection PyTypeChecker
    args = get_args()
    variant = dict(
        algorithm_kwargs=dict(
            batch_size=128,
            min_num_steps_before_training=3300,
            num_epochs=1000000,
            num_eval_steps_per_epoch=2500,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
        ),
        controller_configs=dict(
            type="OSC_POSE",
            input_max=1,
            input_min=-1,
            output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            kp=150,
            damping=1,
            impedance_mode="fixed",
            kp_limits=[0, 300],
            damping_limits=[0, 10],
            position_limits=None,
            orientation_limits=None,
            uncouple_pos_ori=True,
            control_delta=True,
            interpolation=None,
            ramp_ratio=0.2,
        ),
        environment_kwargs=dict(
            robots="Panda",
            reward_shaping=True,
            control_freq=20,
            ignore_done=True,
            use_object_obs=True,
            env_name="PickPlaceCan",
        ),
        mp_env_kwargs=dict(
            vertical_displacement=0.08,
            teleport_instead_of_mp=True,
            randomize_init_target_pos=False,
            mp_bounds_low=(-1.45, -1.25, 0.45),
            mp_bounds_high=(0.45, 0.85, 2.25),
            backtrack_movement_fraction=0.001,
            clamp_actions=True,
            update_with_true_state=True,
            grip_ctrl_scale=0.0025,
            planning_time=20,
            teleport_on_grasp=True,
            verify_stable_grasp=True,
        ),
        policy_kwargs=dict(hidden_sizes=(256, 256)),
        qf_kwargs=dict(hidden_sizes=(256, 256)),
        trainer_kwargs=dict(
            discount=0.99,
            policy_lr=0.001,
            qf_lr=0.0005,
            reward_scale=1.0,
            soft_target_tau=0.005,
            target_update_period=5,
            use_automatic_entropy_tuning=True,
        ),
        planner_trainer_kwargs=dict(
            discount=0.5,
            policy_lr=0.001,
            qf_lr=0.0005,
            reward_scale=1.0,
            soft_target_tau=0.005,
            target_update_period=5,
            use_automatic_entropy_tuning=True,
        ),
        mprl=True,
        algorithm="MPRL-SAC",
        max_path_length=25,
        num_ll_actions_per_hl_action=25,
        num_hl_actions_total=2,
        replay_buffer_size=int(1e7),
        seed=np.random.randint(0, 1000000),
        version="normal",
        plan_to_learned_goals=False,
        num_expl_envs=int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count())),
        planner_num_trains_per_train_loop=1000,
        control_path=None,
        planner_path=None,
    )
    setup_sweep_and_launch_exp(preprocess_variant_mp, variant, experiment, args)
