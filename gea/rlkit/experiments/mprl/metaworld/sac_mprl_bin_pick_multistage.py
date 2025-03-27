import os

import numpy as np

from rlkit.mprl.experiment import multi_stage_modular_experiment, preprocess_variant_mp
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
            num_epochs=10000,
            num_eval_steps_per_epoch=2500,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
        ),
        env_suite="metaworld",
        environment_kwargs=dict(
            env_name="bin-picking-v2",
            env_kwargs=dict(
                reward_type="dense",
                usage_kwargs=dict(
                    use_dm_backend=False,
                    use_raw_action_wrappers=False,
                    use_image_obs=False,
                    max_path_length=500,
                    unflatten_images=False,
                ),
                imwidth=480,
                imheight=480,
                action_space_kwargs=dict(
                    control_mode="end_effector",
                    action_scale=1 / 100,
                ),
            ),
        ),
        mp_env_kwargs=dict(
            vertical_displacement=0.05,
            teleport_instead_of_mp=True,
            randomize_init_target_pos=False,
            mp_bounds_low=(-0.2, 0.6, 0.0),
            mp_bounds_high=(0.2, 0.8, 0.2),
            backtrack_movement_fraction=0.001,
            clamp_actions=True,
            update_with_true_state=True,
            grip_ctrl_scale=0.0025,
            planning_time=20,
            verify_stable_grasp=True,
            teleport_on_grasp=True,
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
        max_path_length=200,
        num_ll_actions_per_hl_action=100,
        num_hl_actions_total=2,
        replay_buffer_size=int(1e7),
        seed=np.random.randint(0, 1000000),
        version="normal",
        plan_to_learned_goals=True,
        num_expl_envs=int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count())),
        planner_num_trains_per_train_loop=1000,
        control_path=None,
        planner_path=None,
    )
    setup_sweep_and_launch_exp(
        preprocess_variant_mp, variant, multi_stage_modular_experiment, args
    )
