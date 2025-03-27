import os

from rlkit.mprl.experiment import experiment, preprocess_variant
from rlkit.torch.model_based.dreamer.experiments.arguments import get_args
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    setup_sweep_and_launch_exp,
)

if __name__ == "__main__":
    # noinspection PyTypeChecker
    args = get_args()
    variant = dict(
        max_path_length=250,
        additional_reward_configs=dict(
            use_distance_reduced_to_object_reward=False,
            use_min_prev_distance=False,
            dist_reduced_reward_scale=1,
            first_grasp_reward=False,
        ),
        algorithm_kwargs=dict(
            batch_size=256,
            min_num_steps_before_training=3300,
            num_epochs=5000000,
            num_eval_steps_per_epoch=2500,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            num_train_loops_per_epoch=10,
            max_path_length=500,
        ),
        robosuite_env_kwargs=dict(
            slack_reward=0,
            predict_done_actions=False,
            terminate_on_success=False,
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
            env_name="Lift",
            horizon=500,
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
        mprl=False,
        algorithm="SAC",
        replay_buffer_size=int(5e6),
        seed=129,
        version="normal",
        plan_to_learned_goals=False,
        num_expl_envs=int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count())),
    )
    setup_sweep_and_launch_exp(preprocess_variant, variant, experiment, args)
