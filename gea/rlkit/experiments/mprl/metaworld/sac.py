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
        max_path_length=500,
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
        mprl=False,
        algorithm="SAC",
        replay_buffer_size=int(1e7),
        seed=129,
        version="normal",
        plan_to_learned_goals=False,
        num_expl_envs=int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count())),
    )
    setup_sweep_and_launch_exp(preprocess_variant, variant, experiment, args)
