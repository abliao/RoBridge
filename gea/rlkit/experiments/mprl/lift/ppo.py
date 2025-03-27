import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    setup_sweep_and_launch_exp,
)


def experiment(variant):
    from a2c_ppo_acktr.main import experiment

    experiment(variant)


from rlkit.torch.model_based.dreamer.experiments.arguments import get_args


def preprocess_variant(variant):
    return variant


if __name__ == "__main__":
    args = get_args()
    if args.debug:
        exp_prefix = "test" + args.exp_prefix
    else:
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm_kwargs=dict(
            entropy_coef=0.01,
            value_loss_coef=0.5,
            lr=3e-4,
            num_mini_batch=64,
            ppo_epoch=10,
            clip_param=0.2,
            eps=1e-5,
            max_grad_norm=0.5,
        ),
        rollout_kwargs=dict(
            use_gae=True,
            gamma=0.99,
            gae_lambda=0.95,
            use_proper_time_limits=True,
        ),
        env_kwargs=dict(
            robots="Panda",
            reward_shaping=True,
            control_freq=20,
            ignore_done=True,
            use_object_obs=True,
            env_name="Lift",
            horizon=500,
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
            robosuite_env_kwargs=dict(
                slack_reward=0,
                predict_done_actions=False,
                terminate_on_success=False,
            ),
            mp_env_kwargs=dict(),
            mprl=False,
        ),
        actor_kwargs=dict(recurrent=False, hidden_size=512, hidden_activation="relu"),
        num_processes=16,
        num_env_steps=int(1e7),
        num_steps=2048 // 16,
        log_interval=1,
        eval_interval=1,
        use_raw_actions=True,
        use_linear_lr_decay=False,
        env_name="RobosuiteLift",
        env_suite="robosuite",
    )
    setup_sweep_and_launch_exp(preprocess_variant, variant, experiment, args)
