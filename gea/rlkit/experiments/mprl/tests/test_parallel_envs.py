import os
import time

import numpy as np
import robosuite as suite
import torch
from matplotlib import pyplot as plt
from robosuite.utils.transform_utils import *
from robosuite.wrappers.gym_wrapper import GymWrapper
from tqdm import tqdm

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs.wrappers.mujoco_vec_wrappers import StableBaselinesVecEnv
from rlkit.mprl.experiment import make_env
from rlkit.mprl.mp_env import MPEnv, RobosuiteEnv
from rlkit.torch.model_based.dreamer.visualization import make_video

if __name__ == "__main__":
    mp_env_kwargs = dict(
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
        check_com_grasp=False,
        terminate_on_success=False,
    )
    robosuite_args = dict(
        robots="Panda",
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
        use_object_obs=True,
        env_name="PickPlaceBread",
    )
    # OSC controller spec
    controller_args = dict(
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
    )
    robosuite_args["controller_configs"] = controller_args
    mp_env_kwargs["controller_configs"] = controller_args

    variant = dict(
        mp_env_kwargs=mp_env_kwargs,
        expl_environment_kwargs=robosuite_args,
        robosuite_env_kwargs=dict(),
        mprl=True,
    )

    num_envs = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count()))
    # num_envs = 25
    env_fns = [lambda: make_env(variant) for _ in range(num_envs)]
    env = StableBaselinesVecEnv(
        env_fns=env_fns,
        start_method="fork",
    )
    obs = env.reset()

    # time vec envs:
    num_steps = 10000
    num_steps_taken = 0
    start = time.time()
    for _ in tqdm(range(num_steps // num_envs)):
        action = np.random.normal(size=(num_envs, env.action_space.low.size))
        obs, reward, done, info = env.step(action)
        num_steps_taken += num_envs
        # if _ % 500 == 0:
        #     env.reset()
    end = time.time()
    print(
        f"Num Envs: {num_envs}, Frames per second: ",
        1 / ((end - start) / num_steps_taken),
    )
