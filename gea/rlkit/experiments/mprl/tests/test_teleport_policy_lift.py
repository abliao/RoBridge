import pickle
import random

import cv2
import imageio
import numpy as np
import robosuite as suite
import torch
from robosuite.controllers import controller_factory
from robosuite.controllers.controller_factory import load_controller_config
from robosuite.utils.transform_utils import *
from robosuite.wrappers.gym_wrapper import GymWrapper
from tqdm import tqdm

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv
from rlkit.mprl.mp_env import (
    MPEnv,
    apply_controller,
    set_robot_based_on_ee_pos,
    update_controller_config,
)
from rlkit.torch.model_based.dreamer.visualization import make_video
from rlkit.torch.sac.policies import MakeDeterministic

if __name__ == "__main__":
    robosuite_args = dict(
        robots="Panda",
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
        use_object_obs=True,
        env_name="PickPlaceBread",
        horizon=100,
    )
    # OSC controller spec
    controller_configs = dict(
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
    robosuite_args["controller_configs"] = controller_configs
    mp_env_kwargs = {
        "backtrack_movement_fraction": 0.001,
        "clamp_actions": True,
        "grip_ctrl_scale": 0.0025,
        "mp_bounds_high": [0.45, 0.85, 2.25],
        "mp_bounds_low": [-1.45, -1.25, 0.45],
        "plan_to_learned_goals": False,
        "planning_time": 20,
        "randomize_init_target_pos": False,
        "teleport_on_grasp": True,
        "teleport_instead_of_mp": True,
        "update_with_true_state": True,
        "vertical_displacement": 0.04,
        "controller_configs": controller_configs,
    }
    env = suite.make(
        **robosuite_args,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names="frontview",
        camera_heights=1024,
        camera_widths=1024,
    )
    env = MPEnv(GymWrapper(env), **mp_env_kwargs)
    num_episodes = 10
    total = 0
    load_path = "/home/mdalal/research/mprl/rlkit/data/10-29-sac-mprl-pick-place-teleport-on-grasp-multi-step-policy-better-grasp-v1/10-29-sac_mprl_pick_place_teleport_on_grasp_multi_step_policy_better_grasp_v1_2022_10_29_22_06_35_0000--s-35851/policy_450.pkl"
    policy = pickle.load(open(load_path, "rb"))
    policy = MakeDeterministic(policy)
    ptu.device = torch.device("cuda")
    success_rate = 0
    frames = []

    # seed all random generators:
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    for s in tqdm(range(num_episodes)):
        policy.reset()
        o = env.reset()
        for i in tqdm(range(100)):
            a, _ = policy.get_action(o)
            o, r, d, _ = env.step(a)
            frames.append(env.get_image())
            if d:
                print("ended early")
                break
        print(env._check_success())
        success_rate += env._check_success()
    print(f"Success Rate: {success_rate/num_episodes}")
    video_path = "test.mp4"
    video_writer = imageio.get_writer(video_path, fps=20)
    for frame in frames:
        video_writer.append_data(frame[:, :, ::-1])
    video_writer.close()
