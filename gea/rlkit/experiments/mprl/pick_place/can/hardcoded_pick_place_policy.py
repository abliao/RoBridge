import copy

import cv2
import numpy as np
import robosuite as suite
import torch
from matplotlib import pyplot as plt
from robosuite.controllers import controller_factory
from robosuite.utils.transform_utils import *
from robosuite.wrappers.gym_wrapper import GymWrapper
from tqdm import tqdm

import rlkit.torch.pytorch_util as ptu
from rlkit.mprl.mp_env import MPEnv, RobosuiteEnv, update_controller_config
from rlkit.torch.model_based.dreamer.visualization import make_video

if __name__ == "__main__":
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
    mp_env_kwargs = dict(
        vertical_displacement=0.1,
        teleport_instead_of_mp=True,
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
        controller_args=controller_args,
        verify_stable_grasp=True,
    )
    robosuite_args = dict(
        robots="Panda",
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
        use_object_obs=True,
        env_name="PickPlaceCan",
    )

    robosuite_args["controller_configs"] = controller_args
    env = suite.make(
        **robosuite_args,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        camera_names="frontview",
        camera_heights=1024,
        camera_widths=1024,
    )
    np.random.seed(0)
    env = MPEnv(GymWrapper(env), **mp_env_kwargs)
    num_episodes = 10
    total = 0
    ptu.device = torch.device("cuda")
    success_rate = 0
    frames = []
    for s in tqdm(range(num_episodes)):
        o = env.reset()
        target_pos = env.get_target_pos()
        rs = []
        # for i in range(350):
        #     a = np.concatenate(
        #         (
        #             env.sim.data.qpos[30:33] + np.array([0, 0, 0.1]) - env._eef_xpos,
        #             [0, 0, 0, -1],
        #         )
        #     )
        #     o, r, d, info = env.step(a)
        #     rs.append(r)
        #     env.render()

        for i in range(25):
            a = np.concatenate(([0, 0, -0.2], [0, 0, 0, -1]))
            o, r, d, info = env.step(a)
            rs.append(r)
            env.render()
        for i in range(50):
            a = np.concatenate(([0, 0, 0], [0, 0, 0, 1]))
            o, r, d, info = env.step(a)
            rs.append(r)
            env.render()
        # for i in range(50):
        #     a = np.concatenate(([0, 0, 0.2], [0, 0, 0, 1]))
        #     o, r, d, info = env.step(a)
        #     rs.append(r)
        #     env.render()
        # for i in range(300):
        #     a = np.concatenate((target_pos - env._eef_xpos, [0, 0, 0, 1]))
        #     o, r, d, info = env.step(a)
        #     rs.append(r)
        #     env.render()
        for i in range(50):
            a = np.concatenate(([0, 0, 0.2], [0, 0, 0, 1]))
            o, r, d, info = env.step(a)
            rs.append(r)
            env.render()
        for i in range(50):
            a = np.concatenate(([0, 0, -0.0], [0, 0, 0, -1]))
            o, r, d, info = env.step(a)
            rs.append(r)
            env.render()

        print(env._check_success())
        # plt.plot(rs)
        # plt.show()
        success_rate += env._check_success()
        # make_video(frames, "test", 0)
    print(f"Success Rate: {success_rate/num_episodes}")
