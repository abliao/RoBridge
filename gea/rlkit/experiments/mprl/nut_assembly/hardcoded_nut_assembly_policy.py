import cv2
import numpy as np
import robosuite as suite
import torch
from matplotlib import pyplot as plt
from robosuite.utils.transform_utils import *
from robosuite.wrappers.gym_wrapper import GymWrapper
from tqdm import tqdm

import rlkit.torch.pytorch_util as ptu
from rlkit.mprl.mp_env import MPEnv
from rlkit.torch.model_based.dreamer.visualization import make_video

if __name__ == "__main__":
    mp_env_kwargs = dict(
        vertical_displacement=0.04,
        teleport_instead_of_mp=True,
        use_joint_space_mp=False,
        randomize_init_target_pos=False,
        mp_bounds_low=(-1.45, -1.25, 0.45),
        mp_bounds_high=(0.45, 0.85, 2.25),
        backtrack_movement_fraction=0.001,
        clamp_actions=True,
        update_with_true_state=True,
        grip_ctrl_scale=0.0025,
        planning_time=20,
        hardcoded_high_level_plan=True,
        terminate_on_success=False,
        plan_to_learned_goals=False,
        reset_at_grasped_state=False,
        verify_stable_grasp=True,
        hardcoded_orientations=True,
        use_pcd_collision_check=False,
        use_vision_pose_estimation=True,
        use_vision_placement_check=True,
        use_vision_grasp_check=True,
    )
    robosuite_args = dict(
        robots="Panda",
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
        use_object_obs=True,
        env_name="NutAssemblySquare",
        reward_scale=2.0,
        horizon=500,
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
    env = suite.make(
        **robosuite_args,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names="frontview",
        camera_heights=480,
        camera_widths=640,
    )
    env = MPEnv(GymWrapper(env), **mp_env_kwargs)
    num_episodes = 10
    total = 0
    ptu.device = torch.device("cuda")
    np.random.seed(1)
    frames = []
    all_success_rates = []
    for seed in range(3):
        np.random.seed(seed)
        env.intermediate_frames = []
        success_rate = 0
        for s in tqdm(range(num_episodes)):
            o = env.reset(get_intermediate_frames=True)
            if len(env.intermediate_frames) > 0:
                for frame in env.intermediate_frames:
                    frames.append(frame)
                env.intermediate_frames = []
            rs = []

            for i in range(25):
                a = np.concatenate(([0, 0, -0.2], [0, 0, 0, -1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                # env.render()
                frames.append(env.get_image())
            for i in range(15):
                a = np.concatenate(([0, 0, 0], [0, 0, 0, 1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                # env.render()
                frames.append(env.get_image())
            for i in range(20):
                a = np.concatenate(([0, 0, 0.1], [0, 0, 0, 1]))
                o, r, d, info = env.step(a, get_intermediate_frames=True)
                if len(env.intermediate_frames) > 0:
                    for frame in env.intermediate_frames:
                        frames.append(frame)
                    env.intermediate_frames = []
                rs.append(r)
                # env.render()
                frames.append(env.get_image())
            cv2.imwrite('square_wide.png', frames[-1][:, :, ::-1])
            exit()
            for i in range(40):
                a = np.concatenate(([0, 0, -0.3], [0, 0, 0, 1]))
                o, r, d, info = env.step(a)
                rs.append(r)
                # env.render()
                frames.append(env.get_image())
            for i in range(10):
                a = np.concatenate(([0, 0, 0], [0, 0, 0, -1]))
                o, r, d, info = env.step(a, get_intermediate_frames=True)
                if len(env.intermediate_frames) > 0:
                    for frame in env.intermediate_frames:
                        frames.append(frame)
                    env.intermediate_frames = []
                rs.append(r)
                # env.render()
                frames.append(env.get_image())

            # for i in range(25):
            #     a = np.concatenate(([0, 0, -0.2], [0, 0, 0, -1]))
            #     o, r, d, info = env.step(a)
            #     rs.append(r)
            #     # env.render()
            #     frames.append(env.get_image())
            # for i in range(15):
            #     a = np.concatenate(([0, 0, 0], [0, 0, 0, 1]))
            #     o, r, d, info = env.step(a)
            #     rs.append(r)
            #     # env.render()
            #     frames.append(env.get_image())
            # for i in range(10):
            #     a = np.concatenate(([0, 0, 0.1], [0, 0, 0, 1]))
            #     o, r, d, info = env.step(a, get_intermediate_frames=True)
            #     if len(env.intermediate_frames) > 0:
            #         for frame in env.intermediate_frames:
            #             frames.append(frame)
            #         env.intermediate_frames = []
            #     rs.append(r)
            #     # env.render()
            #     frames.append(env.get_image())
            # for i in range(40):
            #     a = np.concatenate(([0, 0, -0.3], [0, 0, 0, 1]))
            #     o, r, d, info = env.step(a)
            #     rs.append(r)
            #     # env.render()
            #     frames.append(env.get_image())
            # for i in range(10):
            #     a = np.concatenate(([0, 0, 0], [0, 0, 0, -1]))
            #     o, r, d, info = env.step(a)
            #     rs.append(r)
            #     # env.render()
            #     frames.append(env.get_image())
            print(env._check_success())
            plt.plot(rs)
            # plt.savefig(f"plots/{s}.png")
            success_rate += env._check_success()
            print("Running success rate: ", success_rate / (s + 1))
        print(f"Success Rate: {success_rate/num_episodes}")
        all_success_rates.append(success_rate / num_episodes)
    print("Mean success", np.mean(all_success_rates))
    print("Std success", np.std(all_success_rates))
    make_video(frames, "videos", 2, use_wandb=False)
