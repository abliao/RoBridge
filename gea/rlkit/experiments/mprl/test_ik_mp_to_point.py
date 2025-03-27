import cv2
import matplotlib.pyplot as plt
import numpy as np
import robosuite as suite
from robosuite.controllers import controller_factory
from robosuite.controllers.controller_factory import load_controller_config
from robosuite.controllers.ik import InverseKinematicsController
from robosuite.utils.control_utils import orientation_error
from robosuite.utils.transform_utils import (
    axisangle2quat,
    euler2mat,
    mat2quat,
    quat2mat,
    quat_conjugate,
    quat_distance,
    quat_multiply,
)
from robosuite.wrappers.gym_wrapper import GymWrapper
from tqdm import tqdm

from rlkit.core import logger
from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv
from rlkit.mprl.mp_env import MPEnv, mp_to_point, mp_to_point_fast


def make_video(frames, logdir, epoch):
    height, width, _ = frames[0].shape
    size = (width, height)

    out = cv2.VideoWriter(
        logdir + "/" + f"viz_{epoch}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 60, size
    )

    for frame in frames:
        out.write(frame)
    out.release()


def create_env():
    environment_kwargs = {
        "control_freq": 20,
        "controller": "OSC_POSE",
        "env_name": "Lift",
        "hard_reset": False,
        "ignore_done": True,
        "reward_scale": 1.0,
        "robots": "Panda",
    }
    controller = environment_kwargs.pop("controller")
    controller_config = load_controller_config(default_controller=controller)
    env = suite.make(
        **environment_kwargs,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=True,
        controller_configs=controller_config,
        camera_names="frontview",
        camera_heights=256,
        camera_widths=256,
    )
    mp_env_kwargs = {
        "vertical_displacement": 0.04,
        "teleport_position": False,
        "plan_to_learned_goals": True,
        "planning_time": 20,
        "mp_bounds_low": (-1.45, -1.25, 0.45),
        "mp_bounds_high": (0.45, 0.85, 2.25),
        "update_with_true_state": True,
    }
    env = MPEnv(NormalizedBoxEnv(GymWrapper(env)), **mp_env_kwargs)
    return env


def update_controller_config(env, controller_config):
    controller_config["robot_name"] = env.robots[0].name
    controller_config["sim"] = env.robots[0].sim
    controller_config["eef_name"] = env.robots[0].gripper.important_sites["grip_site"]
    controller_config["eef_rot_offset"] = env.robots[0].eef_rot_offset
    controller_config["joint_indexes"] = {
        "joints": env.robots[0].joint_indexes,
        "qpos": env.robots[0]._ref_joint_pos_indexes,
        "qvel": env.robots[0]._ref_joint_vel_indexes,
    }
    controller_config["actuator_range"] = env.robots[0].torque_limits
    controller_config["policy_freq"] = env.robots[0].control_freq
    controller_config["ndim"] = len(env.robots[0].robot_joints)


def check_valid(env, pos, quat):
    ik_controller_config = env.ik_controller_config.copy()
    update_controller_config(env, ik_controller_config)
    ctrl = controller_factory("IK_POSE", ik_controller_config)
    ctrl.sync_state()
    cur_rot_inv = quat_conjugate(env._eef_xquat.copy())
    rot_diff = quat2mat(quat_multiply(quat, cur_rot_inv))
    joint_pos = ctrl.joint_positions_for_eef_command(pos - env._eef_xpos, rot_diff)
    env.robots[0].set_robot_joint_positions(joint_pos)
    assert (env.sim.data.qpos[:7] - joint_pos).sum() < 1e-10
    error = np.linalg.norm(env._eef_xpos - pos)
    return error < 0.01


"""
This is testing being able to do random positions on the lifting task
- it should mostly work with linear interpolation.
"""


def test_random_positions(num_tests):
    np.random.seed(0)
    env = create_env()
    env.reset()
    errors = []
    valid_cnt = 0
    for _ in tqdm(range(num_tests)):
        rand_quat = env.reset_ori
        rand_pos = env._eef_xpos + np.random.randn(3) / 10
        if not check_valid(env, rand_pos, rand_quat):
            continue
        valid_cnt += 1
        # do validity checking
        env.reset()
        mp_to_point_fast(
            env,
            env.ik_controller_config,
            env.osc_controller_config,
            np.concatenate((rand_pos, env.reset_ori)).astype(np.float64),
            qpos=env.reset_qpos,
            qvel=env.reset_qvel,
            grasp=False,
            ignore_object_collision=False,
            planning_time=5,
            # planning_time=env.planning_time,
            get_intermediate_frames=True,
        )
        errors.append(np.linalg.norm(rand_pos - env._eef_xpos))
    plt.hist(errors)
    plt.savefig("errors_lift.png")
    print(f"Valid cnt: {valid_cnt}")
    print(f"Average error: {np.mean(errors)}")
    return


def test_pick_place(num_tests):
    return


if __name__ == "__main__":
    environment_kwargs = {
        "control_freq": 20,
        "controller": "OSC_POSE",
        "env_name": "PickPlaceBread",
        "hard_reset": False,
        "ignore_done": True,
        "reward_scale": 1.0,
        "robots": "Panda",
        "initialization_noise": {"magnitude": 0, "type": "gaussian"},
    }
    controller = environment_kwargs.pop("controller")
    controller_config = load_controller_config(default_controller=controller)
    env = suite.make(
        **environment_kwargs,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=True,
        controller_configs=controller_config,
        camera_names="frontview",
        camera_heights=512,
        camera_widths=512,
    )
    mp_env_kwargs = {
        "vertical_displacement": 0.04,
        "teleport_position": False,
        "plan_to_learned_goals": True,
        "planning_time": 20,
        "mp_bounds_low": (-1.45, -1.25, 0.45),
        "mp_bounds_high": (0.45, 0.85, 2.25),
        "update_with_true_state": True,
    }
    env = MPEnv(NormalizedBoxEnv(GymWrapper(env)), **mp_env_kwargs)
    # logger.set_snapshot_dir(
    #     "/home/mdalal/research/mprl/rlkit/data/controller_debugging"
    # )
    num_steps = 1
    total = 0
    np.random.seed(0)
    env.seed(0)
    planning_time = 7
    mp_fast_dists = []
    mp_slow_dists = []
    target_second_pos = np.array([0.03056791, 0.36607588, 0.87947118])
    for _ in tqdm(range(5)):
        for s in range(num_steps):
            o = env.reset()
            # for _ in range(50):
            #     env.step(env.action_space.sample())
            target_pos = env.get_target_pos()
            target_pos = env.sim.data.get_body_xpos("Bread_main") + np.array(
                [0.0, 0.0, 0.02]
            )
            # cube_pos = env.sim.data.get_body_xpos('cube_main')
            # target_pos = cube_pos + np.array([0., 0., 0.02])
            mp_to_point_fast(
                env,
                env.ik_controller_config,
                env.osc_controller_config,
                np.concatenate((target_pos, env.reset_ori)).astype(np.float64),
                qpos=env.reset_qpos,
                qvel=env.reset_qvel,
                grasp=False,
                ignore_object_collision=False,
                planning_time=planning_time,
                # planning_time=env.planning_time,
                get_intermediate_frames=True,
            )
            og_xpos = env._eef_xpos.copy() + np.array([0.0, 0.6, 0.0])
            mp_to_point_fast(
                env,
                env.ik_controller_config,
                env.osc_controller_config,
                np.concatenate((target_second_pos, env.reset_ori)).astype(np.float64),
                qpos=env.reset_qpos,
                qvel=env.reset_qvel,
                grasp=False,
                ignore_object_collision=False,
                planning_time=planning_time,
                # planning_time=env.planning_time,
                get_intermediate_frames=True,
            )
            target_dist = np.linalg.norm(target_second_pos - env._eef_xpos)
            mp_fast_dists.append(target_dist)
            print(
                f"Distance from target: {np.linalg.norm(target_second_pos - env._eef_xpos)}"
            )
            # make_video(env.intermediate_frames, '.', -3)
    make_video(env.intermediate_frames, ".", 1)
    env.intermediate_frames = []
    print(f"MP fast results: {np.mean(mp_fast_dists)}")
    np.random.seed(0)
    env.seed(0)
    for _ in tqdm(range(5)):
        for s in range(num_steps):
            o = env.reset()
            # for _ in range(50):
            #     env.step(env.action_space.sample())
            target_pos = env.get_target_pos()
            target_pos = env.sim.data.body_xpos[env.obj_body_id[env.obj_to_use]]
            # cube_pos = env.sim.data.get_body_xpos('cube_main')
            # target_pos = cube_pos + np.array([0., 0., 0.02])
            mp_to_point(
                env,
                env.ik_controller_config,
                env.osc_controller_config,
                np.concatenate((target_pos, env.reset_ori)).astype(np.float64),
                qpos=env.reset_qpos,
                qvel=env.reset_qvel,
                grasp=False,
                ignore_object_collision=False,
                planning_time=planning_time,
                # planning_time=env.planning_time,
                get_intermediate_frames=True,
            )
            og_xpos = env._eef_xpos.copy() + np.array([0.0, 0.6, 0.0])
            mp_to_point(
                env,
                env.ik_controller_config,
                env.osc_controller_config,
                np.concatenate((target_second_pos, env.reset_ori)).astype(np.float64),
                qpos=env.reset_qpos,
                qvel=env.reset_qvel,
                grasp=False,
                ignore_object_collision=False,
                planning_time=planning_time,
                # planning_time=env.planning_time,
                get_intermediate_frames=True,
            )
            target_dist = np.linalg.norm(target_second_pos - env._eef_xpos)
            mp_slow_dists.append(target_dist)
            print(
                f"Distance from target: {np.linalg.norm(target_second_pos - env._eef_xpos)}"
            )
            # make_video(env.intermediate_frames, '.', -3)
    make_video(env.intermediate_frames, ".", 2)
    env.intermediate_frames = []
    print(f"MP slow results: {np.mean(mp_slow_dists)}")
    # test_random_positions(500)
