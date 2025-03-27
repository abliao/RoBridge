import cv2
import numpy as np
import robosuite as suite
from robosuite.controllers.controller_factory import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper

from rlkit.core import logger
from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv
from rlkit.mprl.mp_env import MPEnv, mp_to_point

if __name__ == "__main__":
    environment_kwargs = {
        "control_freq": 20,
        "controller": "OSC_POSE",
        "env_name": "PickPlaceBread",
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
        "teleport_instead_of_mp": False,
        "planning_time": 20,
        "mp_bounds_low": (-1.45, -1.25, 0.45),
        "mp_bounds_high": (0.45, 0.85, 2.25),
        "update_with_true_state": True,
    }
    env = MPEnv(NormalizedBoxEnv(GymWrapper(env)), **mp_env_kwargs)
    logger.set_snapshot_dir(
        "/home/mdalal/research/mprl/rlkit/data/controller_debugging"
    )
    num_steps = 1
    total = 0
    for s in range(num_steps):
        env.reset()
        for _ in range(50):
            env.step(env.action_space.sample())
        target_pos = env.get_target_pos()
        mp_to_point(
            env,
            env.ik_controller_config,
            env.osc_controller_config,
            np.concatenate((target_pos, env.reset_ori)).astype(np.float64),
            qpos=env.reset_qpos,
            qvel=env.reset_qvel,
            grasp=False,
            ignore_object_collision=False,
            planning_time=env.planning_time,
            get_intermediate_frames=True,
        )
