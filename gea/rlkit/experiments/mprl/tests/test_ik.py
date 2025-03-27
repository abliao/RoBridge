import cv2
import numpy as np
import robosuite as suite
from robosuite.controllers import controller_factory
from robosuite.controllers.controller_factory import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from tqdm import tqdm

from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv
from rlkit.mprl.mp_env import MPEnv, set_robot_based_on_ee_pos, update_controller_config

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
    mp_env_kwargs = {
        "vertical_displacement": 0.04,
        "teleport_instead_of_mp": True,
        "randomize_init_target_pos": False,
        "mp_bounds_low": (-1.45, -1.25, 0.45),
        "mp_bounds_high": (0.45, 0.85, 2.25),
        "backtrack_movement_fraction": 0.001,
        "clamp_actions": True,
        "update_with_true_state": True,
        "grip_ctrl_scale": 0.0025,
        "planning_time": 20,
    }
    controller = environment_kwargs.pop("controller")
    controller_config = load_controller_config(default_controller=controller)
    mp_env_kwargs["controller_configs"] = controller_config
    env = suite.make(
        **environment_kwargs,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=True,
        controller_configs=controller_config,
        camera_names="frontview",
        camera_heights=1024,
        camera_widths=1024,
    )
    env = MPEnv(NormalizedBoxEnv(GymWrapper(env)), **mp_env_kwargs)
    num_steps = 10
    total = 0
    for s in range(num_steps):
        env.reset()
        qpos, qvel = env.sim.data.qpos.copy(), env.sim.data.qvel.copy()
        for i in tqdm(range(100)):
            env.step(env.action_space.sample())
        error = set_robot_based_on_ee_pos(
            env,
            env._eef_xpos.copy(),
            env.reset_ori.copy(),  # gets 1e-8 error
            # env._eef_xquat.copy(), # gets 1e-1 error!
            env.ik_ctrl,
            qpos,
            qvel,
            False,
            controller_config,
        )
        total += error
    print(f"Avg Distance to target: {total/num_steps})")
