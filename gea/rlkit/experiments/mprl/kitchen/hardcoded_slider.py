import time

import cv2
import d4rl  # Import to register d4rl environments to Gym
import disrep4rl.environments.kitchen_custom_envs  # Import to register kitchen custom environments to Gym
import dm_env
import gym
import numpy as np
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import specs
import robosuite.utils.transform_utils as T
from rlkit.mprl.mp_env_kitchen import (
    MPEnv,
    compute_object_pcd,
    compute_pcd,
    get_object_pose_mp,
)

if __name__ == "__main__":
    np.random.seed(0)
    env_name = "kitchen-tlb-v0"
    env = gym.make(env_name)
    env = MPEnv(env, use_vision_pose_estimation=True)
    env.reset()
    t = time.time()
    num_steps = 25
    for step in range(num_steps):
        o, r, d, i = env.step(np.concatenate((np.random.uniform(-1, 1, 8), [0])))
        # cv2.imwrite(f"test_{step}.png", env.get_image())
        # print(repr(env.sim.data.qpos[:9]))
        pts = compute_object_pcd(env)
        gt_pose = get_object_pose_mp(env)
        print(np.mean(pts, axis=0) - gt_pose[0][:3])
        # print(repr(env.sim.data.body_xpos[env.sim.model.body_name2id("panda0_link0")]))
        # np.save("pcd.npy", pts)
        # cv2.imwrite(f"test_agentview_{step}.png", env.sim.render(
        # camera_name="sideview", height=512, width=512
        # )[::-1])
        # if d:
        #     env.reset()
    print("FPS: ", num_steps / (time.time() - t))
