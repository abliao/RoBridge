"""
Test script for camera transforms. This test will read the ground-truth 
object state in the Lift environment, transform it into a pixel location
in the camera frame, then transform it back to the world frame, and assert
that the values are close.
"""
import random
import time

import cv2
import numpy as np
import robosuite
import robosuite.utils.camera_utils as CU
from robosuite.controllers import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from urdfpy import URDF

from rlkit.mprl.mp_env import (
    RobosuiteEnv,
    check_robot_collision,
    compute_pcd,
    pcd_collision_check,
)


def test_camera_transforms():
    # set seeds
    random.seed(0)
    np.random.seed(0)

    env = robosuite.make(
        "PickPlaceCan",
        robots=["Panda"],
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names="birdview",
        camera_heights=1024,
        camera_widths=1024,
        hard_reset=False,
    )
    env = RobosuiteEnv(GymWrapper(env))
    obs_dict = env.reset()
    env.robot = URDF.load(
        robosuite.__file__[: -len("/__init__.py")]
        + "/models/assets/bullet_data/panda_description/urdf/panda_arm_hand.urdf"
    )
    xyz, object_pcd = compute_pcd(env, is_grasped=True)
    np.save("object_pcd.npy", object_pcd)
    # for i in range(100):
    #     env.step(np.random.uniform(-1, 1, 7))
    #     t = time.time()
    #     collision = pcd_collision_check(
    #         env,
    #         xyz,
    #         env.sim.data.qpos[:7],
    #         env.sim.data.qpos[7:9],
    #         False,
    #         False,
    #         obj_idx=0,
    #     )
    #     print("pcd collision check time", time.time() - t)
    #     t = time.time()
    #     gt_collision = check_robot_collision(env, False)
    #     print("robot collision check time", time.time() - t)
    #     cv2.imwrite(f"test_{i}.png", env.get_image())
    #     print(i, gt_collision, collision)
    # np.save("pointcloud.npy", xyz)

    env.close()


if __name__ == "__main__":
    test_camera_transforms()
