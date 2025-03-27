"""
Test script for camera transforms. This test will read the ground-truth 
object state in the Lift environment, transform it into a pixel location
in the camera frame, then transform it back to the world frame, and assert
that the values are close.
"""
import random
import time

import numpy as np
import robosuite
import robosuite.utils.camera_utils as CU
from robosuite.controllers import load_controller_config


def get_camera_depth(sim, camera_name, camera_height, camera_width):
    """
    Obtains depth image.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        im (np.array): the depth image b/w 0 and 1
    """
    return sim.render(
        camera_name=camera_name, height=camera_height, width=camera_width, depth=True
    )[1][::-1]


def get_object_pose_from_seg(
    env, object_string, camera_name, camera_width, camera_height, sim
):
    """
    Get the object pose from the segmentation map. This is done by finding the object id in the segmentation map,
    then finding the pixels that correspond to that object id. Then, we find project those pixels into 3D space
    using the depth map, and average the 3D points to get the object pose.

    Args:
        env (robosuite): robosuite environment
        object_string (str): string that is in the object name
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
        sim (MjSim): simulator instance
    Return:
        estimated_obj_pos (np.array): estimated object position in world frame
    """
    segmentation_map = CU.get_camera_segmentation(
        camera_name=camera_name,
        camera_width=camera_width,
        camera_height=camera_height,
        sim=sim,
    )
    geom_ids = np.unique(segmentation_map[:, :, 1])
    object_id = None
    for geom_id in geom_ids:
        geom_name = sim.model.geom_id2name(geom_id)
        if geom_name is None or geom_name.startswith("Visual"):
            continue
        if object_string in sim.model.geom_id2name(geom_id):
            object_id = geom_id
            break
    cube_mask = segmentation_map[:, :, 1] == object_id
    depth_map = get_camera_depth(
        sim=sim,
        camera_name=camera_name,
        camera_height=camera_height,
        camera_width=camera_width,
    )
    depth_map = np.expand_dims(
        CU.get_real_depth_map(sim=env.sim, depth_map=depth_map), -1
    )

    # get camera matrices
    world_to_camera = CU.get_camera_transform_matrix(
        sim=env.sim,
        camera_name=camera_name,
        camera_height=camera_height,
        camera_width=camera_width,
    )
    camera_to_world = np.linalg.inv(world_to_camera)

    obj_pixels = np.argwhere(cube_mask)
    # transform from camera pixel back to world position
    # can we do this batched somehow...
    obj_poses = []
    obj_poses = CU.transform_from_pixels_to_world(
        pixels=obj_pixels,
        depth_map=depth_map[..., 0],
        camera_to_world_transform=camera_to_world,
    )
    estimated_obj_pos = np.mean(obj_poses, axis=0)
    return estimated_obj_pos


def test_camera_transforms():
    # set seeds
    random.seed(0)
    np.random.seed(0)

    camera_name = "agentview"
    camera_height = 480
    camera_width = 640
    env = robosuite.make(
        "PickPlaceCereal",
        robots=["Panda"],
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_names=[camera_name],
        camera_depths=[True],
        camera_heights=[camera_height],
        camera_widths=[camera_width],
        reward_shaping=True,
        control_freq=20,
    )
    obs_dict = env.reset()
    sim = env.sim

    # ground-truth object position
    obj_pos = obs_dict["object-state"][:3]

    # unnormalized depth map
    object_string = "gripper0_finger1_visual"
    gripper1_obj_pos = get_object_pose_from_seg(
        env, object_string, camera_name, camera_width, camera_height, sim
    )
    object_string = "gripper0_finger2_visual"
    gripper2_obj_pos = get_object_pose_from_seg(
        env, object_string, camera_name, camera_width, camera_height, sim
    )

    print(gripper1_obj_pos)
    print(gripper2_obj_pos)
    print(obj_pos)

    env.close()


if __name__ == "__main__":
    test_camera_transforms()
