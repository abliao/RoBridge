import numpy as np
import gym
from tqdm import tqdm
from gym import spaces
import sys
import env
import cv2
from mopa_rl.env.inverse_kinematics import qpos_from_site_pose_sampling, qpos_from_site_pose
from mopa_rl.util.env import joint_convert, mat2quat, quat_mul, rotation_matrix, quat2mat
from mopa_rl.util.transform_utils import mat2pose, convert_quat, pose2mat
from mopa_rl.config import sawyer
import matplotlib.pyplot as plt
from collections import namedtuple, OrderedDict
from mopa_rl.config.default_configs import LIFT_CONFIG, LIFT_OBSTACLE_CONFIG, ASSEMBLY_OBSTACLE_CONFIG
import mujoco_py

def save_img(env, filename="test.png"):
    frame = env.render("rgb_array")
    plt.imshow(frame)
    plt.savefig(filename)

def get_site_pose(env, name):
    """
    Gets pose of site.
    Args:
        env: Gym environment
        name: name of site in sim
    Returns:
        (pos, orn) tuple where pos is xyz location of site, orn
            is (4,) numpy.ndarray corresponding to quat in xyzw format
    """
    xpos = env.sim.data.get_site_xpos(name)[: len(env.min_world_size)].copy()
    model = env.sim.model
    xquat = mat2quat(env.sim.data.get_site_xmat(name).copy())
    return xpos, xquat

def get_object_pose(env, name):
    """
    Gets pose of desired object.
    Args:
        env: Gym environment
        name: name of manipulation object from body_names in sim
    Returns:
        (7,) numpy.ndarray corresponding to position and quaternion,
            where quaternion is in xyzw format
    """
    start = env.sim.model.body_jntadr[env.sim.model.body_name2id(name)]
    xpos = env.sim.data.qpos[start:start+3].copy()
    xquat = env.sim.data.qpos[start+3:start+7].copy()
    xquat = convert_quat(xquat, to="xyzw")
    return np.concatenate((xpos, xquat))

def set_object_pose(env, name, new_xpos, new_xquat):
    """
    Gets pose of desired object.
    Args:
        env: Gym environment
        name: name of manipulation object from body_names in sim
        new_xpos: (3,) numpy.ndarray of new xyz coordinates
        new_xquat: (4,) numpy.ndarary of new quaternion, in xyzw format
    Returns:
        None
    """
    start = env.sim.model.body_jntadr[env.sim.model.body_name2id(name)]
    new_xquat = convert_quat(new_xquat, to='wxyz')
    env.sim.data.qpos[start:start+3] = new_xpos
    env.sim.data.qpos[start+3:start+7] = new_xquat

def set_robot_based_on_ee_pos(
    env,
    ac,
    ik_env,
    qpos,
    qvel, 
    is_grasped,
    target_object,
    config,
):
    """
    Takes in action in the format of desired delta in orientation and position
    and teleports there.
    Args:
        env: Gym environment
        ac: OrderedDict - should have keys 'default' and optionally 'quat'
            corresponding to target xyz and quat
        ik_env: Gym environment - copy of env where ik algorithm is run
        qpos: canonical pose to reset to when running IK
        qvel: same as above
        is_grasped: whether object is grasped or not
        config: config file of environment
    Returns:
        (success, err_norm), where success is if ik is successful, err_norm
            is how far we are from desired target
    """
    # keep track of gripper pos, etc
    gripper_qpos = env.sim.data.qpos[env.ref_gripper_joint_pos_indexes].copy()
    gripper_qvel = env.sim.data.qvel[env.ref_gripper_joint_pos_indexes].copy()
    object_pose = np.concatenate([
        env.sim.data.get_body_xpos(target_object),
        convert_quat(env.sim.data.get_body_xquat(target_object))
    ])
    old_eef_xpos, old_eef_xquat = get_site_pose(env, config['ik_target'])
    object_pose = get_object_pose(env, target_object).copy()
    target_cart = np.clip(
        env.sim.data.get_site_xpos(config["ik_target"])[: len(env.min_world_size)]
        + config["action_range"] * ac["default"],
        env.min_world_size,
        env.max_world_size,
    )
    if "quat" in ac.keys():
        target_quat = mat2quat(env.sim.data.get_site_xmat(config["ik_target"]))
        #target_quat = target_quat[[3, 0, 1, 1]] i've commented this out for now
        # since i don't think it's the right thing to do here (will address if it is needed)
        target_quat = quat_mul(
            target_quat,
            (ac["quat"] / np.linalg.norm(ac["quat"])).astype(np.float64),
        )
    else:
        target_quat = None
    ik_env.set_state(env.sim.data.qpos.copy(), env.data.qvel.copy())
    result = qpos_from_site_pose(
        ik_env,
        config["ik_target"],
        target_pos=target_cart,
        target_quat=target_quat,
        rot_weight=2.0,
        joint_names=env.robot_joints,
        max_steps=100,
        tol=1e-2,
    )
    # set state here 
    env.set_state(ik_env.sim.data.qpos.copy(), ik_env.sim.data.qvel.copy())
    if is_grasped:
        
        env.sim.data.qpos[env.ref_gripper_joint_pos_indexes] = gripper_qpos 
        env.sim.data.qvel[env.ref_gripper_joint_pos_indexes] = gripper_qvel

        # compute transform between new and old 
        ee_old_mat = pose2mat((old_eef_xpos, old_eef_xquat))
        new_eef_xpos, new_eef_xquat = get_site_pose(env, config['ik_target'])
        ee_new_mat = pose2mat((new_eef_xpos, new_eef_xquat))
        transform = ee_new_mat @ np.linalg.inv(ee_old_mat)
        
        # get new object pose
        new_object_pose = mat2pose(
            np.dot(transform, pose2mat((object_pose[:3], object_pose[3:])))
        )
        set_object_pose(env, target_object, new_object_pose[0], new_object_pose[1])
        env.sim.forward()
    return result.success, result.err_norm

def run_hardcoded_lift_policy(
    env, 
    ik_env,
    do_grasp=True, # set to false if you want negative control (teleport but don't grasp, should fail)
    save=False # set to true if you want snapshots of grasping process
    ):
    # set gripper to open (not strictly necessary but useful for negative control)
    open_gripper_ac = OrderedDict()
    open_gripper_ac = np.array([0., 0., 0., 0., 0., 0., 0., -1.])
    _, _, _, _ = env.step(open_gripper_ac)
    
    if save:
        save_img(env, f"start_state_lift_do_grasp_{do_grasp}.png")
    

    # 
    # teleport to cube
    cube_pos = get_object_pose(env, "cube")[:3]
    gripper_pos = get_site_pose(env, "grip_site")[0][: len(env.min_world_size)]
    teleport_ac = OrderedDict() 
    teleport_ac['default'] = cube_pos - gripper_pos
    set_robot_based_on_ee_pos(
        env,
        teleport_ac,
        ik_env,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(), 
        False,
        "cube",
        LIFT_CONFIG,
    )
    # if we want to complete task successfully
    if do_grasp:
        grip_ac = OrderedDict()
        grip_ac['default'] = np.array([0., 0., 0., 0., 0., 0., 0., 1.])
        _, _, _, _ = env.step(grip_ac)
    # check to see whether we have or have not grasped the object: 
    env.compute_reward(None)
    if save:
        save_img(env, f"grasped_state_lift_do_grasp_{do_grasp}.png")
    
    # teleport back up to complete task
    success_ac = OrderedDict()
    success_ac['default'] = np.array([0., 0., 0.39])
    set_robot_based_on_ee_pos(
        env,
        success_ac,
        ik_env,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(), 
        do_grasp,
        "cube",
        LIFT_CONFIG,
    )
    # compute reward to set success to true
    env.compute_reward(None)

    if save:
        save_img(env, f"end_state_lift_do_grasp_{do_grasp}.png")
    
    assert env._success == do_grasp


def run_hardcoded_lift_obstacle_policy(
    env, 
    ik_env,
    do_grasp=True, # set to false if you want negative control (teleport but don't grasp, should fail)
    save=False # set to true if you want snapshots of grasping process
):
    # set gripper to open (not strictly necessary but useful for negative control)
    open_gripper_ac = OrderedDict()
    open_gripper_ac = np.array([0., 0., 0., 0., 0., 0., 0., -1.])
    _, _, _, _ = env.step(open_gripper_ac)
    
    if save:
        save_img(env, f"start_state_lift_do_grasp_{do_grasp}.png")
    
    # teleport to cube
    cube_pos = get_object_pose(env, "cube")[:3]
    gripper_pos = get_site_pose(env, "grip_site")[0][: len(env.min_world_size)]
    teleport_ac = OrderedDict() 
    teleport_ac['default'] = cube_pos - gripper_pos
    set_robot_based_on_ee_pos(
        env,
        teleport_ac,
        ik_env,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(), 
        False,
        "cube",
        LIFT_OBSTACLE_CONFIG,
    )
    # if we want to complete task successfully
    if do_grasp:
        grip_ac = OrderedDict()
        grip_ac['default'] = np.array([0., 0., 0., 0., 0., 0., 0., 1.])
        _, _, _, _ = env.step(grip_ac)
    # check to see whether we have or have not grasped the object: 
    env.compute_reward(None)
    if save:
        save_img(env, f"grasped_state_lift_do_grasp_{do_grasp}.png")
    
    # teleport back up to complete task
    success_ac = OrderedDict()
    success_ac['default'] = np.array([0., 0., 0.39])
    set_robot_based_on_ee_pos(
        env,
        success_ac,
        ik_env,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(), 
        do_grasp,
        "cube",
        LIFT_OBSTACLE_CONFIG,
    )
    # compute reward to set success to true
    env.compute_reward(None)

    if save:
        save_img(env, f"end_state_lift_do_grasp_{do_grasp}.png")
    
    assert env._success == do_grasp

def run_hardcoded_assembly_obstacle_policy(
    env,
    ik_env,
    save=False,
):
    if save:
        save_img(env, f"start_state_assembly.png")
    """
    Environment notes:
        - you start off in the grasped state?
        - Reward notes
            - peghead is the blue thing
            - hole is what we care about 
            - assumption - if i just do a transform on the assembly object
                then everything else should be an attached geom and it should work
            - gripper is attached to arm, no need to worry about it
    """
    # teleport to peg
    hole_pos = get_site_pose(env, "hole")[0]
    desired_pos = hole_pos + np.array([0., 0., 0.3])
    gripper_pos = get_site_pose(env, "grip_site")[0]
    teleport_ac = OrderedDict() 
    teleport_ac['default'] = desired_pos - gripper_pos 
    set_robot_based_on_ee_pos(
        env,
        teleport_ac,
        ik_env,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(), 
        False,
        "0_part0",
        ASSEMBLY_OBSTACLE_CONFIG,
    )
    if save:
        save_img(env, f"teleported_state_assembly.png")

def main():
    """
    Create env, ik_env, look through config dict to get environments
    """
    np.random.seed(0)
    env = gym.make(**ASSEMBLY_OBSTACLE_CONFIG)
    ik_env = gym.make(**ASSEMBLY_OBSTACLE_CONFIG)
    env.reset()
    run_hardcoded_assembly_obstacle_policy(
        env,
        ik_env,
        save=True,
    )

if __name__ == "__main__":
    main()