import copy
import io
import xml.etree.ElementTree as ET

import cv2
import gym
import numpy as np
import robosuite.utils.transform_utils as T
from gym import spaces
from robosuite.controllers import controller_factory
from robosuite.utils.control_utils import orientation_error
from robosuite.utils.transform_utils import (
    euler2mat,
    mat2pose,
    mat2quat,
    pose2mat,
    quat2mat,
    quat_conjugate,
    quat_multiply,
)

from rlkit.core import logger
from rlkit.envs.proxy_env import ProxyEnv
from rlkit.mprl import module
from rlkit.mprl.inverse_kinematics import qpos_from_site_pose
from rlkit.torch.model_based.dreamer.visualization import add_text

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    import sys
    from os.path import abspath, dirname, join

    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "py-bindings"))
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou


def get_object_pos(env):
    """
    Note this is only used for computing the target for MP
    this is NOT the true object pose
    """
    object_pos = env._get_pos_objects().copy()
    return object_pos


def get_object_pose(env):
    object_pos = env._get_pos_objects().copy()
    object_quat = env._get_quat_objects().copy()
    return np.concatenate((object_pos, object_quat))


def set_object_pose(env, object_pos, object_quat):
    """
    Set the object pose in the environment.
    Makes sure to convert from xyzw to wxyz format for quaternion. qpos requires wxyz!
    Arguments:
        env
        object_pos (np.ndarray): 3D position of the object
        object_quat (np.ndarray): 4D quaternion of the object (xyzw format)

    """
    object_quat = T.convert_quat(object_quat, to="wxyz")
    env._set_obj_pose(np.concatenate((object_pos, object_quat)))


def get_object_string(env):
    obj_string = "obj"
    return obj_string


def check_robot_string(string):
    if string is None:
        return False
    return (
        string.startswith("robot")
        or string.startswith("leftclaw")
        or string.startswith("rightclaw")
        or string.startswith("rightpad")
        or string.startswith("leftpad")
    )


def check_string(string, other_string):
    if string is None:
        return False
    return string.startswith(other_string)


def check_robot_collision(env, ignore_object_collision):
    obj_string = get_object_string(env)
    d = env.sim.data
    for coni in range(d.ncon):
        con1 = env.sim.model.geom_id2name(d.contact[coni].geom1)
        con2 = env.sim.model.geom_id2name(d.contact[coni].geom2)
        if check_robot_string(con1) ^ check_robot_string(con2):
            if (
                check_string(con1, obj_string)
                or check_string(con2, obj_string)
                and ignore_object_collision
            ):
                # if the robot and the object collide, then we can ignore the collision
                continue
            return True
        elif ignore_object_collision:
            if check_string(con1, obj_string) or check_string(con2, obj_string):
                # if we are supposed to be "ignoring object collisions" then we assume the
                # robot is "joined" to the object. so if the object collides with any non-robot
                # object, then we should call that a collision
                return True
    return False


def gripper_contact(string, side):
    if string is None:
        return False
    if side == "left":
        return string.startswith("leftclaw") or string.startswith("leftpad")
    elif side == "right":
        return string.startswith("rightclaw") or string.startswith("rightpad")


def check_object_grasp(env):
    # TODO: finish this function
    obs = env._get_obs()
    obj = obs[4:7]
    action = np.zeros(4)
    action[-1] = 1
    object_grasped = env._gripper_caging_reward(
        action,
        obj,
        obj_radius=0.015,
        pad_success_thresh=0.05,
        object_reach_radius=0.01,
        xz_thresh=0.01,
        desired_gripper_effort=0.7,
        high_density=True,
    )
    thresh = 0.9
    # also check that object is in contact with gripper
    object_gripper_contact = False
    d = env.sim.data
    obj_string = get_object_string(env)
    left_gripper_contact = False
    right_gripper_contact = False
    object_in_contact_with_env = False
    for coni in range(d.ncon):
        con1 = env.sim.model.geom_id2name(d.contact[coni].geom1)
        con2 = env.sim.model.geom_id2name(d.contact[coni].geom2)
        if (gripper_contact(con1, "left") and check_string(con2, obj_string)) or (
            gripper_contact(con2, "left") and check_string(con1, obj_string)
        ):
            left_gripper_contact = True
        elif (gripper_contact(con1, "right") and check_string(con2, obj_string)) or (
            gripper_contact(con2, "right") and check_string(con1, obj_string)
        ):
            right_gripper_contact = True
        else:
            if not check_robot_string(con1) and check_string(con2, obj_string):
                object_in_contact_with_env = True
            if not check_robot_string(con2) and check_string(con1, obj_string):
                object_in_contact_with_env = True
        # if check_robot_string(con1) and check_string(con2, obj_string):
        #     object_gripper_contact = True
        # if check_robot_string(con2) and check_string(con1, obj_string):
        #     object_gripper_contact = True
    # check if there exists a string starting with left and a string starting with right in gripper contacts
    object_gripper_contact = left_gripper_contact and right_gripper_contact
    is_grasped = object_gripper_contact and (not object_in_contact_with_env)
    # is_grasped = object_grasped > thresh and object_gripper_contact and object_lifted
    return is_grasped


def set_robot_based_on_ee_pos(
    env,
    target_pos,
    target_quat,
    qpos,
    qvel,
    is_grasped,
):
    """
    Set robot joint positions based on target ee pose. Uses IK to solve for joint positions.
    If grasping an object, ensures the object moves with the arm in a consistent way.
    """
    # cache quantities from prior to setting the state
    object_pose = get_object_pose(env).copy()
    gripper_qpos = env.sim.data.qpos[7:9].copy()
    gripper_qvel = env.sim.data.qvel[7:9].copy()
    old_eef_xquat = env._eef_xquat.copy()
    old_eef_xpos = env._eef_xpos.copy()

    # reset to canonical state before doing IK
    env.sim.data.qpos[:7] = qpos[:7]
    env.sim.data.qvel[:7] = qvel[:7]
    env.sim.forward()

    qpos_from_site_pose(
        env,
        "endEffector",
        target_pos=target_pos,
        target_quat=target_quat.astype(np.float64),
        joint_names=[
            "right_j0",
            "right_j1",
            "right_j2",
            "right_j3",
            "right_j4",
            "right_j5",
            "right_j6",
        ],
        tol=1e-14,
        rot_weight=1.0,
        regularization_threshold=0.1,
        regularization_strength=3e-2,
        max_update_norm=2.0,
        progress_thresh=20.0,
        max_steps=1000,
    )
    if is_grasped:
        env.sim.data.qpos[7:9] = gripper_qpos
        env.sim.data.qvel[7:9] = gripper_qvel

        # compute the transform between the old and new eef poses
        ee_old_mat = pose2mat((old_eef_xpos, old_eef_xquat))
        ee_new_mat = pose2mat((env._eef_xpos, env._eef_xquat))
        transform = ee_new_mat @ np.linalg.inv(ee_old_mat)

        # apply the transform to the object
        new_object_pose = mat2pose(
            np.dot(transform, pose2mat((object_pose[:3], object_pose[3:])))
        )
        set_object_pose(env, new_object_pose[0], new_object_pose[1])
        env.sim.forward()
    else:
        # make sure the object is back where it started
        set_object_pose(env, object_pose[:3], object_pose[3:])

    env.sim.data.qpos[7:9] = gripper_qpos
    env.sim.data.qvel[7:9] = gripper_qvel
    env.sim.forward()

    ee_error = np.linalg.norm(env._eef_xpos - target_pos)
    # need to update the mocap pos post teleport
    env.reset_mocap2body_xpos(env.sim)
    return ee_error


def backtracking_search_from_goal(
    env,
    ignore_object_collision,
    start_pos,
    start_ori,
    goal_pos,
    ori,
    qpos,
    qvel,
    movement_fraction,
    is_grasped,
):
    # only search over the xyz position, orientation should be the same as commanded
    curr_pos = goal_pos.copy()
    set_robot_based_on_ee_pos(env, curr_pos, ori, qpos, qvel, is_grasped)
    collision = check_robot_collision(env, ignore_object_collision)
    iters = 0
    max_iters = int(1 / movement_fraction)
    while collision and iters < max_iters:
        curr_pos = curr_pos - movement_fraction * (goal_pos - start_pos)
        set_robot_based_on_ee_pos(
            env,
            curr_pos,
            ori,
            qpos,
            qvel,
            is_grasped,
        )
        collision = check_robot_collision(env, ignore_object_collision)
        iters += 1
    if collision:
        return np.concatenate(
            (start_pos, start_ori)
        )  # assumption is this is always valid!
    else:
        return np.concatenate((curr_pos, ori))


def mp_to_point(
    env,
    ik_controller_config,
    osc_controller_config,
    pos,
    qpos,
    qvel,
    grasp=False,
    ignore_object_collision=False,
    planning_time=1,
    get_intermediate_frames=False,
    backtrack_movement_fraction=0.001,
):
    # TODO: this code has NOT been updated to work with metaworld
    qpos_curr = env.sim.data.qpos.copy()
    qvel_curr = env.sim.data.qvel.copy()

    og_eef_xpos = env._eef_xpos.copy()
    og_eef_xquat = env._eef_xquat.copy()

    def isStateValid(state):
        pos = np.array([state.getX(), state.getY(), state.getZ()])
        quat = np.array(
            [
                state.rotation().x,
                state.rotation().y,
                state.rotation().z,
                state.rotation().w,
            ]
        )
        if all(pos == og_eef_xpos) and all(quat == og_eef_xquat):
            # start state is always valid.
            return True
        else:
            # TODO; if it was grasping before ik and not after automatically set to invalid
            set_robot_based_on_ee_pos(env, pos, quat, qpos, qvel, grasp)
            valid = not check_robot_collision(
                env, ignore_object_collision=ignore_object_collision
            )
            return valid

    # create an SE3 state space
    space = ob.SE3StateSpace()

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(3)

    # compare bounds to start state
    bounds_low = env.mp_bounds_low
    bounds_high = env.mp_bounds_high

    bounds_low = np.minimum(env.mp_bounds_low, og_eef_xpos)
    bounds_high = np.maximum(env.mp_bounds_high, og_eef_xpos)
    pos[:3] = np.clip(pos[:3], bounds_low, bounds_high)

    bounds.setLow(0, bounds_low[0])
    bounds.setLow(1, bounds_low[1])
    bounds.setLow(2, bounds_low[2])
    bounds.setHigh(0, bounds_high[0])
    bounds.setHigh(1, bounds_high[1])
    bounds.setHigh(2, bounds_high[2])
    space.setBounds(bounds)

    # construct an instance of space information from this state space
    si = ob.SpaceInformation(space)
    # set state validity checking for this space
    si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
    # create a random start state
    start = ob.State(space)
    start().setXYZ(*og_eef_xpos)
    start().rotation().x = og_eef_xquat[0]
    start().rotation().y = og_eef_xquat[1]
    start().rotation().z = og_eef_xquat[2]
    start().rotation().w = og_eef_xquat[3]

    goal = ob.State(space)
    goal().setXYZ(*pos[:3])
    goal().rotation().x = pos[3]
    goal().rotation().y = pos[4]
    goal().rotation().z = pos[5]
    goal().rotation().w = pos[6]
    goal_valid = isStateValid(goal())
    goal_error = set_robot_based_on_ee_pos(env, pos[:3], pos[3:], qpos, qvel, grasp)
    print(f"Goal Validity: {goal_valid}")
    print(f"Goal Error {goal_error}")
    if not goal_valid:
        pos = backtracking_search_from_goal(
            env,
            ignore_object_collision,
            og_eef_xpos,
            og_eef_xquat,
            pos[:3],
            pos[3:],
            qpos,
            qvel,
            is_grasped=grasp,
            movement_fraction=backtrack_movement_fraction,
        )
        goal = ob.State(space)
        goal().setXYZ(*pos[:3])
        goal().rotation().x = pos[3]
        goal().rotation().y = pos[4]
        goal().rotation().z = pos[5]
        goal().rotation().w = pos[6]
        goal_error = set_robot_based_on_ee_pos(
            env,
            pos[:3],
            pos[3:],
            qpos,
            qvel,
            grasp,
        )
        goal_valid = isStateValid(goal())
        print(f"Updated Goal Validity: {goal_valid}")
        print(f"Goal Error {goal_error}")
        if not goal_valid:
            cv2.imwrite(
                f"{logger.get_snapshot_dir()}/failed_{env.num_steps}.png",
                env.get_image(),
            )
    if grasp and get_intermediate_frames:
        print(f"Goal state has reward {env.reward(None)}")
    # create a problem instance
    pdef = ob.ProblemDefinition(si)
    # set the start and goal states
    pdef.setStartAndGoalStates(start, goal)
    # create a planner for the defined space
    planner = og.RRTConnect(si)
    # set the problem we are trying to solve for the planner
    planner.setProblemDefinition(pdef)
    planner.setRange(0.05)
    # perform setup steps for the planner
    planner.setup()
    # attempt to solve the problem within planning_time seconds of planning time
    solved = planner.solve(planning_time)

    if get_intermediate_frames:
        set_robot_based_on_ee_pos(
            env,
            og_eef_xpos,
            og_eef_xquat,
            qpos,
            qvel,
            grasp,
        )
        set_robot_based_on_ee_pos(
            env,
            pos[:3],
            pos[3:],
            qpos,
            qvel,
            grasp,
        )
    intermediate_frames = []
    if solved:
        path = pdef.getSolutionPath()
        success = og.PathSimplifier(si).simplify(path, 1.0)
        converted_path = []
        for s, state in enumerate(path.getStates()):
            new_state = [
                state.getX(),
                state.getY(),
                state.getZ(),
                state.rotation().x,
                state.rotation().y,
                state.rotation().z,
                state.rotation().w,
            ]
            if env.update_with_true_state:
                # get actual state that we used for collision checking on
                set_robot_based_on_ee_pos(
                    env,
                    new_state[:3],
                    new_state[3:],
                    qpos,
                    qvel,
                    grasp,
                )
                new_state = np.concatenate((env._eef_xpos, env._eef_xquat))
            else:
                new_state = np.array(new_state)
            converted_path.append(new_state)
        # reset env to original qpos/qvel
        env._wrapped_env.reset()
        env.sim.data.qpos[:] = qpos_curr.copy()
        env.sim.data.qvel[:] = qvel_curr.copy()
        env.sim.forward()

        for state in converted_path:
            desired_rot = quat2mat(state[3:])
            for _ in range(50):
                current_rot = quat2mat(env._eef_xquat)
                rot_delta = orientation_error(desired_rot, current_rot)
                pos_delta = state[:3] - env._eef_xpos
                if grasp:
                    grip_ctrl = env.grip_ctrl_scale
                else:
                    grip_ctrl = -1
                action = np.concatenate((pos_delta, rot_delta, [grip_ctrl]))
                if np.linalg.norm(action[:-4]) < 1e-3:
                    break
                policy_step = True
                for i in range(int(env.control_timestep / env.model_timestep)):
                    env.sim.forward()
                    env.sim.step()
                    env._update_observables()
                    policy_step = False
                if hasattr(env, "num_steps"):
                    env.num_steps += 1
                if get_intermediate_frames:
                    im = env.get_image()
                    add_text(im, "Planner", (1, 10), 0.5, (0, 255, 0))
                    intermediate_frames.append(im)
        env.mp_mse = (
            np.linalg.norm(state - np.concatenate((env._eef_xpos, env._eef_xquat))) ** 2
        )
        print(f"Controller reaching MSE: {env.mp_mse}")
        env.goal_error = goal_error
    else:
        env._wrapped_env.reset()
        env.sim.data.qpos[:] = qpos_curr.copy()
        env.sim.data.qvel[:] = qvel_curr.copy()
        env.sim.forward()
        env.mp_mse = 0
        env.goal_error = 0
        env.num_failed_solves += 1
    env.intermediate_frames = intermediate_frames
    return env._get_observations()


class MetaworldEnv(ProxyEnv):
    def __init__(
        self,
        env,
        slack_reward=0,
        predict_done_actions=False,
        terminate_on_success=False,
        terminate_on_drop=False,
    ):
        super().__init__(env)
        self.num_steps = 0
        self.slack_reward = slack_reward
        self.predict_done_actions = predict_done_actions
        self.terminate_on_success = terminate_on_success
        self.terminate_on_drop = terminate_on_drop
        if self.predict_done_actions:
            self.action_space = spaces.Box(
                np.concatenate((self._wrapped_env.action_space.low, [-1])),
                np.concatenate((self._wrapped_env.action_space.high, [1])),
            )

    def get_image(self):
        im = self.render(mode="rgb_array")[:, :, ::-1]
        return im

    def reset(self, **kwargs):
        self.num_steps = 0
        self.was_in_hand = False
        self.has_succeeded = False
        self.terminal = False
        return super().reset(**kwargs)

    def check_grasp(
        self,
    ):
        return check_object_grasp(self)

    def update_done_info_based_on_termination(self, i, d):
        if self.terminal:
            # if we've already terminated, don't let the agent get any more reward
            i["bad_mask"] = 1
        else:
            i["bad_mask"] = 0
        if i["grasped"] and not self.was_in_hand:
            self.was_in_hand = True
        if self.was_in_hand and not i["grasped"] and self.terminate_on_drop:
            # if we've dropped the object, terminate
            self.terminal = True
        if i["success"] and self.terminate_on_success:
            self.has_succeeded = True
            self.terminal = True
        d = d or self.terminal
        return d

    def step(self, action):
        if self.predict_done_actions:
            old_action = action
            action = action[:-1]
        o, r, d, i = super().step(action)
        self.num_steps += 1
        i["grasped"] = float(self.check_grasp())
        i["num_steps"] = self.num_steps
        r += self.slack_reward
        if self.predict_done_actions:
            d = old_action[-1] > 0
        # if self.num_steps == self.horizon:
        #     # TODO: remove this
        #     d = True
        d = self.update_done_info_based_on_termination(i, d)
        return o, r, d, i

    @property
    def _eef_xpos(self):
        return self.get_endeff_pos()

    @property
    def _eef_xquat(self):
        return self.get_endeff_quat()


class MPEnv(MetaworldEnv):
    def __init__(
        self,
        env,
        controller_configs=None,
        recompute_reward_post_teleport=False,
        num_ll_actions_per_hl_action=25,
        planner_only_actions=False,
        add_grasped_to_obs=False,
        terminate_on_last_state=False,
        # mp
        planning_time=1,
        mp_bounds_low=None,
        mp_bounds_high=None,
        update_with_true_state=False,
        grip_ctrl_scale=1,
        backtrack_movement_fraction=0.001,
        # teleport
        vertical_displacement=0.03,
        teleport_instead_of_mp=True,
        plan_to_learned_goals=False,
        learn_residual=False,
        clamp_actions=False,
        randomize_init_target_pos=False,
        randomize_init_target_pos_range=(0.04, 0.06),
        teleport_on_grasp=False,
        use_teleports_in_step=True,
        # upstream env
        slack_reward=0,
        predict_done_actions=False,
        terminate_on_success=False,
        terminate_on_drop=False,
        # grasp checks
        check_com_grasp=False,
        verify_stable_grasp=False,
        reset_at_grasped_state=False,
        max_path_length=200,
    ):
        super().__init__(
            env,
            slack_reward=slack_reward,
            predict_done_actions=predict_done_actions,
            terminate_on_success=terminate_on_success,
            terminate_on_drop=terminate_on_drop,
        )
        self.num_steps = 0
        self.vertical_displacement = vertical_displacement
        self.teleport_instead_of_mp = teleport_instead_of_mp
        self.planning_time = planning_time
        self.plan_to_learned_goals = plan_to_learned_goals
        self.learn_residual = learn_residual
        self.mp_bounds_low = mp_bounds_low
        self.mp_bounds_high = mp_bounds_high
        self.update_with_true_state = update_with_true_state
        self.grip_ctrl_scale = grip_ctrl_scale
        self.clamp_actions = clamp_actions
        self.backtrack_movement_fraction = backtrack_movement_fraction
        self.randomize_init_target_pos = randomize_init_target_pos
        self.teleport_on_grasp = teleport_on_grasp
        self.check_com_grasp = check_com_grasp
        self.recompute_reward_post_teleport = recompute_reward_post_teleport
        self.verify_stable_grasp = verify_stable_grasp
        self.randomize_init_target_pos_range = randomize_init_target_pos_range
        self.num_ll_actions_per_hl_action = num_ll_actions_per_hl_action
        self.planner_only_actions = planner_only_actions
        self.add_grasped_to_obs = add_grasped_to_obs
        self.use_teleports_in_step = use_teleports_in_step
        self.take_planner_step = True
        self.current_ll_policy_steps = 0
        self.reset_at_grasped_state = reset_at_grasped_state
        self.terminate_on_last_state = terminate_on_last_state
        self.planner_command_orientation = False
        self.max_path_length = max_path_length

        if self.add_grasped_to_obs:
            # update observation space
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.observation_space.shape[0] + 1,),
            )

    def get_target_pos_list(self):
        pos_list = []
        # init target pos (object pos + vertical displacement)
        pos = get_object_pos(self)
        pos = pos + np.array([0, 0, self.vertical_displacement])
        pos_list.append(pos)
        # final target positions, depending on the task
        pos_list.append(self._target_pos + np.array([0, 0, 0.15]))
        return pos_list

    def get_target_pos(self):
        target_pos_list = self.get_target_pos_list()
        if self.high_level_step > len(target_pos_list) - 1:
            return target_pos_list[-1]
        return self.get_target_pos_list()[self.high_level_step]

    def get_init_target_pos(self):
        pos = get_object_pos(self)
        qpos, qvel = self.sim.data.qpos.copy(), self.sim.data.qvel.copy()
        if self.randomize_init_target_pos:
            # sample a random position in a sphere around the target (not in collision)
            # the orientation of the arm should not be changed
            stop_sampling_target_pos = False
            xquat = self._eef_xquat
            while not stop_sampling_target_pos:
                random_perturbation = np.random.normal(0, 1, 3)
                random_perturbation[2] = np.abs(random_perturbation[2])
                random_perturbation /= np.linalg.norm(random_perturbation)
                scale = np.random.uniform(*self.randomize_init_target_pos_range)
                shifted_pos = pos + random_perturbation * scale
                # backtrack from the position just in case we sampled a point in collision
                set_robot_based_on_ee_pos(
                    self,
                    shifted_pos.copy(),
                    self._eef_xquat,
                    qpos,
                    qvel,
                    is_grasped=False,
                )
                ori_cond = np.linalg.norm(self._eef_xquat - xquat) < 1e-6
                grasp_cond = not self.check_grasp()
                collision_cond = not check_robot_collision(
                    self, ignore_object_collision=False
                )
                if ori_cond and grasp_cond and collision_cond:
                    stop_sampling_target_pos = True
                else:
                    self.sim.data.qpos[:] = qpos
                    self.sim.data.qvel[:] = qvel
                    self.sim.forward()
        else:
            shifted_pos = pos + np.array([0, -0.01, self.vertical_displacement])
            set_robot_based_on_ee_pos(
                self,
                shifted_pos.copy(),
                self._eef_xquat,
                qpos,
                qvel,
                is_grasped=False,
            )
        return pos

    def reset(self, get_intermediate_frames=False, **kwargs):
        obs = self._wrapped_env.reset(**kwargs)
        self.ep_step_ctr = 0
        self.high_level_step = 0
        self.num_failed_solves = 0
        self.num_steps = 0
        self.reset_pos = self._eef_xpos.copy()
        self.reset_ori = self._eef_xquat.copy()
        self.reset_qpos = self.sim.data.qpos.copy()
        self.reset_qvel = self.sim.data.qvel.copy()
        self.initial_object_pos = get_object_pos(self).copy()
        self.was_in_hand = False
        self.has_succeeded = False
        self.terminal = False
        self.take_planner_step = True
        self.current_ll_policy_steps = 0
        if not self.plan_to_learned_goals and not self.planner_only_actions:
            if self.teleport_instead_of_mp:
                pos = self.get_init_target_pos()
                obs = self._get_obs()
                # self.num_steps += 100 #don't log this
            else:
                pos = self.get_init_target_pos()
                pos = np.concatenate((pos, self.reset_ori))
                obs = mp_to_point(
                    self,
                    pos.astype(np.float64),
                    qpos=self.reset_qpos,
                    qvel=self.reset_qvel,
                    grasp=False,
                    planning_time=self.planning_time,
                    get_intermediate_frames=get_intermediate_frames,
                    backtrack_movement_fraction=self.backtrack_movement_fraction,
                )
        if self.reset_at_grasped_state:
            pos = self.get_init_target_pos()
            for i in range(15):
                a = np.concatenate(([0, 0, -0.3], [-1]))
                o, r, d, info = self._wrapped_env.step(a)
            for i in range(10):
                a = np.concatenate(([0, 0, 0], [1]))
                o, r, d, info = self._wrapped_env.step(a)
            if not self.check_grasp():
                print("Grasp failed, resetting")
                self.reset()
        self.hasnt_teleported = True
        if self.add_grasped_to_obs:
            obs = np.concatenate((obs, np.array([0])))
        return obs

    def check_grasp(self, verify_stable_grasp=False):
        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()
        is_grasped = super().check_grasp()

        if is_grasped and verify_stable_grasp:
            # verify grasp is stable by lifting the arm and seeing if still in contact with object
            for i in range(10):
                action = np.zeros(4)
                action[2] = 0.1
                action[-1] = 1.0
                self._wrapped_env.step(action)
            is_grasped = super().check_grasp()
            self.sim.data.qpos[:] = qpos
            self.sim.data.qvel[:] = qvel
            self.sim.forward()

        if is_grasped and self.check_com_grasp:
            # check if left gripper pad is left of the com of object, right gripper pad is right of the com of object

            def name2id(type_name, name):
                obj_id = self.mjlib.mj_name2id(
                    self.model.ptr, self.mjlib.mju_str2Type(type_name), name.encode()
                )
                if obj_id < 0:
                    raise ValueError(
                        'No {} with name "{}" exists.'.format(type_name, name)
                    )
                return obj_id

            left_pos = self.sim.data.geom_xpos[
                name2id(
                    "geom", self.robots[0].gripper.important_geoms["left_fingerpad"][0]
                )
            ]
            right_pos = self.sim.data.geom_xpos[
                name2id(
                    "geom", self.robots[0].gripper.important_geoms["right_fingerpad"][0]
                )
            ]
            object_pos = get_object_pos(self)
            below_com_grasp = (left_pos[-1] - object_pos[-1] - 0.025) < 0 and (
                right_pos[-1] - object_pos[-1] - 0.025
            ) < 0
            if below_com_grasp:
                return True
            else:
                return False
        return is_grasped and not self.check_com_grasp

    def get_target_pos_no_planner(
        self,
    ):
        pose = self._target_pos + np.array([0, 0, 0.15])
        return pose

    def clamp_planner_action_mp_space_bounds(self, action):
        action[:3] = np.clip(action[:3], self.mp_bounds_low, self.mp_bounds_high)
        return action

    def step(self, action, get_intermediate_frames=False):
        if self.plan_to_learned_goals or self.planner_only_actions:
            if self.take_planner_step:
                target_pos = self.get_target_pos()
                if self.learn_residual:
                    pos = action[:3] + target_pos
                    if self.clamp_actions:
                        pos = self.clamp_planner_action_mp_space_bounds(pos)
                else:
                    pos = action[:3] + self._eef_xpos
                    if self.clamp_actions:
                        pos = self.clamp_planner_action_mp_space_bounds(pos)
                if self.planner_command_orientation:
                    rot_delta = euler2mat(action[3:6])
                    quat = mat2quat(rot_delta @ quat2mat(self.reset_ori))
                else:
                    quat = self.reset_ori
                action = action.astype(np.float64)
                # quat = quat / np.linalg.norm(quat) # might be necessary for MP code?
                is_grasped = self.check_grasp()
                if self.teleport_instead_of_mp:
                    # make gripper fully open at start
                    pos = backtracking_search_from_goal(
                        self,
                        ignore_object_collision=is_grasped,
                        start_pos=self._eef_xpos,
                        start_ori=self._eef_xquat,
                        goal_pos=pos,
                        ori=quat,
                        qpos=self.reset_qpos,
                        qvel=self.reset_qvel,
                        movement_fraction=0.01,
                        is_grasped=is_grasped,
                    )
                    self.hasnt_teleported = False
                    o = self._get_obs()
                else:
                    o = mp_to_point(
                        self,
                        self.ik_controller_config,
                        self.osc_controller_config,
                        np.concatenate((pos, quat), dtype=np.float64),
                        qpos=self.reset_qpos,
                        qvel=self.reset_qvel,
                        grasp=is_grasped,
                        ignore_object_collision=is_grasped,
                        planning_time=self.planning_time,
                        get_intermediate_frames=get_intermediate_frames,
                        backtrack_movement_fraction=self.backtrack_movement_fraction,
                    )
                    o = self._get_obs()
                r, i = self.evaluate_state(o, action)
                r = r * self.reward_scale
                d = False
                self.take_planner_step = False
                self.high_level_step += 1
            else:
                o, r, d, i = self._wrapped_env.step(action)
                self.current_ll_policy_steps += 1
                if self.current_ll_policy_steps == self.num_ll_actions_per_hl_action:
                    self.take_planner_step = True
                    self.current_ll_policy_steps = 0
                self.num_steps += 1
            # print(self.take_planner_step, self.ep_step_ctr)
            self.ep_step_ctr += 1
        else:
            o, r, d, i = self._wrapped_env.step(action)
            self.num_steps += 1
            self.ep_step_ctr += 1
            if self.hasnt_teleported:
                is_grasped = self.check_grasp(
                    verify_stable_grasp=self.verify_stable_grasp
                )
            else:
                is_grasped = False
            if (self.teleport_on_grasp and is_grasped) and self.use_teleports_in_step:
                target_pos = self.get_target_pos_no_planner()
                if self.teleport_instead_of_mp:
                    set_robot_based_on_ee_pos(
                        self,
                        target_pos,
                        self.reset_ori,
                        self.reset_qpos,
                        self.reset_qvel,
                        is_grasped=is_grasped,
                    )
                    self.hasnt_teleported = False
                    print(
                        "distance to goal: ",
                        np.linalg.norm(target_pos - self._eef_xpos),
                    )
                else:
                    mp_to_point(
                        self,
                        self.ik_controller_config,
                        self.osc_controller_config,
                        np.concatenate((target_pos, self.reset_ori)).astype(np.float64),
                        qpos=self.reset_qpos,
                        qvel=self.reset_qvel,
                        grasp=is_grasped,
                        ignore_object_collision=is_grasped,
                        planning_time=self.planning_time,
                        get_intermediate_frames=get_intermediate_frames,
                        backtrack_movement_fraction=self.backtrack_movement_fraction,
                    )
                # TODO: should re-compute reward here so it is clear what action caused high reward
                if self.recompute_reward_post_teleport:
                    r += self.env.reward()
        i["grasped"] = float(self.check_grasp())
        i["num_steps"] = self.num_steps
        if not self.teleport_instead_of_mp:
            # add in planner logs
            i["mp_mse"] = self.mp_mse
            i["num_failed_solves"] = self.num_failed_solves
            i["goal_error"] = self.goal_error
        if self.add_grasped_to_obs:
            o = np.concatenate((o, np.array([i["grasped"]])))
        r += self.slack_reward
        if self.predict_done_actions:
            d = action[-1] > 0
        d = self.update_done_info_based_on_termination(i, d)
        if self.terminate_on_last_state:
            d = self.ep_step_ctr == self.horizon
        return o, r, d, i
