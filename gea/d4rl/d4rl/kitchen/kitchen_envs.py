"""Environments using kitchen and Franka robot."""
import numpy as np
from gym import spaces
from gym.spaces.box import Box

from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1

OBS_ELEMENT_INDICES = {
    "bottom left burner": np.array([11]), #correct
    "bottom right burner": np.array([9]),
    "top burner": np.array([15]), #correct
    "top right burner": np.array([13]), #correct
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "left hinge cabinet": np.array([20]),
    "hinge cabinet": np.array([21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
    "close hinge cabinet": np.array([13, 21]),
    "close microwave": np.array([22, 23, 24, 25, 26, 27, 28, 29]),
    "close slide": np.array([15, 19]),
}
OBS_ELEMENT_GOALS = {
    "bottom left burner": np.array([-0.92]),
    "bottom right burner": np.array([-0.92]),
    "top burner": np.array([-0.92]),
    "top right burner": np.array([-0.92]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "left hinge cabinet": np.array([-1.45]),
    "hinge cabinet": np.array([1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
    "close hinge cabinet": np.array([-0.92, 0.0]),
    "close microwave": np.array([0., -0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
    "close slide": np.array([-0.92, 0.0]),
}
BONUS_THRESH = 0.3


class KitchenBase(KitchenTaskRelaxV1):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True
    OBS_ELEMENT_INDICES = {
        "bottom left burner": np.array([11]),
        "bottom right burner": np.array([9]),
        "top burner": np.array([15]),
        "top right burner": np.array([13]),
        "light switch": np.array([17, 18]),
        "slide cabinet": np.array([19]),
        "left hinge cabinet": np.array([20]),
        "hinge cabinet": np.array([21]),
        "microwave": np.array([22]),
        "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
        "close hinge cabinet": np.array([13, 21]),
        "close microwave": np.array([22, 23, 24, 25, 26, 27, 28, 29]),
        "close slide": np.array([15, 19]),
    }
    OBS_ELEMENT_GOALS = {
        "bottom left burner": np.array([-0.92]),
        "bottom right burner": np.array([-0.92]),
        "top burner": np.array([-0.92]),
        "top right burner": np.array([-0.92]),
        "light switch": np.array([-0.69, -0.05]),
        "slide cabinet": np.array([0.37]),
        "left hinge cabinet": np.array([-1.45]),
        "hinge cabinet": np.array([1.45]),
        "microwave": np.array([-0.75]),
        "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
        "close hinge cabinet": np.array([-0.92, 0.0]),
        "close microwave": np.array([0., -0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
        "close slide": np.array([-0.92, 0.0]),
    }
        
    def __init__(self, dense=True, use_combined_action_space=False, **kwargs):
        self.tasks_to_complete = [_ for _ in self.TASK_ELEMENTS]
        self.dense = dense
        super(KitchenBase, self).__init__(**kwargs)
        combined_action_space_low = -1.4 * np.ones(self.max_arg_len)
        combined_action_space_high = 1.4 * np.ones(self.max_arg_len)
        self.combined_action_space = Box(
            combined_action_space_low, combined_action_space_high, dtype=np.float32
        )
        self.use_combined_action_space = use_combined_action_space
        if self.use_combined_action_space and self.control_mode == "primitives":
            self.action_space = self.combined_action_space
            if not self.fixed_schema:
                act_lower_primitive = np.zeros(self.num_primitives)
                act_upper_primitive = np.ones(self.num_primitives)
                act_lower = np.concatenate((act_lower_primitive, self.action_space.low))
                act_upper = np.concatenate(
                    (
                        act_upper_primitive,
                        self.action_space.high,
                    )
                )
                self.action_space = Box(act_lower, act_upper, dtype=np.float32)

    def _get_obs(self):
        t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(
            self, robot_noise_ratio=self.robot_noise_ratio
        )

        self.obs_dict = {}
        self.obs_dict["t"] = t
        self.obs_dict["qp"] = qp
        self.obs_dict["qv"] = qv
        self.obs_dict["obj_qp"] = obj_qp
        self.obs_dict["obj_qv"] = obj_qv
        self.obs_dict["goal"] = self.goal
        if self.image_obs:
            img = self.render(mode="rgb_array")
            img = img.transpose(2, 0, 1).flatten()
            if self.proprioception:
                if not self.initializing:
                    proprioceptive_obs = np.concatenate(
                        (
                            qp,
                            self.get_ee_pose(),
                            self.get_ee_quat(),
                        )
                    )
                    if self.normalize_proprioception_obs:
                        proprioceptive_obs /= np.linalg.norm(proprioceptive_obs)
                    return np.concatenate((img, proprioceptive_obs))
                else:
                    return img
            return img

        else:
            return np.concatenate(
                [self.obs_dict["qp"], self.obs_dict["obj_qp"], self.obs_dict["goal"]]
            )

    def _get_task_goal(self):
        new_goal = np.zeros_like(self.goal)
        for element in self.TASK_ELEMENTS:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal
        return new_goal

    def reset_model(self):
        self.tasks_to_complete = [_ for _ in self.TASK_ELEMENTS]
        self.episodic_cumulative_reward = 0
        return super(KitchenBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        next_q_obs = obs_dict["qp"]
        next_obj_obs = obs_dict["obj_qp"]
        idx_offset = len(next_q_obs)
        completions = []
        dense = 0
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] - OBS_ELEMENT_GOALS[element]
            )
            dense += -1 * distance  # reward must be negative distance for RL
            is_grasped = True

            if element == 'slide cabinet':
                is_grasped = False
                if not self.initializing:
                    for i in range(1, 6):
                        obj_pos = self.get_site_xpos("schandle{}".format(i))
                        left_pad = self.get_site_xpos('leftpad')
                        right_pad = self.get_site_xpos('rightpad')
                        within_sphere_left = np.linalg.norm(obj_pos-left_pad) < .07
                        within_sphere_right = np.linalg.norm(obj_pos-right_pad) < .07
                        right = right_pad[0] < obj_pos[0]
                        left = obj_pos[0] < left_pad[0]
                        if right and left and within_sphere_right and within_sphere_left:
                            is_grasped = True
            if element == 'top burner':
                is_grasped = False
                if not self.initializing:
                    obj_pos = self.get_site_xpos("tlbhandle")
                    left_pad = self.get_site_xpos('leftpad')
                    right_pad = self.get_site_xpos('rightpad')
                    within_sphere_left = np.linalg.norm(obj_pos-left_pad) < .035
                    within_sphere_right = np.linalg.norm(obj_pos-right_pad) < .04
                    right = right_pad[0] < obj_pos[0]
                    left = obj_pos[0] < left_pad[0]
                    if within_sphere_right and within_sphere_left and right and left:
                        is_grasped = True
            if element == 'top right burner':
                is_grasped = False
                if not self.initializing:
                    obj_pos = self.get_site_xpos("trbhandle")
                    left_pad = self.get_site_xpos('leftpad')
                    right_pad = self.get_site_xpos('rightpad')
                    within_sphere_left = np.linalg.norm(obj_pos-left_pad) < .035
                    within_sphere_right = np.linalg.norm(obj_pos-right_pad) < .04
                    right = right_pad[0] < obj_pos[0]
                    left = obj_pos[0] < left_pad[0]
                    if within_sphere_right and within_sphere_left and right and left:
                        is_grasped = True
            if element == 'microwave':
                is_grasped = False
                if not self.initializing:
                    for i in range(1, 6):
                        obj_pos = self.get_site_xpos("mchandle{}".format(i))
                        left_pad = self.get_site_xpos('leftpad')
                        right_pad = self.get_site_xpos('rightpad')
                        within_sphere_left = np.linalg.norm(obj_pos-left_pad) < .05
                        within_sphere_right = np.linalg.norm(obj_pos-right_pad) < .05
                        if right_pad[0] < obj_pos[0] and obj_pos[0] < left_pad[0] and within_sphere_right and within_sphere_left:
                            is_grasped = True
            if element == 'hinge cabinet':
                is_grasped = False
                if not self.initializing:
                    for i in range(1, 6):
                        obj_pos = self.get_site_xpos("hchandle{}".format(i))
                        left_pad = self.get_site_xpos('leftpad')
                        right_pad = self.get_site_xpos('rightpad')
                        within_sphere_left = np.linalg.norm(obj_pos-left_pad) < .06
                        within_sphere_right = np.linalg.norm(obj_pos-right_pad) < .06
                        if right_pad[0] < obj_pos[0] and obj_pos[0] < left_pad[0] and within_sphere_right:
                            is_grasped = True
            if element == 'left hinge cabinet':
                is_grasped = False
                if not self.initializing:
                    for i in range(1, 6):
                        obj_pos = self.get_site_xpos("hchandle_left{}".format(i))
                        left_pad = self.get_site_xpos('leftpad')
                        right_pad = self.get_site_xpos('rightpad')
                        within_sphere_left = np.linalg.norm(obj_pos-left_pad) < .06
                        within_sphere_right = np.linalg.norm(obj_pos-right_pad) < .06
                        if right_pad[0] < obj_pos[0] and obj_pos[0] < left_pad[0] and within_sphere_right:
                            is_grasped = True
            if element == 'light switch':
                is_grasped = False
                if not self.initializing:
                    for i in range(1, 4):
                        obj_pos = self.get_site_xpos("lshandle{}".format(i))
                        left_pad = self.get_site_xpos('leftpad')
                        right_pad = self.get_site_xpos('rightpad')
                        within_sphere_left = np.linalg.norm(obj_pos-left_pad) < .045
                        within_sphere_right = np.linalg.norm(obj_pos-right_pad) < .03
                        if within_sphere_right and within_sphere_left:
                            is_grasped = True
            if not self.use_grasp_rewards:
                is_grasped = True # is_grasped is basically ignored if we are not using grasp_rewards
            complete = distance < BONUS_THRESH and is_grasped
            if complete:
                completions.append(element)
            else:
                break
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict["bonus"] = bonus
        reward_dict["r_total"] = bonus
        if self.dense:
            reward_dict["r_total"] = dense
        score = bonus
        return reward_dict, score

    def step(
        self,
        a,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        obs, reward, done, env_info = super(KitchenBase, self).step(
            a,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        self.episodic_cumulative_reward += reward

        if self.TERMINATE_ON_TASK_COMPLETE:
            done = not self.tasks_to_complete
        self.update_info(env_info)
        return obs, reward, done, env_info

    def update_info(self, info):
        next_q_obs = self.obs_dict["qp"]
        next_obj_obs = self.obs_dict["obj_qp"]
        idx_offset = len(next_q_obs)
        if self.initializing:
            self.per_task_cumulative_reward = {
                k: 0.0 for k in OBS_ELEMENT_INDICES.keys()
            }
        for element in OBS_ELEMENT_INDICES.keys():
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] - OBS_ELEMENT_GOALS[element]
            )
            info[element + " distance to goal"] = distance
            info[element + " success"] = float(distance < BONUS_THRESH)
            success = float(distance < BONUS_THRESH)
            self.per_task_cumulative_reward[element] += success
            info[element + " cumulative reward"] = self.per_task_cumulative_reward[
                element
            ]
            info[element + " success"] = success
        info["coverage"] = self.coverage_grid.sum() / (
            np.prod(self.coverage_grid.shape)
        )
        info["episodic cumulative reward"] = self.episodic_cumulative_reward
        return info

class KitchenMS5V0(KitchenBase):
    TASK_ELEMENTS = ["microwave", "kettle", "light switch", "top burner", "slide cabinet"]
    REMOVE_TASKS_WHEN_COMPLETE = True

class KitchenMS6V0(KitchenBase):
    TASK_ELEMENTS = [
        "microwave", 
        "kettle", 
        "light switch", 
        "top burner", 
        "slide cabinet", 
        "bottom right burner",
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True 

class KitchenMS7V0(KitchenBase):
    TASK_ELEMENTS = [
        "kettle",
        "light switch",
        "slide cabinet",
        "top burner",
        "microwave",
        "bottom right burner",
        "hinge cabinet",
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True 

class KitchenMS8V0(KitchenBase):
    TASK_ELEMENTS = [
        "microwave", 
        "kettle", 
        "light switch", 
        "top burner", 
        "slide cabinet", 
        "bottom right burner",
        "hinge cabinet",
        "bottom left burner",
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True 

class KitchenMS10V0(KitchenBase):
    TASK_ELEMENTS = [
        "microwave",
        "kettle",
        "close microwave",
        "slide cabinet",
        "top burner",
        "close slide",
        "hinge cabinet",
        "light switch",
        "close hinge cabinet",
        "bottom right burner"
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True 

class KitchenMS10V1(KitchenBase):
    TASK_ELEMENTS = [
        "kettle",
        "light switch",
        "slide cabinet",
        "top burner",
        "microwave",
        "bottom right burner",
        "hinge cabinet",
        "top right burner",
        "bottom left burner",
        "close microwave",
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True 

class KitchenMS10V2(KitchenBase):
    TASK_ELEMENTS = [
        "kettle",
        "light switch",
        "slide cabinet",
        "top burner",
        "bottom right burner",
        "top right burner",
        "bottom left burner",
        "microwave",
        "hinge cabinet",
        "close microwave",
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True 

class KitchenKettleLightBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["kettle", "light switch", "top burner"]
    REMOVE_TASKS_WHEN_COMPLETE = True 

class KitchenMicrowaveKettleLightTopLeftBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["microwave", "kettle", "light switch", "top burner"]
    REMOVE_TASKS_WHEN_COMPLETE = True

class KitchenKettleBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["kettle", "top burner"]
    REMOVE_TASKS_WHEN_COMPLETE = True 


class KitchenHingeSlideBottomLeftBurnerLightV0(KitchenBase):
    TASK_ELEMENTS = [
        "hinge cabinet",
        "slide cabinet",
        "bottom left burner",
        "light switch",
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True


class KitchenMicrowaveV0(KitchenBase):
    TASK_ELEMENTS = ["microwave"]

    def __init__(self, delta=0, **kwargs):
        super(KitchenMicrowaveV0, self).__init__(**kwargs)
        self.step_to_primitive_name = {
            0: "drop",
            1: "angled_x_y_grasp",
            2: "move_backward",
            3: "no_op",
            4: "no_op",
        }
        if not self.use_combined_action_space and self.control_mode == "primitives":
            action_low = np.array(
                [
                    -np.pi / 6,
                    -0.3,
                    0.95,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.55,
                    0,
                    0,
                    0,
                    0.6,
                    0.0,
                ]
            )

            action_high = np.array(
                [
                    -np.pi / 6,
                    -0.3,
                    0.95,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.55,
                    0,
                    0,
                    0,
                    0.6,
                    0.0,
                ]
            )
            action_low -= delta
            action_high += delta
            if not self.fixed_schema:
                act_lower_primitive = np.zeros(self.num_primitives)
                act_upper_primitive = np.ones(self.num_primitives)
                action_low = np.concatenate((act_lower_primitive, action_low))
                action_high = np.concatenate((act_upper_primitive, action_high))
            self.action_space = Box(action_low, action_high, dtype=np.float32)


class KitchenKettleV0(KitchenBase):
    TASK_ELEMENTS = ["kettle"]

    def __init__(self, delta=0, **kwargs):
        super(KitchenKettleV0, self).__init__(**kwargs)
        self.step_to_primitive_name = {
            0: "drop",
            1: "angled_x_y_grasp",
            2: "move_delta_ee_pose",
            3: "drop",
            4: "open_gripper",
        }
        if not self.use_combined_action_space and self.control_mode == "primitives":
            action_low = np.array(
                [
                    0,
                    0.15,
                    0.7,
                    0.25,
                    1.1,
                    0.25,
                    0,
                    0,
                    0.25,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                ]
            )

            action_high = np.array(
                [
                    0,
                    0.15,
                    0.7,
                    0.25,
                    1.1,
                    0.25,
                    0,
                    0,
                    0.5,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                ]
            )
            action_low -= delta
            action_high += delta
            if not self.fixed_schema:
                act_lower_primitive = np.zeros(self.num_primitives)
                act_upper_primitive = np.ones(self.num_primitives)
                action_low = np.concatenate((act_lower_primitive, action_low))
                action_high = np.concatenate((act_upper_primitive, action_high))
            self.action_space = Box(action_low, action_high, dtype=np.float32)


class KitchenBottomLeftBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["bottom left burner"]

    def __init__(self, delta=0.0, **kwargs):
        super(KitchenBottomLeftBurnerV0, self).__init__(**kwargs)

        self.step_to_primitive_name = {
            0: "lift",
            1: "angled_x_y_grasp",
            2: "rotate_about_y_axis",
            3: "no_op",
            4: "no_op",
        }
        if not self.use_combined_action_space and self.control_mode == "primitives":
            action_low = np.array(
                [
                    0,
                    0.55,
                    1.1,
                    0.0,
                    0.0,
                    0.0,
                    -np.pi / 4,
                    0.3,
                    0.0,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                ]
            )

            action_high = np.array(
                [
                    0,
                    0.55,
                    1.1,
                    0.0,
                    0.0,
                    0.0,
                    -np.pi / 4,
                    0.3,
                    0.0,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                ]
            )
            action_low -= delta
            action_high += delta
            if not self.fixed_schema:
                act_lower_primitive = np.zeros(self.num_primitives)
                act_upper_primitive = np.ones(self.num_primitives)
                action_low = np.concatenate((act_lower_primitive, action_low))
                action_high = np.concatenate((act_upper_primitive, action_high))
            self.action_space = Box(action_low, action_high, dtype=np.float32)


class KitchenTopLeftBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["top burner"]

    def __init__(self, delta=0, **kwargs):
        super(KitchenTopLeftBurnerV0, self).__init__(**kwargs)

        self.step_to_primitive_name = {
            0: "lift",
            1: "angled_x_y_grasp",
            2: "rotate_about_y_axis",
            3: "no_op",
            4: "no_op",
        }
        if not self.use_combined_action_space and self.control_mode == "primitives":
            action_low = np.array(
                [
                    0,
                    0.5,
                    1,
                    0.0,
                    0.0,
                    0.0,
                    -np.pi / 4,
                    0.6,
                    0.0,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                ]
            )

            action_high = np.array(
                [
                    0,
                    0.5,
                    1,
                    0.0,
                    0.0,
                    0.0,
                    -np.pi / 4,
                    0.6,
                    0.0,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                ]
            )
            action_low -= delta
            action_high += delta
            if not self.fixed_schema:
                act_lower_primitive = np.zeros(self.num_primitives)
                act_upper_primitive = np.ones(self.num_primitives)
                action_low = np.concatenate((act_lower_primitive, action_low))
                action_high = np.concatenate((act_upper_primitive, action_high))
            self.action_space = Box(action_low, action_high, dtype=np.float32)


class KitchenSlideCabinetV0(KitchenBase):
    TASK_ELEMENTS = ["slide cabinet"]

    def __init__(self, delta=0, **kwargs):
        super(KitchenSlideCabinetV0, self).__init__(**kwargs)
        self.step_to_primitive_name = {
            0: "lift",
            1: "angled_x_y_grasp",
            2: "move_right",
            3: "no_op",
            4: "no_op",
        }
        if not self.use_combined_action_space and self.control_mode == "primitives":
            action_low = np.array(
                [
                    0.0,
                    0.7,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1,
                    0.0,
                    0.0,
                    0.6,
                    0.0,
                    0.0,
                    0.0,
                ]
            )

            action_high = np.array(
                [
                    0.0,
                    0.7,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1,
                    0.0,
                    0.0,
                    0.6,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            action_low -= delta
            action_high += delta
            if not self.fixed_schema:
                act_lower_primitive = np.zeros(self.num_primitives)
                act_upper_primitive = np.ones(self.num_primitives)
                action_low = np.concatenate((act_lower_primitive, action_low))
                action_high = np.concatenate((act_upper_primitive, action_high))
            self.action_space = Box(action_low, action_high, dtype=np.float32)


class KitchenHingeCabinetV0(KitchenBase):
    TASK_ELEMENTS = ["hinge cabinet"]

    def __init__(self, delta=0, **kwargs):
        super(KitchenHingeCabinetV0, self).__init__(**kwargs)

        self.step_to_primitive_name = {
            0: "lift",
            1: "angled_x_y_grasp",
            2: "move_delta_ee_pose",
            3: "rotate_about_x_axis",
            4: "no_op",
        }
        if not self.use_combined_action_space and self.control_mode == "primitives":
            action_low = np.array(
                [
                    -np.pi / 6,
                    -0.3,
                    1.4,
                    0.5,
                    -1,
                    0.0,
                    0.0,
                    1,
                    0.0,
                    0,
                    1,
                    0,
                    0.3,
                    1.0,
                ]
            )

            action_high = np.array(
                [
                    -np.pi / 6,
                    -0.3,
                    1.4,
                    0.5,
                    -1,
                    0.0,
                    0.0,
                    1,
                    0.0,
                    0,
                    1,
                    0,
                    0.3,
                    1.0,
                ]
            )
            action_low -= delta
            action_high += delta
            if not self.fixed_schema:
                act_lower_primitive = np.zeros(self.num_primitives)
                act_upper_primitive = np.ones(self.num_primitives)
                action_low = np.concatenate((act_lower_primitive, action_low))
                action_high = np.concatenate((act_upper_primitive, action_high))
            self.action_space = Box(action_low, action_high, dtype=np.float32)


class KitchenLightSwitchV0(KitchenBase):
    TASK_ELEMENTS = ["light switch"]

    def __init__(self, delta=0, **kwargs):
        super(KitchenLightSwitchV0, self).__init__(**kwargs)
        self.step_to_primitive_name = {
            0: "close_gripper",
            1: "lift",
            2: "move_right",
            3: "move_forward",
            4: "move_left",
        }
        if not self.use_combined_action_space and self.control_mode == "primitives":
            action_low = np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.6,
                    0.0,
                    0.45,
                    0.45,
                    1.25,
                    0.0,
                    0.0,
                ]
            )

            action_high = np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.6,
                    0.0,
                    0.45,
                    0.45,
                    1.25,
                    0.0,
                    0.0,
                ]
            )
            action_low -= delta
            action_high += delta
            if not self.fixed_schema:
                act_lower_primitive = np.zeros(self.num_primitives)
                act_upper_primitive = np.ones(self.num_primitives)
                action_low = np.concatenate((act_lower_primitive, action_low))
                action_high = np.concatenate((act_upper_primitive, action_high))
            self.action_space = Box(action_low, action_high, dtype=np.float32)

class KitchenKettleLightBurnerSliderV0(KitchenBase):
    TASK_ELEMENTS = ["kettle", "light switch", "top burner", "slide cabinet"]


class KitchenKettleLightBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["kettle", "light switch", "top burner"]


class KitchenKettleBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["kettle", "top burner"]


class KitchenLightBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["light switch", "top burner"]


class KitchenMicrowaveKettleLightBurnerSliderV0(KitchenBase):
    TASK_ELEMENTS = [
        "microwave",
        "kettle",
        "light switch",
        "top burner",
        "slide cabinet",
    ]

class KitchenHingeMicrowaveKettleLightBurnerSliderV0(KitchenBase):
    TASK_ELEMENTS = [
        "microwave",
        "kettle",
        "light switch",
        "top burner",
        "slide cabinet",
        "hinge cabinet",
    ]

class KitchenLeftHingeCabinetV0(KitchenBase):
    TASK_ELEMENTS = ["left hinge cabinet"]

class KitchenTopRightBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["top right burner"]

class KitchenBottomRightBurnerV0(KitchenBase):
    TASK_ELEMENTS = ["bottom right burner"]