import random

import dm_env
import metaworld
import numpy as np
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import specs
from gea.cfgs.env_policys import env_name2policy
from gea.environments.wrappers import (
    ActionDTypeWrapper,
    ActionRepeatWrapper,
    ExtendedTimeStepWrapper,
    FrameStackWrapper,
    PixelsWrapper,
    Render_Wrapper,
    get_env_action_spec,
    get_env_observation_spec,
)
from gea.psl.metaworld_mp_env import MetaworldPSLEnv

ENV_CAMERA_DICT = {
    "assembly-v2": "gripperPOVneg",
    "disassemble-v2": "gripperPOVneg",
    "peg-insert-side-v2": "gripperPOVneg",
    "bin-picking-v2": "gripperPOVpos",
    "hammer-v2": "gripperPOVpos",
}


def get_proprioceptive_spec(spec, num_proprioceptive_features):
    return specs.BoundedArray(
        shape=(num_proprioceptive_features,),
        dtype=spec.dtype,
        minimum=spec.minimum[0:num_proprioceptive_features],
        maximum=spec.maximum[0:num_proprioceptive_features],
        name="observation",
    )


class MT_Wrapper(dm_env.Environment):
    def __init__(
        self,
        env_name: str,
        text_plan,
        discount=1.0,
        seed=None,
        proprioceptive_state=True,
        psl=False,
        use_mp=False,
        use_sam_segmentation=False,
        use_vision_pose_estimation=False,
        is_eval=False,
    ):
        self.discount = discount
        # self.mt = metaworld.ML10(env_name, seed=seed)
        if env_name =='MT10':
            self.mt = metaworld.MT10(seed=seed)
        elif env_name =='MT50':
            self.mt = metaworld.MT50(seed=seed)
        elif env_name =='ML45':
            self.mt = metaworld.ML45(seed=seed)
        elif env_name =='MTV':
            self.mt = metaworld.MTV(seed=seed)
        elif env_name =='MLGrasp':
            self.mt = metaworld.MLGrasp(seed=seed)
        elif env_name =='MLV':
            self.mt = metaworld.MLV(seed=seed)
        else:
            self.mt = metaworld.MT1(env_name, seed=seed)
        if env_name in ['ML45',"MLGrasp", "MLV"] and is_eval:
            self.all_envs = {
                name: env_cls() for name, env_cls in self.mt.test_classes.items()
            }
        else:
            self.all_envs = {
                name: env_cls() for name, env_cls in self.mt.train_classes.items()
            }
        self.sample_weights = {
            name: 1 for name in self.all_envs.keys()
        }
        # self.all_envs = {
        #     # "hammer-v2": metaworld.ML1("hammer-v2",seed=seed).train_classes["hammer-v2"](),
        #     "assembly-v2": metaworld.ML1("assembly-v2",seed=seed).train_classes["assembly-v2"](),
        #     "bin-picking-v2": metaworld.ML1("bin-picking-v2",seed=seed).train_classes["bin-picking-v2"](),
        # }
        # self.all_tasks = {
        #     "hammer-v2": metaworld.ML1("hammer-v2",seed=seed).train_tasks,
        #     "assembly-v2": metaworld.ML1("assembly-v2",seed=seed).train_tasks,
        #     "bin-picking-v2": metaworld.ML1("bin-picking-v2",seed=seed).train_tasks,
        # }
        mp_env_kwargs = dict(
            vertical_displacement=0.05,
            teleport_instead_of_mp=not use_mp,
            mp_bounds_low=(-0.2, 0.6, 0.0),
            mp_bounds_high=(0.2, 0.8, 0.2),
            backtrack_movement_fraction=0.001,
            grip_ctrl_scale=0.0025,
            planning_time=20,
            max_path_length=200,
            use_vision_pose_estimation=use_vision_pose_estimation,
            use_sam_segmentation=use_sam_segmentation,
            text_plan=text_plan,
            is_eval=is_eval
        )
        self.psl = psl
        self.is_eval=is_eval
        self.mp_env_kwargs = mp_env_kwargs
        self.env_name, self._env = self.sample_env()
        if psl:
            self._env = MetaworldPSLEnv(
                self._env,
                self.env_name,
                **mp_env_kwargs,
            )
            
        self.physics = Render_Wrapper(self._env.sim.render)
        self._reset_next_step = True
        self.current_step = 0
        self.proprioceptive_state = proprioceptive_state
        self.NUM_PROPRIOCEPTIVE_STATES = 7
        self._observation_spec = None
        self._action_spec = None
        self.robot_segmentation_ids = list(range(8, 35))

    def sample_env(self):
        if self.sample_weights is not None:
            names = list(self.sample_weights.keys())
            weights = list(self.sample_weights.values())
            sampled_name = random.choices(names, weights=weights, k=1)[0]
            return sampled_name, self.all_envs[sampled_name]
        return random.choice(list(self.all_envs.items()))

    def sample_task(self):
        # return random.choice(
        #     [task for task in self.all_tasks[self.env_name] if task.env_name == self.env_name]
        # )
        if self.is_eval == False:
            return random.choice(
                [task for task in self.mt.train_tasks if task.env_name == self.env_name]
            )
        else:
            try:
                return random.choice(
                    [task for task in self.mt.test_tasks if task.env_name == self.env_name]
                )
            except:
                return random.choice(
                [task for task in self.mt.train_tasks if task.env_name == self.env_name]
            )

    def reset(self) -> dm_env.TimeStep:
        self._reset_next_step = False
        self.current_step = 0
        self.env_name, self._env = self.sample_env()
        self.policy = env_name2policy[self.env_name]()
        if self.psl:
            self._env = MetaworldPSLEnv(
                self._env, env_name=self.env_name, **self.mp_env_kwargs
            )
        self.physics = Render_Wrapper(self._env.sim.render)
        task = self.sample_task()
        self._env.set_task(task)
        observation = self._env.reset()
        self.origin_obs = observation
        if self.proprioceptive_state:
            observation = self.get_proprioceptive_observation(observation)
        observation = observation.astype(self._env.observation_space.dtype)
        self.current_step += 1
        return dm_env.restart(observation)

    def step(self, action: int) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()

        observation, reward, _, info = self._env.step(action)
        self.origin_obs = observation
        self.current_step += 1
        if self.proprioceptive_state:
            observation = self.get_proprioceptive_observation(observation)
        observation = observation.astype(self._env.observation_space.dtype)

        reward_dict = {"reward": reward, "success": info["success"]}

        if self.current_step == self._env.max_path_length:
            self._reset_next_step = True
            return dm_env.truncation(reward_dict, observation, self.discount)

        return dm_env.transition(reward_dict, observation, self.discount)

    def get_proprioceptive_observation(self, observation):
        observation = observation[0 : self.NUM_PROPRIOCEPTIVE_STATES]
        return observation

    def observation_spec(self) -> specs.BoundedArray:
        if self._observation_spec is not None:
            return self._observation_spec

        spec = None
        for _, env in self.all_envs.items():
            if spec is None:
                spec = get_env_observation_spec(env)
                continue
            assert spec == get_env_observation_spec(
                env
            ), "The observation spec should match for all environments"

        if self.proprioceptive_state:
            spec = get_proprioceptive_spec(spec, self.NUM_PROPRIOCEPTIVE_STATES)

        self._observation_spec = spec
        return self._observation_spec

    def action_spec(self) -> specs.BoundedArray:
        if self._action_spec is not None:
            return self._action_spec

        spec = None
        for _, env in self.all_envs.items():
            if spec is None:
                spec = get_env_action_spec(env)
                continue
            assert spec == get_env_action_spec(
                env
            ), "The action spec should match for all environments"

        self._action_spec = spec
        return self._action_spec

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self._env, name)


def make_metaworld(
    name,
    frame_stack,
    action_repeat,
    discount,
    seed,
    camera_name,
    depth,
    mask,
    dino,
    text_plan,
    psl=False,
    use_mp=False,
    use_sam_segmentation=False,
    use_vision_pose_estimation=False,
    is_eval=False,
):
    assert not (dino and (mask or depth))
    env = MT_Wrapper(
        env_name=name,
        discount=discount,
        seed=seed,
        proprioceptive_state=True,
        psl=psl,
        use_mp=use_mp,
        use_sam_segmentation=use_sam_segmentation,
        text_plan=text_plan,
        use_vision_pose_estimation=use_vision_pose_estimation,
        is_eval=is_eval
    )
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat, use_metaworld_reward_dict=True)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    frame_keys = []
    rgb_key = "pixels"
    frame_keys.append(rgb_key)
    render_kwargs = dict(height=168, width=168, mode="offscreen", camera_name=camera_name,depth=depth)
    env = PixelsWrapper(
        env, pixels_only=False, render_kwargs=render_kwargs, observation_key=rgb_key, mask=mask,dino=dino
    )
    frame_keys.append('origin')
    env = FrameStackWrapper(env, frame_stack, frame_keys)
    env = ExtendedTimeStepWrapper(env, has_success_metric=True)
    return env
