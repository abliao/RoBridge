import random
from collections import deque
from typing import Any, NamedTuple

import collections
from dm_env import specs
from gym import spaces
import dm_env
import numpy as np
import torch
from dm_control.mujoco import engine
from dm_env import StepType, specs
from skimage import morphology
from skimage.filters import gaussian
from skimage.transform import rescale

env_mask_id_dict = {
    "metaworld_hammer-v2":{'obj':['hammer',[36,37,38,39,40]],'goal':['nail_link',53]},
    "metaworld_bin-picking-v2":{'obj':['objA',36],'goal':['binB',39]},
    "metaworld_assembly-v2":{'obj':['asmbly_peg',[36,37,38]],'goal':['peg',49]},
    "metaworld_disassemble-v2":{'obj':['asmbly_peg',[36,37,38]]},

    "metaworld_basketball-v2":{'obj':['basketball',37],'goal':['basketball',36]},
    "metaworld_button-press-topdown-v2":{'obj':['button',43]},
    "metaworld_button-press-topdown-wall-v2":{'obj':['button',43]},
    "metaworld_button-press-v2":{'obj':['button',43]},
    "metaworld_button-press-wall-v2":{'obj':['button',43]},
    "metaworld_coffee-button-v2":{'obj':['cmbutton',54]},
    "metaworld_box-close-v2":{'obj':['top_link',43],'goal':['box',36]},
    "metaworld_dial-turn-v2":{'obj':['dial',37]},
    "metaworld_door-close-v2":{'obj':['door_link',43]},
    "metaworld_door-open-v2":{'obj':['door_link',43]},
    "metaworld_door-lock-v2":{'obj':['lock_link',54]},
    "metaworld_door-unlock-v2":{'obj':['lock_link',54]},
    "metaworld_coffee-push-v2":{'obj':['mug',8]},
    "metaworld_coffee-pull-v2":{'obj':['mug',8]},
    "metaworld_drawer-close-v2":{'obj':['drawer_link',43]},
    "metaworld_drawer-open-v2":{'obj':['drawer_link',43]},
    "metaworld_faucet-close-v2":{'obj':['faucet_link2',41]},
    "metaworld_faucet-open-v2":{'obj':['faucet_link2',41]},
    "metaworld_handle-press-side-v2":{'obj':['handle_link',42]},
    "metaworld_handle-press-v2":{'obj':['handle_link',42]},
    "metaworld_handle-pull-side-v2":{'obj':['handle_link',42]}, 
    "metaworld_lever-pull-v2":{'obj':['lever_link1',42]},
    "metaworld_peg-insert-side-v2":{'obj':['peg',36]},
    "metaworld_hand-insert-v2":{'obj':['obj',41]},
    "metaworld_handle-pull-v2":{'obj':['handle_link',42]},
    "metaworld_pick-place-v2":{'obj':['obj',36]},
    "metaworld_pick-place-wall-v2":{'obj':['obj',36]},
    "metaworld_reach-v2":{},
    "metaworld_pick-out-of-hole-v2":{'obj':['obj',36]},
    "metaworld_plate-slide-v2":{'obj':['puck',36]},
    "metaworld_plate-slide-side-v2":{'obj':['puck',36]},
    "metaworld_plate-slide-back-v2":{'obj':['puck',36]},
    "metaworld_plate-slide-back-side-v2":{'obj':['puck',36]},
    "metaworld_push-v2":{'obj':['obj',36]},
    "metaworld_push-back-v2":{'obj':['obj',36]},
    "metaworld_peg-unplug-side-v2":{'obj':['plug',38]},
    "metaworld_soccer-v2":{'obj':['soccer_ball1',36]},
    "metaworld_stick-push-v2":{'obj':['stick',36],'goal':['handle',41]},
    "metaworld_stick-pull-v2":{'obj':['stick',36],'goal':['handle',41]},
    "metaworld_reach-wall-v2":{},
    "metaworld_sweep-into-v2":{'obj':['obj',41]},
    "metaworld_push-wall-v2":{'obj':['obj',36]},
    "metaworld_window-open-v2":{'obj':['windowb_a',47]},
    "metaworld_window-close-v2":{'obj':['windowb_a',47]},
    "metaworld_shelf-place-v2":{'obj':['obj',36]},
    "metaworld_sweep-v2":{'obj':['obj',36]},
}

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats, use_metaworld_reward_dict=False):
        self._env = env
        self._num_repeats = num_repeats
        self.use_metaworld_reward_dict = use_metaworld_reward_dict

    def step(self, action):
        if self.use_metaworld_reward_dict:
            reward = 0.0
            success = False
            discount = 1.0
            for _ in range(self._num_repeats):
                time_step = self._env.step(action)
                reward += (time_step.reward["reward"] or 0.0) * discount
                success = success or time_step.reward["success"]
                discount *= time_step.discount
                if time_step.last():
                    break
            reward_dict = {"reward": reward, "success": success}
            return time_step._replace(reward=reward_dict, discount=discount)
        else:
            reward = 0.0
            discount = 1.0
            for _ in range(self._num_repeats):
                time_step = self._env.step(action)
                reward += (time_step.reward or 0.0) * discount
                discount *= time_step.discount
                if time_step.last():
                    break
            return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    

class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, frame_keys=None):
        self._env = env
        self._num_frames = num_frames

        if frame_keys is None:
            frame_keys = ["pixels"]
        if not isinstance(frame_keys, list):
            frame_keys = [frame_keys]
        self._frame_keys = frame_keys

        self._frames = [deque([], maxlen=num_frames) for _ in range(len(frame_keys))]
        wrapped_obs_spec = env.observation_spec()
        for key in frame_keys:
            assert key in wrapped_obs_spec

            frame_shape = wrapped_obs_spec[key].shape
            frame_dtype = wrapped_obs_spec[key].dtype
            # remove batch dim
            if len(frame_shape) == 4:
                frame_shape = frame_shape[1:]
            wrapped_obs_spec[key] = specs.BoundedArray(
                shape=np.concatenate(
                    [[frame_shape[2] * num_frames], frame_shape[:2]], axis=0
                ),
                dtype=frame_dtype,
                minimum=0,
                maximum=255,
                name="observation",
            )
        self._obs_spec = wrapped_obs_spec

    def _transform_observation(self, time_step):
        obs = time_step.observation
        for i, key in enumerate(self._frame_keys):
            assert len(self._frames[i]) == self._num_frames
            stacked_frames = np.concatenate(list(self._frames[i]), axis=0)
            obs[key] = stacked_frames
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step, key):
        pixels = time_step.observation[key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        for i, key in enumerate(self._frame_keys):
            pixels = self._extract_pixels(time_step, key)
            for _ in range(self._num_frames):
                self._frames[i].append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        for i, key in enumerate(self._frame_keys):
            pixels = self._extract_pixels(time_step, key)
            self._frames[i].append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

from cfgs.env_ids import env_name2id


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env, has_success_metric=False):
        self._env = env
        self.has_success_metric = has_success_metric

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        if time_step.reward is None and self.has_success_metric:
            reward = {"reward": 0.0, "success": 0}
        elif time_step.reward is None:
            reward = 0.0
        else:
            reward = time_step.reward
        time_step.observation['task'] = np.array([env_name2id[self._env.env_name]],np.int32)
        text_plan = self._env.text_plan[self._env.curr_plan_stage][1]
        action_name2id = {
            "grasp": 0,
            "place": 1,
            "press": 2,
            "push": 3,
            "open": 4,
            "close": 5,
            "turn": 6,
            "reach": 7,
            "pull": 8,
        }
        assert text_plan in action_name2id, f"{text_plan} not in action_name2id"
        time_step.observation['action'] = np.array([action_name2id[text_plan]],np.int32)
        if text_plan in ['grasp','place','press','reach']:
            time_step.observation['orientation'] = np.array([0.,0.,0.]).astype(np.float32)
        else:
            obj = time_step.observation['state'][4:7]
            target = self._env._target_pos.copy()
            orientation = target - obj
            norm = np.linalg.norm(orientation)
            orientation = orientation / norm
            time_step.observation['orientation'] = orientation.astype(np.float32)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=reward,
            discount=time_step.discount or 1.0,
        )

    def observation_spec(self):
        observation_spec = self._env.observation_spec()
        observation_spec['task'] = spaces.Box(low=0, high=999, dtype=np.int32,shape=(1,))
        observation_spec['action'] = spaces.Box(low=0, high=999, dtype=np.int32,shape=(1,))
        observation_spec['orientation'] = spaces.Box(low=-1, high=1, dtype=np.float32,shape=(3,))
        return observation_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class SegmentationToRobotMaskWrapper(dm_env.Environment):
    def __init__(self, env, segmentation_key="segmentation", types_channel=0):
        self._env = env
        self.segmentation_key = segmentation_key
        self.types_channel = types_channel
        assert segmentation_key in env.observation_spec()

        wrapped_obs_spec = env.observation_spec()
        frame_shape = wrapped_obs_spec[segmentation_key].shape
        # remove batch dim
        if len(frame_shape) == 4:
            frame_shape = frame_shape[1:]
        wrapped_obs_spec[segmentation_key] = specs.BoundedArray(
            shape=np.concatenate([frame_shape[:2], [1]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )
        self._obs_spec = wrapped_obs_spec

    def _transform_observation(self, time_step):
        obs = time_step.observation
        seg = obs[self.segmentation_key]

        types = seg[:, :, self.types_channel]
        ids = seg[:, :, 1 - self.types_channel]
        robot_mask = np.logical_and(
            types == const.OBJ_GEOM, np.isin(ids, self._env.robot_segmentation_ids)
        )
        robot_mask = robot_mask.astype(np.uint8)
        robot_mask = robot_mask.reshape(robot_mask.shape[0], robot_mask.shape[1], 1)

        obs[self.segmentation_key] = robot_mask
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class SegmentationFilter(dm_env.Environment):
    def __init__(self, env, segmentation_key="segmentation"):
        self._env = env
        self.segmentation_key = segmentation_key
        self.downsample = 3
        assert segmentation_key in env.observation_spec()

        wrapped_obs_spec = env.observation_spec()
        frame_shape = wrapped_obs_spec[segmentation_key].shape
        # remove batch dim
        if len(frame_shape) == 4:
            frame_shape = frame_shape[1:]
        new_shape = (
            int(frame_shape[0] / self.downsample),
            int(frame_shape[1] / self.downsample),
            frame_shape[2],
        )
        wrapped_obs_spec[segmentation_key] = specs.BoundedArray(
            shape=new_shape, dtype=np.uint8, minimum=0, maximum=255, name="observation"
        )
        self._obs_spec = wrapped_obs_spec

    def _transform_observation(self, time_step):
        obs = time_step.observation
        seg = obs[self.segmentation_key]
        filtered_seg = morphology.binary_opening(seg)[
            :: self.downsample, :: self.downsample
        ]
        obs[self.segmentation_key] = filtered_seg.astype(np.uint8)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class NoisyMaskWrapper(dm_env.Environment):
    def __init__(self, env, segmentation_key, prob_drop):
        self._env = env
        self.segmentation_key = segmentation_key
        self.prob_drop = prob_drop
        assert segmentation_key in env.observation_spec()

    def _transform_observation(self, time_step):
        if self.prob_drop == 0:
            return time_step

        obs = time_step.observation
        mask = obs[self.segmentation_key]
        pixels_to_drop = np.random.binomial(1, self.prob_drop, mask.shape).astype(
            np.uint8
        )
        new_mask = mask * (1 - pixels_to_drop)
        obs[self.segmentation_key] = new_mask.astype(np.uint8)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class SlimMaskWrapper(dm_env.Environment):
    def __init__(self, env, segmentation_key, scale, threshold, sigma):
        self._env = env
        self.segmentation_key = segmentation_key
        self.scale = scale
        self.threshold = threshold
        self.sigma = sigma
        assert segmentation_key in env.observation_spec()

    def _transform_observation(self, time_step):
        obs = time_step.observation
        mask = obs[self.segmentation_key]

        new_mask = rescale(
            mask, (1 / self.scale, 1 / self.scale, 1), anti_aliasing=False
        )
        new_mask = (
            rescale(new_mask, (self.scale, self.scale, 1), anti_aliasing=False) * 255.0
        )
        new_mask = gaussian(new_mask, sigma=self.sigma)
        new_mask = new_mask > self.threshold

        obs[self.segmentation_key] = new_mask.astype(np.uint8)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class StackRGBAndMaskWrapper(dm_env.Environment):
    def __init__(
        self, env, rgb_key="pixels", segmentation_key="segmentation", new_key="pixels"
    ):
        self._env = env
        self.rgb_key = rgb_key
        self.segmentation_key = segmentation_key
        self.new_key = new_key

        assert rgb_key in env.observation_spec()
        assert segmentation_key in env.observation_spec()

        wrapped_obs_spec = env.observation_spec()
        assert (
            wrapped_obs_spec[segmentation_key].shape[:-1]
            == wrapped_obs_spec[rgb_key].shape[:-1]
        )
        frame_shape = wrapped_obs_spec[rgb_key].shape
        # remove batch dim
        if len(frame_shape) == 4:
            frame_shape = frame_shape[1:]
        new_shape = (int(frame_shape[0]), int(frame_shape[1]), 4)
        wrapped_obs_spec.pop(rgb_key)
        wrapped_obs_spec.pop(segmentation_key)
        wrapped_obs_spec[new_key] = specs.BoundedArray(
            shape=new_shape, dtype=np.uint8, minimum=0, maximum=255, name="observation"
        )
        self._obs_spec = wrapped_obs_spec

    def _transform_observation(self, time_step):
        obs = time_step.observation
        rgb = obs.pop(self.rgb_key)
        seg = obs.pop(self.segmentation_key) * 255
        stacked_frames = np.concatenate([rgb, seg], axis=2).astype(np.uint8)
        obs[self.new_key] = stacked_frames
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class Render_Wrapper:
    def __init__(self, render_fn):
        self.render_fn = render_fn

    def render(self, *args, **kwargs):
        return self.render_fn(*args, **kwargs)


class Camera_Render_Wrapper:
    def __init__(self, env_sim, lookat, distance, azimuth, elevation):
        self.lookat = lookat
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self.env_sim = env_sim
        self.camera = None

    def render(self, height, width, *args, **kwargs):
        if (
            self.camera is None
            or self.camera.height != height
            or self.camera.width != width
        ):
            self.camera = engine.MovableCamera(self.env_sim, height, width)
            self.camera.set_pose(
                self.lookat, self.distance, self.azimuth, self.elevation
            )
        return self.camera.render(*args, **kwargs).copy()


class Wrist_Camera_Render_Wrapper:
    def __init__(self, env_sim, lookat, distance, azimuth, elevation):
        self.lookat = lookat
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self.env_sim = env_sim
        self.camera = None

    def render(self, height, width, *args, **kwargs):
        if (
            self.camera is None
            or self.camera.height != height
            or self.camera.width != width
        ):
            self.camera = engine.Camera(
                self.env_sim,
                height,
                width,
                camera_id=self.env_sim.model.camera_name2id("eye_in_hand"),
            )
        return self.camera.render(*args, **kwargs).copy()


class RandomCameraWrapper(dm_env.Environment):
    def __init__(self, env, lookats, distances, azimuths, elevations):
        self._env = env
        self.render_wrappers = [
            Camera_Render_Wrapper(self._env.sim, lookat, distance, azimuth, elevation)
            for lookat, distance, azimuth, elevation in zip(
                lookats, distances, azimuths, elevations
            )
        ]
        self.physics = self.sample_camera()

    def sample_camera(self):
        return random.choice(self.render_wrappers)

    def reset(self):
        self.physics = self.sample_camera()
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


from mujoco_py.generated import const

def get_dino_feature(img,model,h,w):
    img = img.astype(np.float32)/255
    img_tensors = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
    patch_size = 14
    patch_h = h//patch_size
    patch_w = w//patch_size
    with torch.no_grad():
        features_dict = model.forward_features(img_tensors)
    raw_feature_grid = features_dict['x_norm_patchtokens']  # float32 [num_cams, patch_h*patch_w, feature_dim]
    raw_feature_grid = raw_feature_grid.reshape(patch_h, patch_w, -1)  # float32 [num_cams, patch_h, patch_w, feature_dim]
    return raw_feature_grid.cpu().detach().numpy()

class PixelsWrapper(dm_env.Environment):
    """Wraps a control environment and adds a rendered pixel observation."""

    STATE_KEY = 'state'
    def __init__(self, env, pixels_only=True, render_kwargs=None,
                observation_key='pixels',mask=False,dino=False):
        """Initializes a new pixel Wrapper.

        Args:
        env: The environment to wrap.
        pixels_only: If True (default), the original set of 'state' observations
            returned by the wrapped environment will be discarded, and the
            `OrderedDict` of observations will only contain pixels. If False, the
            `OrderedDict` will contain the original observations as well as the
            pixel observations.
        render_kwargs: Optional `dict` containing keyword arguments passed to the
            `mujoco.Physics.render` method.
        observation_key: Optional custom string specifying the pixel observation's
            key in the `OrderedDict` of observations. Defaults to 'pixels'.

        Raises:
        ValueError: If `env`'s observation spec is not compatible with the
            wrapper. Supported formats are a single array, or a dict of arrays.
        ValueError: If `env`'s observation already contains the specified
            `observation_key`.
        """
        if render_kwargs is None:
            render_kwargs = {}
        if 'depth' in render_kwargs and render_kwargs['depth']:
            self.depth = True
            self.depth_render_kwargs = []
            for camera_name in ['gripperPOVpos', 'gripperPOVneg']:
                kwargs = render_kwargs.copy()
                kwargs['camera_name'] = camera_name
                self.depth_render_kwargs.append(kwargs)
            del render_kwargs['depth']
        else:
            self.depth = False
        self.mask = mask
        mask_render_kwargs = render_kwargs.copy()
        if 'depth' in mask_render_kwargs:
            del mask_render_kwargs['depth']
        mask_render_kwargs['segmentation'] = True
        wrapped_observation_spec = env.observation_spec()

        if isinstance(wrapped_observation_spec, specs.Array):
            self._observation_is_dict = False
            invalid_keys = set([self.STATE_KEY])
        elif isinstance(wrapped_observation_spec, collections.abc.MutableMapping):
            self._observation_is_dict = True
            invalid_keys = set(wrapped_observation_spec.keys())
        else:
            raise ValueError('Unsupported observation spec structure.')

        if not pixels_only and observation_key in invalid_keys:
            raise ValueError('Duplicate or reserved observation key {!r}.'
                            .format(observation_key))

        if pixels_only:
            self._observation_spec = collections.OrderedDict()
        elif self._observation_is_dict:
            self._observation_spec = wrapped_observation_spec.copy()
        else:
            self._observation_spec = collections.OrderedDict()
            self._observation_spec[self.STATE_KEY] = wrapped_observation_spec

        # Extend observation spec.
        pixels = env.physics.render(**render_kwargs)
        rgb = pixels
        self.dino = dino
        if dino:
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().cuda()
            pixels = get_dino_feature(pixels,self.dinov2,render_kwargs['height'],render_kwargs['width'])
        
        
        if self.depth:
            pixels = [rgb/255]
            for kwargs in self.depth_render_kwargs:
                image = env.physics.render(**kwargs)
                image = image[1]
                image = np.expand_dims(image,axis=-1)
                pixels.append(image)
            pixels = np.concatenate(pixels, axis=-1).astype(np.float32)
            
        pixels_spec = specs.Array(
            shape=pixels.shape, dtype=pixels.dtype, name=observation_key)
        self._observation_spec[observation_key] = pixels_spec
        # mask = env.physics.render(**mask_render_kwargs)
        # mask = mask[:,:,1]
        # mask_spec = specs.Array(
        #     shape=mask.shape, dtype=mask.dtype, name='mask')
        # self._observation_spec['mask'] = mask_spec
        # masked_rgb_spec = specs.Array(
        #     shape=pixels.shape, dtype=pixels.dtype, name='masked_rgb')
        # self._observation_spec['mask'] = mask_spec
        rgb_spec = specs.Array(
            shape=rgb.shape, dtype=rgb.dtype, name='origin')
        self._observation_spec['origin'] = rgb_spec
        self._env = env
        self._pixels_only = pixels_only
        self._render_kwargs = render_kwargs
        self._mask_render_kwargs = mask_render_kwargs
        self._observation_key = observation_key

    def reset(self):
        time_step = self._env.reset()
        return self._add_pixel_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._add_pixel_observation(time_step)

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._env.action_spec()

    def get_mask(self,segmentation,rgb):
        rgb = rgb.copy()
        assert len(rgb.shape)==3, f'rgb.shape is {rgb.shape}'
        if rgb.shape[2]==3:
            A, B, C = np.array([255,0,0]), np.array([0,255,0]), np.array([0,0,255])
            Background = np.array([0,0,0])
        elif  rgb.shape[2]==1:
            A = np.array(1.0)
            Background = np.array(1.0)
        else:
            raise NotImplementedError
        types = segmentation[:, :, 0]
        ids = segmentation[:, :, 1]
        geoms = types == const.OBJ_GEOM
        geoms_ids = np.unique(ids[geoms])
        excluded_ids = []
        excluded_names = []
        for i in geoms_ids:
            name = self._env.sim.model.body_id2name(
                        self._env.sim.model.geom_bodyid[i]
                    )
            if name is None:
                pass
            elif 'leftpad' in name:
                excluded_ids.append(i)
                excluded_names.append(name)
            elif 'rightpad' in name:
                excluded_ids.append(i)
                excluded_names.append(name)
        mask = np.isin(ids, excluded_ids)
        env_name = 'metaworld_'+self._env.env_name
        mask_id_dict = env_mask_id_dict[env_name]
        rgb[ids==0] = Background
        if rgb.shape[2]==3:
            rgb[ids!=0] = A
            rgb[:] = 0
            rgb[mask] = np.array([255,0,0])
            if 'obj' in mask_id_dict.keys():
                mask_ids = mask_id_dict['obj'][1]
                if isinstance(mask_ids,int):
                    mask_ids = [mask_ids]
                for mask_id in mask_ids:
                    rgb[ids==mask_id] = np.array([0,255,0])
            if 'goal' in mask_id_dict.keys() and self._env.text_plan[self._env.curr_plan_stage][1]=='place':
                mask_ids = mask_id_dict['goal'][1]
                if isinstance(mask_ids,int):
                    mask_ids = [mask_ids]
                for mask_id in mask_ids:
                    rgb[ids==mask_id] = np.array([0,0,255])
            rgb[types==6] = np.array([0,0,255])
        elif rgb.shape[2]==1:
            ids[mask] =0
            if 'obj' in mask_id_dict.keys():
                mask_ids = mask_id_dict['obj'][1]
                if isinstance(mask_ids,int):
                    mask_ids = [mask_ids]
                for mask_id in mask_ids:
                    ids[ids==mask_id] = 0
            if 'goal' in mask_id_dict.keys() and self._env.text_plan[self._env.curr_plan_stage][1]=='place':
                mask_ids = mask_id_dict['goal'][1]
                if isinstance(mask_ids,int):
                    mask_ids = [mask_ids]
                for mask_id in mask_ids:
                    ids[ids==mask_id] = 0
            ids[types==6] = 0
            rgb[ids!=0] = 1.0
        return rgb

    def get_real_depth(self,depth):
        extent = self._env.sim.model.stat.extent
        near = self._env.sim.model.vis.map.znear * extent
        far = self._env.sim.model.vis.map.zfar * extent
        # Convert from [0 1] to depth in meters, see links below:
        # http://stackoverflow.com/a/6657284/1461210
        # https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
        real_depth = near / (1 - depth * (1 - near / far))
        return real_depth

    def _add_pixel_observation(self, time_step):
        if self._pixels_only:
            observation = collections.OrderedDict()
        elif self._observation_is_dict:
            observation = type(time_step.observation)(time_step.observation)
        else:
            observation = collections.OrderedDict()
            observation[self.STATE_KEY] = time_step.observation

        pixels = self._env.physics.render(**self._render_kwargs)
        rgb = pixels
        observation['origin'] = rgb.copy()
        if self.depth:
            if self.mask:
                segmentation = self._env.physics.render(**self._mask_render_kwargs)
                rgb = self.get_mask(segmentation,rgb)
            pixels = [rgb/255]
            for kwargs in self.depth_render_kwargs:
                image = self._env.physics.render(**kwargs)
                image = image[1]
                image = self.get_real_depth(image)
                image = np.expand_dims(image,axis=-1)
                if self.mask:
                    kwargs = kwargs.copy()
                    kwargs['segmentation'] = True
                    kwargs['depth'] = False
                    segmentation = self._env.physics.render(**kwargs)
                    image = self.get_mask(segmentation,image)
                pixels.append(image)
            pixels = np.concatenate(pixels, axis=-1).astype(np.float32)
        if self.dino:
            pixels = get_dino_feature(pixels,self.dinov2,self._render_kwargs['height'],self._render_kwargs['width'])
        observation[self._observation_key] = pixels
        
        return time_step._replace(observation=observation)

    def __getattr__(self, name):
        return getattr(self._env, name)

class CropWrapper(dm_env.Environment):
    def __init__(self, env, keys_to_crop, top_left, bottom_right):
        self._env = env
        self.keys_to_crop = keys_to_crop
        self.top_left = top_left
        self.bottom_right = bottom_right

        wrapped_obs_spec = env.observation_spec()
        for key in keys_to_crop:
            old_spec = wrapped_obs_spec[key]
            frame_shape = list(old_spec.shape)
            frame_shape[0] = bottom_right[0] - top_left[0]
            frame_shape[1] = bottom_right[1] - top_left[1]
            new_spec = specs.BoundedArray(
                shape=tuple(frame_shape),
                dtype=old_spec.dtype,
                minimum=0,
                maximum=255,
                name=old_spec.name,
            )
            wrapped_obs_spec[key] = new_spec

        self._obs_spec = wrapped_obs_spec

    def _transform_observation(self, time_step):
        obs = time_step.observation
        for key in self.keys_to_crop:
            frame = obs[key][
                self.top_left[0] : self.bottom_right[0],
                self.top_left[1] : self.bottom_right[1],
            ]
            obs[key] = frame
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def get_env_observation_spec(env):
    return specs.BoundedArray(
        shape=env.observation_space.shape,
        dtype=env.observation_space.dtype,
        minimum=env.observation_space.low,
        maximum=env.observation_space.high,
        name="observation",
    )


def get_env_action_spec(env):
    return specs.BoundedArray(
        shape=env.action_space.shape,
        dtype=env.action_space.dtype,
        minimum=env.action_space.low,
        maximum=env.action_space.high,
        name="action",
    )

