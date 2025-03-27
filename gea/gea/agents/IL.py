# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import numpy as np
import time
import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur, RandomErasing, RandomAffine
import torch.nn as nn
from gea import utils
# from gea.models.actor import Actor
from gea.models.critic import Critic
from gea.models.encoder import DrQV2Encoder, PoolEncoder, ModalEncoder
from gea.models.random_shifts_aug import RandomShiftsAug
# from gea.models.robotic_transformer import Transformer

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        return mu


class DepthObsHandler():
    def __init__(self, cfg):
        self.cfg = cfg

    def get_visual_obs(self, envs, prev_vis_obs, **kwargs):
        depths, seg, rgb = envs.get_depth_observations()
        assert seg is None and rgb is None
        return depths

    def _depth_warping(self, depths, std=0.5, prob=0.8, device=None):
        n, _, h, w = depths.shape

        # Generate Gaussian shifts
        gaussian_shifts = torch.normal(mean=0, std=std, size=(n, h, w, 2), device=device).float()
        apply_shifts = torch.rand(n, device=device) < prob
        gaussian_shifts[~apply_shifts] = 0.0

        # Create grid for the original coordinates
        xx = torch.linspace(0, w - 1, w, device=device)
        yy = torch.linspace(0, h - 1, h, device=device)
        xx = xx.unsqueeze(0).repeat(h, 1)
        yy = yy.unsqueeze(1).repeat(1, w)
        grid = torch.stack((xx, yy), 2).unsqueeze(0)  # Add batch dimension

        # Apply Gaussian shifts to the grid
        grid = grid + gaussian_shifts

        # Normalize grid values to the range [-1, 1] for grid_sample
        grid[..., 0] = (grid[..., 0] / (w - 1)) * 2 - 1
        grid[..., 1] = (grid[..., 1] / (h - 1)) * 2 - 1

        # Perform the remapping using grid_sample
        depth_interp = F.grid_sample(depths, grid, mode='bilinear', padding_mode='border', align_corners=True)

        # Remove the batch and channel dimensions
        depth_interp = depth_interp.squeeze(0).squeeze(0)

        return depth_interp
    
    def _generate_mask(self, n, h, w, device):
        k_lower = self.cfg.augmentation.holes.kernel_size_lower
        k_upper = self.cfg.augmentation.holes.kernel_size_upper
        s_lower = self.cfg.augmentation.holes.sigma_lower
        s_upper = self.cfg.augmentation.holes.sigma_upper
        thresh_lower = self.cfg.augmentation.holes.thresh_lower
        thresh_upper = self.cfg.augmentation.holes.thresh_upper

        # generate random noise
        noise = torch.rand(n, 1, h, w, device=device)

        # apply gaussian blur
        k = random.choice(list(range(k_lower, k_upper+1, 2)))
        noise = GaussianBlur(kernel_size=k, sigma=(s_lower, s_upper))(noise)

        # normalize noise
        noise = (noise - noise.min()) / (noise.max() - noise.min())

        # apply thresholding
        thresh = torch.rand(n, 1, 1, 1, device=device) * (thresh_upper - thresh_lower) + thresh_lower
        mask = (noise > thresh)

        return mask

    def apply_noise(self, visual_obs, device=None, **kwargs):
        if device is None:
            device = visual_obs.device

        visual_obs_shape = visual_obs.shape
        h, w = visual_obs_shape[-2:]
        obs = visual_obs.reshape(-1, 1, h, w).clone().to(device)
        n = obs.shape[0]

        # apply depth warping
        if self.cfg.augmentation.depth_warping.enabled:
            obs = self._depth_warping(
                obs, 
                std=self.cfg.augmentation.depth_warping.std, 
                prob=self.cfg.augmentation.depth_warping.prob,
                device=device
            )

        # apply blurring
        if self.cfg.augmentation.gaussian_blur.enabled:
            transform = GaussianBlur(
                kernel_size=self.cfg.augmentation.gaussian_blur.kernel_size, 
                sigma=(self.cfg.augmentation.gaussian_blur.sigma_lower, self.cfg.augmentation.gaussian_blur.sigma_upper)
            )
            obs = transform(obs)

        # apply non-linear scaling
        if self.cfg.augmentation.scale.enabled:
            intensity = torch.rand(n, 1, 1, 1, device=device) * self.cfg.augmentation.scale.intensity
            corners = (2 * torch.rand(n, 1, 2, 2, device=device) - 1) * intensity + 1
            scale_map = F.interpolate(corners, size=(h, w), mode='bicubic', align_corners=False).reshape(n, 1, h, w)
            apply_scaling = torch.rand(n, device=device) < self.cfg.augmentation.scale.prob
            scale_map[~apply_scaling] = 1.0
            obs = obs * scale_map
            obs = torch.clamp(obs, 0, 1)

        # apply holes
        if self.cfg.augmentation.holes.enabled:
            mask = self._generate_mask(n, h, w, device)
            prob = self.cfg.augmentation.holes.prob
            keep_mask = torch.rand(n, device=device) < prob
            mask[~keep_mask, :] = 0
            obs[mask] = self.cfg.augmentation.holes.fill_value

        ret = obs.reshape(*visual_obs_shape).to(visual_obs.device)

        return ret
def DepthMapPseudoColorize(depth_map, output_path, max_depth=None, min_depth=None):
    # Load 16 units depth_map(element ranges from 0~65535),
    # Convert it to 8 units (element ranges from 0~255) and pseudo colorize
    if isinstance(depth_map, np.ndarray):
        uint16_img =  depth_map
    # Indicate the argument load mode -1, otherwise loading default 8 units
    if isinstance(depth_map, str):
        uint16_img = cv2.imread(depth_map, -1)
    if None == max_depth:
        max_depth = uint16_img.max()
    if None == min_depth:
        min_depth = uint16_img.min()

    # uint16_img -= min_depth
    # uint16_img = uint16_img / (max_depth - min_depth)
    # uint16_img *= 255

    # cv2.COLORMAP_JET, blue represents a higher depth value, and red represents a lower depth value
    im_color=cv2.applyColorMap(cv2.convertScaleAbs(uint16_img, alpha=1), cv2.COLORMAP_JET)
    # im_color=cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
    # convert to mat png
    im=Image.fromarray(im_color)
    im.save(output_path)

class ILAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
        use_pool_encoder,
        pool_encoder_latent_dim,
        added_encode_modal,
        data_aug,
        depth,
        dino,
        only_act=False,
        use_depth_handler=False,
        depth_handler=None
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.depth = depth
        self.dino = dino
        self.use_depth_handler = use_depth_handler
        if use_depth_handler:
            self.depth_handler = DepthObsHandler(depth_handler)

        # models
        if use_pool_encoder:
            self.encoder = PoolEncoder(obs_shape, repr_dim=pool_encoder_latent_dim).to(
                device
            )
        else:
            self.encoder = DrQV2Encoder(obs_shape,depth=depth,dino=dino).to(device)
        self.modal_encoder = ModalEncoder(added_encode_modal).to(device)

        self.actor = Actor(
            self.encoder.repr_dim+self.modal_encoder.repr_dim, action_shape, feature_dim, hidden_dim
        ).to(device)
        # self.actor = Transformer(
        #     dim = 512,
        #     dim_head = 8,
        #     heads = 32,
        #     depth = 6,
        #     ff_dropout=0.1,
        #     attn_dropout=0.1,
        # ).to(device)

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # data augmentation
        if data_aug and (not self.dino):
            self.aug = RandomShiftsAug(pad=4)
        else:
            self.aug = None
        self.train()
        encoder_para = sum(p.numel() for p in self.encoder.parameters())
        actor_para = sum(p.numel() for p in self.actor.parameters())
        print(f'{encoder_para} parameters in encoder')
        print(f'{actor_para} parameters in actor')
        print(f'{encoder_para + actor_para} parameters in agent')

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.modal_encoder.train(training)
        self.actor.train(training)

    def act(self, obs, step, eval_mode):
        modal_emb = self.modal_encoder(obs, device=self.device)
        observation = obs
        obs = obs["pixels"]
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        obs = torch.cat([obs, modal_emb], dim=-1)
        stddev = utils.schedule(self.stddev_schedule, step)
        action = self.actor(obs, stddev)
        # if eval_mode:
        #     action = dist.mean
        # else:
        #     action = dist.sample(clip=None)
        #     if step < self.num_expl_steps:
        #         action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(
        self, encoded_obs, action, reward, discount, encoded_next_obs, step
    ):
        metrics = {}

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(encoded_next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_q1, target_q2 = self.critic_target(encoded_next_obs, next_action)
            target_v = torch.min(target_q1, target_q2)
            target_q = reward + (discount * target_v)

        q1, q2 = self.critic(encoded_obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        if self.use_tb:
            metrics["critic_target_q"] = target_q.mean().item()
            metrics["critic_q1"] = q1.mean().item()
            metrics["critic_q2"] = q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        return metrics

    def update_actor(self, encoded_obs, step, gt):
        metrics = {}

        stddev = utils.schedule(self.stddev_schedule, step)
        action = self.actor(encoded_obs, stddev)
        actor_loss = oss = F.mse_loss(action, gt)

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()

        return metrics

    def generate_random_mask(self, shape, p):
        random_matrix = torch.rand(*shape)
        mask = (random_matrix < p).int()
        return mask

    def update(self, replay_iter, step):
        metrics = {}

        if step % self.update_every_steps != 0:
            return metrics
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = batch
        observation = obs
        next_observation = next_obs
        rgb_obs = obs["pixels"]
        next_rgb_obs = next_obs["pixels"]
        rgb_obs, action, reward, discount, next_rgb_obs = utils.to_torch(
            (rgb_obs, action, reward, discount, next_rgb_obs), self.device
        )
        modal_emb = self.modal_encoder(obs, device=self.device,mask=random.random()>0.1)
        # augment
        if self.use_depth_handler:
            obs_mask = rgb_obs==1.
            rgb_obs[:,[3,4,8,9]] = self.depth_handler.apply_noise(rgb_obs[:,[3,4,8,9]], device=self.device)
            rgb_obs[obs_mask] = 1.

        if self.aug:
            rgb_obs = self.aug(rgb_obs.float())
            next_rgb_obs = self.aug(next_rgb_obs.float())  

            bs, _, h, w = rgb_obs.shape
            random_mask = self.generate_random_mask((bs,h,w),0.01).to(rgb_obs.device).unsqueeze(1)
            
            target_values = torch.tensor([1., 0, 0, 1., 1. ]*(rgb_obs.shape[1]//5), dtype=rgb_obs.dtype, device=rgb_obs.device)
            expanded_mask = random_mask.expand(-1, len(target_values), -1, -1)
            rgb_obs = torch.where(expanded_mask == 1, target_values.view(1, -1, 1, 1), rgb_obs)
            next_rgb_obs = torch.where(expanded_mask == 1, target_values.view(1, -1, 1, 1), next_rgb_obs)

        # encode
        encoded_obs = self.encoder(rgb_obs)
        # with torch.no_grad():
        #     encoded_next_obs = self.encoder(next_rgb_obs)
        encoded_obs = torch.cat([encoded_obs, modal_emb], dim=-1)
        # encoded_next_obs = torch.cat([encoded_next_obs, modal_emb], dim=-1)
        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update actor
        metrics.update(self.update_actor(encoded_obs.detach(), step, action))

        return metrics

    def get_frames_to_record(self, obs):
        rgb_obs = obs["pixels"]
        if rgb_obs.max()<=1:
            rgb_obs = rgb_obs*255
        frame = rgb_obs[:3].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
        # frame = rgb_obs[3].clip(0, 255).astype(np.uint8)
        return {"rgb": frame}

    def load_pretrained_weights(self, pretrain_path, just_encoder_decoders):
        if just_encoder_decoders:
            print("Loading pretrained encoder and decoders")
        else:
            print("Loading entire agent")

        payload = torch.load(pretrain_path, map_location="cpu")
        pretrained_agent = payload["agent"]

        self.encoder.load_state_dict(pretrained_agent.encoder.state_dict())
        self.modal_encoder.load_state_dict(pretrained_agent.modal_encoder.state_dict())
        if not just_encoder_decoders:
            self.actor.load_state_dict(pretrained_agent.actor.state_dict())
            # self.critic.load_state_dict(pretrained_agent.critic.state_dict())
            # self.critic_target.load_state_dict(self.critic.state_dict())
