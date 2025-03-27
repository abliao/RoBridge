# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import pickle
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from gea import utils
from gea.models.actor import Actor
from gea.models.critic import Critic
from gea.models.encoder import DrQV2Encoder, PoolEncoder, ModalEncoder
from gea.models.random_shifts_aug import RandomShiftsAug

class DrQV2Agent:
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
        only_act=False
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
        self.only_act = only_act
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
        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)

        if not only_act:
            self.critic = Critic(
                self.encoder.repr_dim+self.modal_encoder.repr_dim, action_shape, feature_dim, hidden_dim
            ).to(device)
            self.critic_target = Critic(
                self.encoder.repr_dim+self.modal_encoder.repr_dim, action_shape, feature_dim, hidden_dim
            ).to(device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
            self.critic_target.train()
        
        

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

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.modal_encoder.train(training)
        self.actor.train(training)
        if not self.only_act:
            self.critic.train(training)

    def act(self, obs, step, eval_mode):
        modal_emb = self.modal_encoder(obs, device=self.device)
        observation = obs
        obs = obs["pixels"]
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        obs = torch.cat([obs, modal_emb], dim=-1)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
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

    def update_actor(self, encoded_obs, step):
        metrics = {}

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(encoded_obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        q1, q2 = self.critic(encoded_obs, action)
        q = torch.min(q1, q2)

        actor_loss = -q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

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
        modal_emb = self.modal_encoder(obs, device=self.device,mask=random.random()>0.3)
        # augment
        if self.aug:
            rgb_obs = self.aug(rgb_obs.float())
            next_rgb_obs = self.aug(next_rgb_obs.float())

            # bs, _, h, w = rgb_obs.shape
            # random_mask = self.generate_random_mask((bs,h,w),0.1).to(rgb_obs.device).unsqueeze(1)
            
            # target_values = torch.tensor([255, 0, 0, 255, 0, 0], dtype=rgb_obs.dtype, device=rgb_obs.device)
            # expanded_mask = random_mask.expand(-1, len(target_values), -1, -1)
            # rgb_obs = torch.where(expanded_mask == 1, target_values.view(1, -1, 1, 1), rgb_obs)
            # next_rgb_obs = torch.where(expanded_mask == 1, target_values.view(1, -1, 1, 1), next_rgb_obs)

        # encode
        encoded_obs = self.encoder(rgb_obs)
        with torch.no_grad():
            encoded_next_obs = self.encoder(next_rgb_obs)
        encoded_obs = torch.cat([encoded_obs, modal_emb], dim=-1)
        encoded_next_obs = torch.cat([encoded_next_obs, modal_emb], dim=-1)
        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()
        # update critic and decoder
        metrics.update(
            self.update_critic(
                encoded_obs, action, reward, discount, encoded_next_obs, step
            )
        )

        # update actor
        metrics.update(self.update_actor(encoded_obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )
        return metrics

    def get_frames_to_record(self, obs):
        rgb_obs = obs["pixels"]
        frame = rgb_obs[-3:].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
        return {"rgb": frame}

    def load_pretrained_weights(self, pretrain_path, just_encoder_decoders):
        if just_encoder_decoders:
            print("Loading pretrained encoder and decoders")
        else:
            print("Loading entire agent")

        import sys
        sys.modules['planseqlearn'] = sys.modules['gea']
        payload = torch.load(pretrain_path, map_location="cpu")
        pretrained_agent = payload["agent"]

        self.encoder.load_state_dict(pretrained_agent.encoder.state_dict())
        self.modal_encoder.load_state_dict(pretrained_agent.modal_encoder.state_dict())
        self.actor.load_state_dict(pretrained_agent.actor.state_dict())
        if not just_encoder_decoders:
            self.critic.load_state_dict(pretrained_agent.critic.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
