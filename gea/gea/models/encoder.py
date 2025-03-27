import random

from torch import nn
import torch.nn.functional as F
import torch
from gea import utils

class ModalEncoder(nn.Module):
    def __init__(self, added_encode_modal):
        super().__init__()
        self.repr_dim = 0
        self.encode_modal = added_encode_modal.encode_modal
        if "task" in added_encode_modal.encode_modal:
            self._task_emb = nn.Embedding(added_encode_modal.num_tasks, added_encode_modal.task_dim, max_norm=1)
            self.repr_dim += added_encode_modal.task_dim
        if "action" in added_encode_modal.encode_modal:
            self._action_emb = nn.Embedding(added_encode_modal.num_actions, added_encode_modal.action_dim, max_norm=1)
            self.repr_dim += added_encode_modal.action_dim
        if "state" in added_encode_modal.encode_modal:
            self._state_emb = nn.Linear(added_encode_modal.num_states, added_encode_modal.state_dim)
            self.repr_dim += added_encode_modal.state_dim
        if "orientation" in added_encode_modal.encode_modal:
            self._orientation_emb = nn.Linear(added_encode_modal.num_orientation, added_encode_modal.orientation_dim)
            self.repr_dim += added_encode_modal.state_dim
            
        self.apply(utils.weight_init)

    def forward(self, obs, device, mask=False):
        embs = []
        if "task" in self.encode_modal:
            task = torch.tensor(obs['task'], device=device)
            task_emb = self._task_emb(task.long())
            if len(task_emb.shape)==3:
                task_emb = task_emb.squeeze(1)
            embs.append(task_emb)
        if "action" in self.encode_modal:
            action = torch.tensor(obs['action'], device=device)
            action_emb = self._action_emb(action.long())
            if len(action_emb.shape)==3:
                action_emb = action_emb.squeeze(1)
            embs.append(action_emb)
        if "state" in self.encode_modal:
            state = torch.tensor(obs['state'], device=device)
            if len(state.shape)==1:
                state = state.unsqueeze(0)
            state_emb = self._state_emb(state)
            state_emb = F.normalize(state_emb, p=2, dim=-1)
            if mask:
                state_emb*=0
            embs.append(state_emb)
        if "orientation" in self.encode_modal:
            orientation = torch.tensor(obs['orientation'], device=device)
            if len(orientation.shape)==1:
                orientation = orientation.unsqueeze(0)
            orientation_emb = self._orientation_emb(orientation)
            orientation_emb = F.normalize(orientation_emb, p=2, dim=-1)
            if mask:
                orientation_emb*=0
            embs.append(orientation_emb)
        return torch.cat(embs, dim=-1)

class DrQV2Encoder(nn.Module):
    def __init__(self, obs_shape,depth=False,dino=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 0
        self.depth = depth
        self.dino = dino
        if dino:
            self.repr_dim = 12*12*64
            self.projection = nn.Sequential(
                nn.Linear(obs_shape[0], 64),
            )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1),
                nn.ReLU(),
            )
            self.repr_dim += 32 * 35 * 35
        # self.repr_dim += 32 * 33 * 33
        self.apply(utils.weight_init)

    def forward(self, obs):
        if self.dino:
            obs = obs.permute(0,2,3,1)
            h = self.projection(obs) 
            h = h.view(h.shape[0],-1)
            return h
        else:
            if self.depth:
                pass
            else:
                obs = obs / 255.0 - 0.5
            h = self.convnet(obs)
            h = h.view(h.shape[0], -1)
            return h


class PoolEncoder(nn.Module):
    def __init__(self, obs_shape, repr_dim=None):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),  # 41 x 41
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),  # 39 x 39
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),  # 37 x 37
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),  # 35 x 35
            nn.ReLU(),
            nn.AvgPool2d(4, stride=4),  # 32 * 8 * 8
        )

        if repr_dim is None:
            self.repr_dim = 32 * 8 * 8
            self.projection = nn.Identity()
        else:
            self.repr_dim = repr_dim
            self.projection = nn.Sequential(
                nn.Linear(32 * 8 * 8, repr_dim),
            )
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.projection(h)
        return h
