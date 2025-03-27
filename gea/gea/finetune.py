# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import pickle
import datetime
import os
import subprocess
import traceback
import torch
import dateutil.tz
import sys
import hydra

def load_all_episodes(data_folder='data'):

    if not os.path.exists(data_folder):
        raise ValueError(f"can't find folders: {data_folder}")
    
    episode_folders = sorted([f for f in os.listdir(data_folder) if f.startswith('episode_')])
    
    all_episodes = []
    for folder in episode_folders:
        episode_num = int(folder.split('_')[1])
        episode_path = os.path.join(data_folder, folder)
        
        files = sorted([f for f in os.listdir(episode_path) if f.endswith('.pkl')])
        
        for file in files:
            with open(os.path.join(episode_path, file), 'rb') as f:
                data = pickle.load(f)
                all_episodes.append(data)
        
    return all_episodes

@hydra.main(config_path="cfgs", config_name="train_config")
def main(cfg):
    import os
    from collections import OrderedDict
    from pathlib import Path

    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

    import hydra
    import numpy as np
    import torch
    import wandb
    from dm_env import specs
    from omegaconf import OmegaConf
    import json
    from gea import utils
    from gea.environments.metaworld_dm_env import make_metaworld
    from gea.logger import Logger, compute_path_info
    from gea.replay_buffer import ReplayBufferStorage, make_replay_loader, RealBuffer
    from gea.video import TrainVideoRecorder, VideoRecorder
    from rlkit.core import logger as rlkit_logger
    from rlkit.core.eval_util import create_stats_ordered_dict

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    def make_agent(obs_spec, action_spec, agent_cfg, pretrain_cfg):
        assert (
            "pixels" in obs_spec
        ), "Observation spec passed to make_agent must contain a observation named 'pixels'"
        try:
            if agent_cfg.obs_shape is not None:
                pass
        except:
                agent_cfg.obs_shape = obs_spec["pixels"].shape
        
        agent_cfg.action_shape = action_spec.shape
        agent = hydra.utils.instantiate(agent_cfg)
        if "path" in pretrain_cfg:
            agent.load_pretrained_weights(
                pretrain_cfg.path, pretrain_cfg.just_encoder_decoders
            )
        return agent

    def make_env(cfg, is_eval):
        if cfg.task_name.split("_", 1)[0] == "metaworld":
            env = make_metaworld(
                name=cfg.task_name.split("_", 1)[1],
                frame_stack=cfg.frame_stack,
                action_repeat=cfg.action_repeat,
                discount=cfg.discount,
                seed=cfg.seed,
                camera_name=cfg.camera_name,
                depth=cfg.depth,
                mask=cfg.mask,
                dino=cfg.dino,
                psl=cfg.psl,
                text_plan=cfg.text_plan,
                use_vision_pose_estimation=cfg.use_vision_pose_estimation,
                use_mp=cfg.use_mp,
                is_eval=is_eval
            )
        elif cfg.task_name.split("_", 1)[0] == "robosuite":
            env = make_robosuite(
                name=cfg.task_name.split("_", 1)[1],
                frame_stack=cfg.frame_stack,
                action_repeat=cfg.action_repeat,
                discount=cfg.discount,
                camera_name=cfg.camera_name,
                psl=cfg.psl,
                path_length=cfg.path_length,
                vertical_displacement=cfg.vertical_displacement,
                estimate_orientation=cfg.estimate_orientation,
                valid_obj_names=cfg.valid_obj_names,
                use_proprio=cfg.use_proprio,
                text_plan=cfg.text_plan,
                use_vision_pose_estimation=cfg.use_vision_pose_estimation,
                use_mp=cfg.use_mp,
            )
        elif cfg.task_name.split("_", 1)[0] == "kitchen":
            env = make_kitchen(
                name=cfg.task_name.split("_", 1)[1],
                frame_stack=cfg.frame_stack,
                action_repeat=cfg.action_repeat,
                discount=cfg.discount,
                seed=cfg.seed,
                camera_name=cfg.camera_name,
                path_length=cfg.path_length,
                psl=cfg.psl,
                text_plan=cfg.text_plan,
                use_mp=cfg.use_mp,
            )
        elif cfg.task_name.split("_", 1)[0] == "mopa":
            env = make_mopa(
                name=cfg.task_name.split("_", 1)[1],
                frame_stack=cfg.frame_stack,
                action_repeat=cfg.action_repeat,
                seed=cfg.seed,
                horizon=cfg.path_length,
                psl=cfg.psl,
                text_plan=cfg.text_plan,
                use_vision_pose_estimation=cfg.use_vision_pose_estimation,
                use_mp=cfg.use_mp,
            )
        return env

    class Workspace:
        def __init__(self, cfg):
            self.work_dir = Path.cwd()
            print(f"workspace: {self.work_dir}")
            rlkit_logger.set_snapshot_dir(self.work_dir)
            rlkit_logger.use_wandb = False
            # create progress csv file
            rlkit_logger.add_tabular_output(os.path.join(self.work_dir, "progress.csv"))

            self.cfg = cfg
            self.expert_schedule = cfg.expert_schedule
            utils.set_seed_everywhere(cfg.seed)
            self.device = torch.device(cfg.device)

            self.logger = Logger(
                self.work_dir, use_tb=self.cfg.use_tb, use_wandb=self.cfg.use_wandb
            )

            self.train_env = make_env(self.cfg, is_eval=False)
            self.eval_env = make_env(self.cfg, is_eval=False)
            if self.cfg.has_success_metric:
                reward_spec = OrderedDict(
                    [
                        ("reward", specs.Array((1,), np.float32, "reward")),
                        ("success", specs.Array((1,), np.int16, "reward")),
                    ]
                )
            else:
                reward_spec = specs.Array((1,), np.float32, "reward")

            discount_spec = specs.Array((1,), np.float32, "discount")
            from gym import spaces
            observation_space = self.train_env.observation_spec()
            observation_space_new = {}
            for key,value in observation_space.items():
                if key=='origin':
                    continue
                observation_space_new[key] = value
            data_specs = {
                "observation": observation_space_new,
                "action": self.train_env.action_spec(),
                "reward": reward_spec,
                "discount": discount_spec,
            }
            self.replay_storage = ReplayBufferStorage(
                data_specs, self.work_dir / "buffer"
            )

            self.replay_loader = make_replay_loader(
                self.work_dir / "buffer",
                self.cfg.replay_buffer_size,
                self.cfg.batch_size,
                self.cfg.replay_buffer_num_workers,
                self.cfg.save_buffer_snapshot,
                self.cfg.nstep,
                self.cfg.discount,
                self.cfg.has_success_metric,
            )
            self._replay_iter = None
            self.cfg.agent.depth = self.cfg.depth
            self.cfg.agent.dino = self.cfg.dino
            self.agent = make_agent(
                self.train_env.observation_spec(),
                self.train_env.action_spec(),
                self.cfg.agent,
                self.cfg.pretrain,
            )

            self.video_recorder = VideoRecorder(
                self.work_dir if self.cfg.save_video else None,
                metaworld_camera_name=self.cfg.camera_name
                if cfg.task_name.split("_", 1)[0] == "metaworld"
                else None,
                use_wandb=self.cfg.use_wandb,
            )
            self.train_video_recorder = TrainVideoRecorder(
                self.work_dir if self.cfg.save_train_video else None
            )

            cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
            with open(os.path.join(self.work_dir, "variant.json"), "w") as f:
                json.dump(cfg_dict, f, indent=4)
            if self.cfg.use_wandb:
                wandb_worked = False
                while not wandb_worked:
                    try:
                        self.wandb_run = wandb.init(
                            project=self.cfg.wandb.project_name,
                            config=cfg_dict,
                            name=self.cfg.wandb.run_name,
                        )
                        wandb_worked = True
                    except:
                        print(traceback.format_exc())
                        print("Wandb failed, retrying...")
                        time.sleep(5)

            self.timer = utils.Timer()
            self._global_step = 0
            self._global_episode = 0
            self._max_success_rate = 0

        @property
        def global_step(self):
            return self._global_step

        @property
        def global_episode(self):
            return self._global_episode

        @property
        def global_frame(self):
            return self.global_step * self.cfg.action_repeat

        @property
        def replay_iter(self):
            if self._replay_iter is None:
                self._replay_iter = iter(self.replay_loader)
            return self._replay_iter

        def train(self):
            
            datas = load_all_episodes(data_folder='data')
            print('total data:',len(datas))

            time_steps = []
            for data in datas:
                obs = {}
                obs['pixels'] = data['obs_dict']['pixels']
                obs['action'] = data['obs_dict']['action']
                obs['orientation'] = data['obs_dict']['orientation']
                obs['obj_state'] = data['obs_dict']['obj_state']
                obs['gripper_state'] = data['obs_dict']['gripper_state']
                obs['diff_state'] = data['obs_dict']['diff_state']
                action = np.array(data['actions']).astype(np.float32)
                action[:3]*=100
                reward = data['actions']
                discount = data['actions']
                next_obs = data['actions']
                time_steps.append([obs,action,reward,discount,next_obs])
            
            dataset = RealBuffer(time_steps)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=4,
                num_workers=4,
                pin_memory=True
            )
            replay_iter = iter(dataloader)
            for i in range(1000):
                metrics = self.agent.update(replay_iter, i)
                print('metrics:',metrics['actor_loss'])
            self.save_snapshot('final.pt')

            
        def save_snapshot(self, file_name="snapshot.pt"):
            snapshot = self.work_dir / file_name
            keys_to_save = ["agent", "timer", "_global_step", "_global_episode"]
            payload = {k: self.__dict__[k] for k in keys_to_save}
            cfg_keys_to_save = [
                "has_success_metric",
                "task_name",
                "frame_stack",
                "action_repeat",
                "discount",
            ]
            if "camera_name" in self.cfg:
                cfg_keys_to_save.append("camera_name")
            payload.update({k: self.cfg[k] for k in cfg_keys_to_save})

            with snapshot.open("wb") as f:
                torch.save(payload, f)

        def load_snapshot(self):
            snapshot = self.work_dir / "snapshot.pt"
            with snapshot.open("rb") as f:
                payload = torch.load(f)
            for k, v in payload.items():
                self.__dict__[k] = v

    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
