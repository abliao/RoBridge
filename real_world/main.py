
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torchvision.ops import box_convert
import sys
import os
import re
import json
import pickle
import time
from PIL import Image
from multiprocessing import Process, Event
from real_world.GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate

from real_world.Gen3 import utilities
from real_world.hcp.planner import Planner
from real_world.Gen3.robot import Robot
from real_world.env import Env
from real_world.utils import *
import argparse
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import \
    BaseCyclicClient

import numpy as np
import cv2

class RoBridge:
    def __init__(self,env,planner, cfg):
        self.env=env
        self.planner = planner
        self.cfg = cfg
        self.GroundingDINO_model = load_model(cfg.GroundingDINO.cfg,cfg.GroundingDINO.checkpoint)
        def make_agent(agent_cfg, pretrain_cfg):
            agent_cfg.obs_shape = (10, 168, 168)
            agent_cfg.action_shape = (4, )
            agent = hydra.utils.instantiate(agent_cfg)
            if "path" in pretrain_cfg:
                agent.load_pretrained_weights(
                    pretrain_cfg.path, pretrain_cfg.just_encoder_decoders
                )
            return agent
        self.agent = make_agent(
                    cfg.agent,
                    cfg.pretrain,
                )
        self.agent.train(False)

        self.reset()

    def solve(self,task):  
        self.env.camera_capture_rgb()
        actions = self.planner.task_planning(f'imgs/{self.env.tracker.name}.jpg',task)
        print('task planning:\n',actions)
        last_action=None
        for action in actions:
            self.do_action(action,last_action)
            last_action = action
        cv2.destroyAllWindows()

    def do_action(self, action, last_action=None):
        segment_length = self.cfg.segment_length
        for tracker in self.env.trackers:
            tracker.reset()
        self.env.step()
        print('action', action)

        self.env.camera_capture_rgb()
        TEXT_PROMPT = f"{self.cfg.GroundingDINO.default_prompt}. {action['obj']}. {action['target']}."
        
        # first view
        input_path=f'imgs/{self.env.tracker2.name}.jpg'
        output_path='imgs/SoM_first.jpg'
        masks_first,keypoints_first = get_som(TEXT_PROMPT,input_path,output_path,self.GroundingDINO_model,sam=self.env.tracker2.model.samcontroler.sam_controler)
        IOR_first = self.planner.get_IOR(output_path,action['task'])
        if 'Gripper' in IOR_first and IOR_first['Gripper'] is not None:
            obj_mask = masks_first[int(IOR_first['Gripper']['number'])]
            self.env.tracker2.final_mask[obj_mask] = 1
        if 'Object' in IOR_first and IOR_first['Object'] is not None:
            obj_mask = masks_first[int(IOR_first['Object']['number'])]
            self.env.tracker2.final_mask[obj_mask] = 2
        if 'Target' in IOR_first and IOR_first['Target'] is not None:
            obj_mask = masks_first[int(IOR_first['Target']['number'])]
            self.env.tracker2.final_mask[obj_mask] = 2
        print('IOR_first',IOR_first)

        # third view
        input_path=f'imgs/{self.env.tracker.name}.jpg'
        output_path='imgs/SoM_third.jpg'
        masks_third,keypoints_third = get_som(TEXT_PROMPT,input_path,output_path,self.GroundingDINO_model,sam=self.env.tracker.model.samcontroler.sam_controler)
        IOR_third = self.planner.get_IOR(output_path,action['task'])
        if 'Gripper' in IOR_third and IOR_third['Gripper'] is not None:
            obj_mask = masks_third[int(IOR_third['Gripper']['number'])]
            self.env.tracker.final_mask[obj_mask] = 1
        if 'Object' in IOR_third and IOR_third['Object'] is not None:
            obj_mask = masks_third[int(IOR_third['Object']['number'])]
            self.env.tracker.final_mask[obj_mask] = 2
        if 'Target' in IOR_third and IOR_third['Target'] is not None:
            obj_mask = masks_third[int(IOR_third['Target']['number'])]
            self.env.tracker.final_mask[obj_mask] = 2
        print('IOR_third',IOR_third)
        
        for tracker in self.env.trackers:
            tracker.control_signal(ord('t'))

        if 'reach' in IOR_first['Action']:
            obj_num = int(IOR_first['Object']['number'])
            keypoint = keypoints_first[obj_num]
            obj_state = self.env.pixel2loc(*keypoint,self.env.tracker2)
            diff = obj_state-self.env.gripper_state
            divided_actions = divide_into_segments(diff,segment_length)
            new_column = np.zeros((len(divided_actions), 1))
            divided_actions = np.hstack((divided_actions, new_column))
            last_actions = np.array([0,0,0,1])
            divided_actions = np.vstack((divided_actions,last_actions))
            last_actions = np.array([[0,0,0.2,1]])
            divided_actions = np.vstack((divided_actions,last_actions))
            return

        obs = self.env.get_obs()                 
        next_check = self.cfg.check_period

        gripper_open = self.env.robot.gripper
        for i in range(self.cfg.max_steps):
            key = cv2.waitKey(1) & 0xFF
            obs_dict = {'pixels':obs,
                                'action':np.array([action_name2id[IOR_first['Action']]],np.int32),
                                'orientation':IOR_first['Constraint'],
                                'gripper_state': self.env.gripper_state.copy(),
                                'obj_state': self.env.obj_state.copy(),
                                'diff_state': self.env.obj_state-self.env.gripper_state,
                                }
            with torch.no_grad():
                value = self.agent.act(obs_dict, 1, True)
            value = np.clip(value, -1, 1)
            value[:3]*=self.cfg.action_scale
            obs=self.env.step(value)
            next_check-=1
            if gripper_open!=self.env.robot.gripper:
                gripper_open=self.env.robot.gripper
                next_check=0
            if next_check==0:
                self.env.camera_capture_rgb()
                input_path=f'imgs/{self.env.tracker2.name}.jpg'
                check_result = self.planner.check_finish(input_path,action['task'],'open' if gripper_open<0.5 else 'closed')
                print('check_result',check_result)
                if 'wrong' in check_result['State']:
                    print('something wrong. Press z to try again or press c to continue')
                    check = cv2.waitKey(0)
                    if check==ord('z'):
                        self.env.step([0,0,0.2,self.env.robot.gripper])
                        self.do_action(action)
                elif 'success' in check_result['State']:
                    return
                next_check = self.cfg.check_period

    def do_action_by_programming(self,action,last_action=None):
        segment_length = self.cfg.segment_length
        self.env.step()
        print('action:',action)
        init_state = self.env.robot.init_state[:3]

        self.env.camera_capture_rgb()
        input_path=f'imgs/{self.env.tracker2.name}.jpg'
        output_path='imgs/SoM.jpg'

        TEXT_PROMPT = f"{action['obj']}. {action['target']}."
        masks,keypoints = get_som(TEXT_PROMPT,input_path,output_path,self.GroundingDINO_model,sam=self.env.tracker2.model.samcontroler.sam_controler)
        IOR = self.planner.get_IOR(output_path,action['task'])
        print('IOR',IOR)
        if 'grasp' in IOR['Action']:
            self.env.robot.gripper_control(0.0)
            obj_num = int(IOR['Object']['number'])
            keypoint = keypoints[obj_num]
            obj_state = self.env.pixel2loc(*keypoint,self.env.tracker2)
            diff = obj_state-self.env.gripper_state
            divided_actions = divide_into_segments(diff,segment_length)
            new_column = np.zeros((len(divided_actions), 1))
            divided_actions = np.hstack((divided_actions, new_column))
            last_actions = np.array([0,0,0,1])
            divided_actions = np.vstack((divided_actions,last_actions))
            last_actions = np.array([[0,0,0.2,1]])
            divided_actions = np.vstack((divided_actions,last_actions))
        elif 'place' in IOR['Action']:
            obj_num = int(IOR['Target']['number'])
            keypoint = keypoints[obj_num]
            obj_state = self.env.pixel2loc(*keypoint,self.env.tracker2)
            diff = obj_state-self.env.gripper_state
            divided_actions = divide_into_segments(diff,segment_length)
            new_column = np.ones((len(divided_actions), 1))
            divided_actions = np.hstack((divided_actions, new_column))
            last_actions = np.array([0,0,0,-0.5])
            divided_actions = np.vstack((divided_actions,last_actions))
            last_actions = np.array([*(init_state-(self.env.gripper_state+diff)),-1])
            divided_actions = np.vstack((divided_actions,last_actions))
        elif 'reach' in IOR['Action']:
            obj_num = int(IOR['Object']['number'])
            keypoint = keypoints[obj_num]
            obj_state = self.env.pixel2loc(*keypoint,self.env.tracker2)
            diff = obj_state-self.env.gripper_state
            diff[2] += 0.1
            divided_actions = divide_into_segments(diff,segment_length)
            new_column = np.zeros((len(divided_actions), 1))
            divided_actions = np.hstack((divided_actions, new_column))
            

        for i in divided_actions:
            key = cv2.waitKey(1) & 0xFF
            obs = self.env.step(i)
        
        
        self.env.camera_capture_rgb()
        input_path=f'imgs/{self.env.tracker2.name}.jpg'
        check_result = self.planner.check_finish(input_path,action)
        print('check_result',check_result)
        if 'wrong' in check_result['State']:
            print('something wrong. Press z to try again or press c to continue')
            check = cv2.waitKey(0)
            if check==ord('z'):
                if 'place' in IOR['Action']:
                    self.do_action(last_action)
                else:
                    self.do_action(action)
        return

    def reset(self):
        obs = self.env.reset(robot=True)

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        robot = Robot(base, base_cyclic)
        env = Env(robot,cfg.env)
        planner = Planner(cfg.planner)
        robridge = RoBridge(env,planner,cfg)
        robridge.solve(cfg.task)

if __name__=='__main__':
    main()