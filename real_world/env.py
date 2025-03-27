import sys
import os
import time
from collections import defaultdict
from typing import Tuple, List
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from collections import deque

from robot import Robot
import argparse
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import \
    BaseCyclicClient

# Import the utilities helper module
import utilities
from D435i import Camera
from track_anything_D435i import InteractiveTracker
#camera test
WIDTH=640
HEIGHT=480
fps =60

def normalize_depth(depth):
    near, far = 0.022578668824764146,112.89334664718366
    mask = depth==1.
    depth[depth<near] = near
    depth[depth>far] = far
    normalized_depth = (1-near/depth)/(1 - near / far)
    normalized_depth[mask] = 1.
    return normalized_depth

class Env:
    def __init__(self, robot, cfg):
        self.robot = robot

        # Third-view camera
        video_source = cfg.third_view.video_source
        self.camera = Camera(WIDTH, HEIGHT, fps, video_source)
        self.tracker = InteractiveTracker(camera=self.camera,**cfg.third_view)

        # First-view camera
        video_source = cfg.first_view.video_source
        self.camera2 = Camera(WIDTH, HEIGHT, fps, video_source)
        self.tracker2 = InteractiveTracker(camera=self.camera2, **cfg.first_view)
        self.trackers = [self.tracker, self.tracker2]

        self.task_info = None
        self.obs = None
        self._frames = deque([], maxlen=cfg.num_frames)
        self.num_frames = cfg.num_frames
        self.w, self.h = cfg.width, cfg.height

        self.gripper_state = None
        self.obj_state = None

    def control_signal(self,signal):
        for tracker in self.trackers:
            tracker.control_signal(signal)

    def reset(self, robot=True):
        if robot:
            self.robot.reset()
        for tracker in self.trackers:
            tracker.reset()
        self.task_info = None
        self.obs = None
        self.obj_state = None
        for i in range(self.num_frames):
            self.get_obs()
        return self.obs

    def show(self,rgb,depth):
        if rgb:
            rgb = (rgb.copy() *255).astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 
            cv2.imshow('Env masked rgb',rgb)


        depth = (depth.copy() *255).astype(np.uint8)
        
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR) 
        
        cv2.imshow('Env masked depth',depth)
    
    def camera_capture_rgb(self):
        for tracker in self.trackers:
            tracker.camera_capture_rgb()

    def pixel2loc(self,u,v,tracker):
        camera_rel_obj_state = tracker.camera.get_pixel_to_ee(u=u, v=v)
        world_rel_obj_state = np.array(tracker.camera2world(camera_rel_obj_state)).astype(np.float32)
        self.obj_state = self.gripper_state+world_rel_obj_state
        return self.obj_state
    
    def get_obs(self):
        for tracker in self.trackers:
            tracker.run()
        rgb, depth, mask = self.tracker.get_resize_img(self.w,self.h)
        masked_rgb = np.zeros_like(rgb)
        masked_rgb[mask==1] =  np.array([1.,0,0])
        masked_rgb[mask==2] =  np.array([0,1.,0])
        masked_rgb[mask==3] =  np.array([0,0,1.])
        
        rgb, depth, mask = self.tracker2.get_resize_img(self.w,self.h)
        depth[mask==0] = 1.

        third_depth = np.ones_like(depth)
        obs = np.concatenate([masked_rgb, depth[...,None], third_depth[...,None]], axis=2).astype(np.float32)
        obs = obs.transpose(2, 0, 1)

        # self.show(masked_rgb,depth)
        self._frames.append(obs)
        self.obs = np.concatenate(list(self._frames),axis=0)

        self.gripper_state = np.array(self.robot.get_ee_pose()[:3]).astype(np.float32)
        if self.tracker2.final_mask.max()>=2:
            camera_rel_obj_state = self.tracker2.get_rel_loc(id=2)
            if camera_rel_obj_state !=False:
                world_rel_obj_state = np.array(self.tracker2.camera2world(camera_rel_obj_state)).astype(np.float32)
                
                if self.obj_state is None:
                    print('camera_rel_obj_state',camera_rel_obj_state)
                    print('self.gripper_state',self.gripper_state)
                    print('world_rel_obj_state',world_rel_obj_state)
                    print('self.obj_state',self.gripper_state+world_rel_obj_state)

                    self.obj_state = self.gripper_state+world_rel_obj_state

        return self.obs.copy()

    def step(self,action=None):
        if action is not None:
            self.robot.move_to_det(pos_x=action[0],pos_y=action[1],pos_z=action[2])
            self.robot.gripper_control(action[3],rel=True)
        obs = self.get_obs()
        return obs
