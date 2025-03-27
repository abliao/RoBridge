import cv2
import numpy as np
import torch
import os
import sys
from PIL import Image


import requests
import json
from real_world.Track_Anything.tools.painter import mask_painter
import time
from real_world.Gen3.D435i import Camera
from real_world.Track_Anything.track_anything import TrackingAnything

#camera test
WIDTH=640
HEIGHT=480
fps =60

class InteractiveTracker:
    def __init__(self, video_source=0, camera = None, rotate=False, world_matrix=None, world_offset=None):
        self.args = self.get_default_args()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = "vit_h"
        SAM_checkpoint_dict = {
            'vit_h': "real_world/Track_Anything/checkpoints/sam_vit_h_4b8939.pth",
            'vit_l': "real_world/Track_Anything/checkpoints/sam_vit_l_0b3195.pth", 
            "vit_b": "real_world/Track_Anything/checkpoints/sam_vit_b_01ec64.pth"
        }
        self.sam_checkpoint = SAM_checkpoint_dict[self.args.sam_model_type] 
        self.xmem_checkpoint = "real_world/Track_Anything/checkpoints/XMem-s012.pth"
        self.e2fgvi_checkpoint = "real_world/Track_Anything/checkpoints/E2FGVI-HQ-CVPR22.pth"
        
        os.makedirs("real_world/Track_Anything/checkpoints", exist_ok=True)
        
        self.download_models()
        
        # init Track-Anything
        
        self.model = TrackingAnything(self.sam_checkpoint, self.xmem_checkpoint, self.e2fgvi_checkpoint, self.args)
        
        self.video_state = None
        self.click_state = [[], []] 
        self.tracking_mode = False
        self.gripper_mode = False
        self.current_frame_idx = 0
        self.mask = None
        self.final_mask = None
        self.xmem_init = False
        self.rotate = rotate
        if camera is None:
            self.camera = Camera(WIDTH, HEIGHT, fps, video_source)
        else:
            self.camera = camera
        self.name = f'Camera {video_source}'
        if world_matrix is None:
            world_matrix = np.eye(3,3)
        if world_offset is None:
            world_offset = np.zeros(3)
        self.world_matrix = np.array(world_matrix)
        self.world_offset = np.array(world_offset)
        self.init()
        
    def download_models(self):
        if not os.path.exists(self.sam_checkpoint):
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            self.download_file(url, self.sam_checkpoint)
            

        if not os.path.exists(self.xmem_checkpoint):
            url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
            self.download_file(url, self.xmem_checkpoint)
            
    def download_file(self, url, filepath):
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    def get_default_args(self):
        class Args:
            def __init__(self):
                self.sam_model_type = "vit_b"
                self.port = 12212
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.mask_save = False
                self.mask_save_path = None
        return Args()
    
    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click - positive sample
            self.click_state[0].append([x, y])
            self.click_state[1].append(1)
            self.process_click()
        elif event == cv2.EVENT_RBUTTONDOWN:  # Right click - Negative samples
            self.click_state[0].append([x, y])
            self.click_state[1].append(0)
            self.process_click()
            
    def process_click(self):
        if self.video_state is not None:
            self.model.samcontroler.sam_controler.reset_image() 
            self.model.samcontroler.sam_controler.set_image(self.video_state["origin_images"][0])
            current_frame = self.video_state["origin_images"][self.current_frame_idx]
            points = np.array(self.click_state[0])
            labels = np.array(self.click_state[1])
            mask, logit, painted_image = self.model.first_frame_click(
                image=current_frame,
                points=points,
                labels=labels,
                multimask=True
            )
            mask = mask.astype(np.uint8)
            self.mask = mask
            if self.gripper_mode:
                self.gripper_mask = mask
            self.video_state["masks"][self.current_frame_idx] = mask
            painted_image = np.array(painted_image)
            self.video_state["painted_images"][self.current_frame_idx] = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)
    
    def get_frame(self):
        self.camera.align_frames()
        rgb = self.camera.get_rgb_frame() 
        depth = self.camera.get_depth_frame()*self.camera.depth_scale
        if self.rotate:
            rgb = cv2.flip(cv2.transpose(rgb), 1)
            depth = cv2.flip(cv2.transpose(depth), 1)
            rgb = rgb[-480:]
            depth = depth[-480:]
        return rgb,depth

    def camera2world(self, ee):
        if not isinstance(ee, np.ndarray):
            ee = np.array(ee)
        ee = np.dot(self.world_matrix, ee)
        ee += self.world_offset
        return ee
    
    def camera_capture_rgb(self):
        self.camera.align_frames()
        rgb = self.camera.get_rgb_frame()
        Image.fromarray(rgb).save(f'imgs/{self.name}.jpg')

    def init(self):
            
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self.handle_click)
        
        # Initialize video status
        frames = []
        ret, (frame, depth) = True, self.get_frame() 
        if ret:
            frames.append(frame)
            self.video_state = {
                "user_name": time.time(),
                "video_name": "realtime",
                "origin_images": frames,
                "origin_depths": [depth],
                "painted_images": [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)],
                "masks": [np.zeros((frame.shape[0], frame.shape[1]), np.uint8)],
                "logits": [None],
                "select_frame_number": 0,
                "fps": fps
            }
            self.final_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)

    def update_frame(self):
        self.camera.align_frames()
        ret, (frame_rgb, depth) = True, self.get_frame() 
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self.video_state["origin_images"] = [frame_rgb]
        self.video_state["origin_depths"] = [depth]
        self.current_frame_idx = 0
        return frame_rgb, frame

    def get_rel_loc(self,id):
        mask = self.final_mask
        depth = self.video_state["origin_depths"][0]
        condition = (mask == id) & (depth>0.05)
        y_indices, x_indices = np.where(condition)

        try:
            # Calculate the center point of these points
            center_x = int(np.mean(x_indices))
            center_y = int(np.mean(y_indices))
            if depth[center_y, center_x] < 0.05:
                for y, x in zip(y_indices, x_indices):
                    if depth[y, x] > 0.05:
                        center_y = y
                        center_x = x
                        print('Adjusted center', center_y, center_x, depth[center_y, center_x])
                        break
            if self.rotate:
                center_y+=160
                center_y, center_x = (480-1-center_x,center_y)
            obj_move = self.camera.get_pixel_to_ee(u=center_x, v=center_y)
            return obj_move
        except:
            return False

    def get_resize_img(self, w, h):
        rgb = self.video_state["origin_images"][0].copy()
        depth = self.video_state["origin_depths"][0].copy()
        mask = self.final_mask.copy()
        if self.rotate:
            depth[mask==0] = 1.
            depth[mask==1] = 0.15
            depth_new = np.ones((640,640),dtype=np.float32)
            depth_new[:480,:480] = depth
            depth = depth_new
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return rgb, depth, mask

    def run(self):    
        frame_rgb, frame = self.update_frame()

        if (self.final_mask.max()>0 or self.xmem_init) and self.tracking_mode:
            if not self.xmem_init:
                mask, logit, painted_image = self.model.generate(frame_rgb, template_mask=self.final_mask)
                self.xmem_init = True
            else:
                mask, logit, painted_image = self.model.generate(frame_rgb)
            self.final_mask = mask
            display_frame = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)
            self.video_state["painted_images"] = [display_frame]
        else:
            display_frame = self.video_state["painted_images"][0]
        
        display_frame = display_frame.copy()
        
        cv2.putText(display_frame, "Left click: Add positive point", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Right click: Add negative point", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, "Press 't': Toggle tracking", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(display_frame, "Press 'r': Reset", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(display_frame, "Press 'q': Quit", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        mode_text = "Tracking" if self.tracking_mode else "Annotation"
        cv2.putText(display_frame, f"Mode: {mode_text}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.putText(display_frame, "Press 'w': Mark mask", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


        cv2.imshow(self.name, display_frame)
        
        
    def control_signal(self,key):
        if key == ord('q'):  # exit
            return
        elif key == ord('r'):  # reset
            self.mask = None
            self.click_state = [[], []]
            self.tracking_mode = False
            self.xmem_init = False
            self.mask = None
            self.final_mask = None
            self.model.xmem.clear_memory()

            frames = []
            ret, (frame, depth) = True, self.get_frame() 
            if ret:
                frames.append(frame)
                self.video_state = {
                    "user_name": time.time(),
                    "video_name": "realtime",
                    "origin_images": frames,
                    "origin_depths": [depth],
                    "painted_images": [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)],
                    "masks": [np.zeros((frame.shape[0], frame.shape[1]), np.uint8)],
                    "logits": [None],
                    "select_frame_number": 0,
                    "fps": fps
                }
                self.final_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        elif key == ord('w'):  
            num = len(np.unique(self.final_mask))-1
            self.mask[self.mask!=0]+=num
            self.final_mask[self.mask>0] = self.mask[self.mask>0]
            frame_rgb, frame = self.update_frame()
            self.video_state['painted_images'] = [frame] 
            self.click_state = [[], []]

        elif key == ord('t'):  
            self.tracking_mode = not self.tracking_mode

    def reset(self):
        self.control_signal(ord('r'))

if __name__ == "__main__":
    tracker = InteractiveTracker(1)
    while True:
        tracker.run() 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key in [ord('r'),ord('w'),ord('t')]:
            tracker.control_signal(key)
    cv2.destroyAllWindows()