import os
import numpy as np
import cv2
import re
import torch
from torchvision.ops import box_convert
from PIL import Image
from real_world.GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

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

def normalize_depth(depth):
    near, far = 0.022578668824764146,112.89334664718366
    mask = depth==1.
    depth[depth<near] = near
    depth[depth>far] = far
    normalized_depth = (1-near/depth)/(1 - near / far)
    normalized_depth[mask] = 1.
    return normalized_depth

def divide_into_segments(point, segment_length=0.05):
        length = np.linalg.norm(point)
        
        num_segments = int(length // segment_length)
        
        points = [np.array([0, 0, 0])] 
        for i in range(1, num_segments + 1):
            new_point = i * segment_length / length * point 
            points.append(new_point)
        
        if not np.allclose(points[-1], point):
            points.append(point)
        
        points = np.array(points)
        points = np.diff(points, axis=0)
        return points

def find_next_episode_number(data_folder='data'):
    os.makedirs(data_folder, exist_ok=True)
    existing_files = [f for f in os.listdir(data_folder) if f.startswith('episode_')]
    existing_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
    n = 0
    while n in existing_numbers:
        n += 1
    return n

def save_video(imgs,path):
    images = imgs 

    # Convert PIL images to numpy arrays
    frame_array = [np.array(img) for img in images]

    # Get dimensions of the first image
    height, width, layers = frame_array[0].shape

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(f'{path}/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    for frame in frame_array:
        # Convert RGB to BGR (OpenCV uses BGR format by default)
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

    out.release()

def _project_keypoints_to_img(rgb, candidate_pixels):
    projected = rgb.copy()
    # overlay keypoints on the image
    for keypoint_count, pixel in enumerate(candidate_pixels):
        pixel = pixel[1], pixel[0]
        displayed_text = f"{keypoint_count}"
        text_length = len(displayed_text)
        # draw a box
        box_width = 25 + 10 * (text_length - 1)
        box_height = 25
        radius = 10
        cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (0, 0, 0), -1)
        # cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (46, 84, 161), 2)
        # draw text
        org = (pixel[1] - 3 * (text_length), pixel[0] + 5)
        color = (255, 255, 255)
        cv2.putText(projected, str(keypoint_count), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        keypoint_count += 1
    return projected

def calculate_mask_center(mask):
    y_indices, x_indices= np.where(mask >0)
    x_center = np.mean(x_indices)
    y_center = np.mean(y_indices)

    return int(x_center), int(y_center)

def get_som(TEXT_PROMPT,input_path,output_path,GroundingDINO_model,sam):
    image_source, image = load_image(input_path)
    boxes, logits, phrases = predict(
        model=GroundingDINO_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device='cpu',
    )
    sam.reset_image() 
    sam.set_image(image_source)
    result_masks = []
    keypoints = []
    img_h, img_w, _ = image_source.shape

    boxes*= torch.Tensor([img_w,img_h,img_w,img_h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    for box in xyxy:
        masks, scores, logits = sam.predictor.predict(
                                        box=box,
                                        multimask_output=True)
        index = np.argmax(scores)
        mask = masks[index]
        mask_image = (mask.astype(np.uint8) * 255)
        img = Image.fromarray(mask_image)

        result_masks.append(mask)
        x,y = calculate_mask_center(mask)
        keypoints.append([x,y])

    projected_img = _project_keypoints_to_img(image_source.copy(),keypoints)
    projected_img = cv2.cvtColor(projected_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, projected_img)
    cv2.imshow('projected_img', projected_img) 
    cv2.waitKey(0)
    cv2.destroyWindow('projected_img')

    return masks,keypoints

def extract_number(string):
    match = re.search(r'\d+', string)
    if match:
        return match.group()
    return None