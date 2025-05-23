print('导入视觉大模型模块')
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# 导入中文字体，指定字号
# font = ImageFont.truetype('asset/SimHei.ttf', 26)

from API_KEY import *

# 系统提示词
# SYSTEM_PROMPT = '''
# 我即将说一句给机械臂的指令，你帮我从这句话中提取出起始物体和终止物体，并从这张图中分别找到这两个物体左上角和右下角的像素坐标，输出json数据结构。

# 例如，如果我的指令是：请帮我把红色方块放在房子简笔画上。
# 你输出这样的格式：
# {
#  "start":"红色方块",
#  "start_xyxy":[[102,505],[324,860]],
#  "end":"房子简笔画",
#  "end_xyxy":[[300,150],[476,310]]
# }

# 只回复json本身即可，不要回复其它内容

# 我现在的指令是：
# '''

SYSTEM_PROMPT = '''
我即将说一句给机械臂的指令，你帮我从这句话中提取出起始物体和终止物体，输出json数据结构。

例如，如果我的指令是：请帮我把红色方块放在白色盒子里。
你输出这样的格式：
{
 "start":"Red squares",
 "end":"White square",
}

如果我的指令是：请帮我抽一张纸。
你输出这样的格式：
{
 "start":"White tissue",
 "end":"White Paper",
}

只回复json本身即可，不要回复其它内容

我现在的指令是：
'''

import base64
import warnings

import cv2
# Yi-Vision调用函数
import openai
from openai import OpenAI

warnings.filterwarnings("ignore")

from GroundingDINO.groundingdino.util.inference import (Model, annotate,
                                                        load_image, load_model,
                                                        predict)

CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"    #源码自带的配置文件
CHECKPOINT_PATH = "GroundingDINO\groundingdino_swint_ogc.pth"   #下载的权重文件
DEVICE = "cpu"   #可以选择cpu/cuda
# IMAGE_PATH = "imgs\captured_image.jpg"    #用户设置的需要读取image的路径
# TEXT_PROMPT = "wooden block, yellow cloth, white tissue, red box. "    #用户给出的文本提示
BOX_TRESHOLD = 0.35     #源码给定的边界框判定阈值
TEXT_TRESHOLD = 0.25    #源码给定的文本端获取关键属性阈值


def yi_vision_api(PROMPT='帮我把红色方块放在绿色方块上', img_path='temp/vl_now.jpg'):

    '''
    零一万物大模型开放平台，yi-vision视觉语言多模态大模型API
    '''
    
    # 编码为base64数据
    with open(img_path, 'rb') as image_file:
        image = 'data:image/jpeg;base64,' + base64.b64encode(image_file.read()).decode('utf-8')
    
    # 向大模型发起请求
    completion = client.chat.completions.create(
      model="gpt-4o-2024-11-20",
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": SYSTEM_PROMPT + PROMPT
            },
            {
              "type": "image_url",
              "image_url": {
                "url": image
              }
            }
          ]
        },
      ]
    )
    
    # 解析大模型返回结果
    result = eval(completion.choices[0].message.content.strip())
    print('大模型调用成功！')
    
    return result

# def post_processing_viz(result, img_path, check=False):
    
#     '''
#     视觉大模型输出结果后处理和可视化
#     check：是否需要人工看屏幕确认可视化成功，按键继续或退出
#     '''

#     # 后处理
#     img_bgr = cv2.imread(img_path)
#     img_h = img_bgr.shape[0]
#     img_w = img_bgr.shape[1]
#     # 缩放因子
#     FACTOR = 999
#     # 起点物体名称
#     START_NAME = result['start']
#     # 终点物体名称
#     END_NAME = result['end']
#     # 起点，左上角像素坐标
#     START_X_MIN = int(result['start_xyxy'][0][0] * img_w / FACTOR)
#     START_Y_MIN = int(result['start_xyxy'][0][1] * img_h / FACTOR)
#     # 起点，右下角像素坐标
#     START_X_MAX = int(result['start_xyxy'][1][0] * img_w / FACTOR)
#     START_Y_MAX = int(result['start_xyxy'][1][1] * img_h / FACTOR)
#     # 起点，中心点像素坐标
#     START_X_CENTER = int((START_X_MIN + START_X_MAX) / 2)
#     START_Y_CENTER = int((START_Y_MIN + START_Y_MAX) / 2)
#     # 终点，左上角像素坐标
#     END_X_MIN = int(result['end_xyxy'][0][0] * img_w / FACTOR)
#     END_Y_MIN = int(result['end_xyxy'][0][1] * img_h / FACTOR)
#     # 终点，右下角像素坐标
#     END_X_MAX = int(result['end_xyxy'][1][0] * img_w / FACTOR)
#     END_Y_MAX = int(result['end_xyxy'][1][1] * img_h / FACTOR)
#     # 终点，中心点像素坐标
#     END_X_CENTER = int((END_X_MIN + END_X_MAX) / 2)
#     END_Y_CENTER = int((END_Y_MIN + END_Y_MAX) / 2)

#     width = START_X_MAX - START_X_MIN

#     # 可视化
#     # 画起点物体框
#     img_bgr = cv2.rectangle(img_bgr, (START_X_MIN, START_Y_MIN), (START_X_MAX, START_Y_MAX), [0, 0, 255], thickness=3)
#     # 画起点中心点
#     img_bgr = cv2.circle(img_bgr, (START_X_CENTER, START_Y_CENTER), 6, [0, 0, 255], thickness=-1)
#     # 画终点物体框
#     img_bgr = cv2.rectangle(img_bgr, (END_X_MIN, END_Y_MIN), (END_X_MAX, END_Y_MAX), [255, 0, 0], thickness=3)
#     # 画终点中心点
#     img_bgr = cv2.circle(img_bgr, (END_X_CENTER, END_Y_CENTER), 6, [255, 0, 0], thickness=-1)
#     # 写中文物体名称
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # BGR 转 RGB
#     img_pil = Image.fromarray(img_rgb) # array 转 pil
#     draw = ImageDraw.Draw(img_pil)
#     # 写起点物体中文名称
#     draw.text((START_X_MIN, START_Y_MIN-32), START_NAME, font=font, fill=(255, 0, 0, 1)) # 文字坐标，中文字符串，字体，rgba颜色
#     # 写终点物体中文名称
#     draw.text((END_X_MIN, END_Y_MIN-32), END_NAME, font=font, fill=(0, 0, 255, 1)) # 文字坐标，中文字符串，字体，rgba颜色
#     img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR) # RGB转BGR
#     # 保存可视化效果图
#     cv2.imwrite('imgs/vl_now_viz.jpg', img_bgr)

#     # formatted_time = time.strftime("%Y%m%d%H%M", time.localtime())
#     # cv2.imwrite('visualizations/{}.jpg'.format(formatted_time), img_bgr)

#     # 在屏幕上展示可视化效果图
#     cv2.imshow('vlm', img_bgr) 

#     if check:
#         print('请确认可视化成功，按c键继续，按q键退出')
#         while(True):
#             key = cv2.waitKey(10) & 0xFF
#             if key == ord('c'): # 按c键继续
#                 break
#             if key == ord('q'): # 按q键退出
#                 # exit()
#                 cv2.destroyAllWindows()   # 关闭所有opencv窗口
#                 raise NameError('按q退出')
#     else:
#         if cv2.waitKey(1) & 0xFF == None:
#             pass

#     return START_X_CENTER, START_Y_CENTER, END_X_CENTER, END_Y_CENTER, width


def _project_keypoints_to_img(rgb, candidate_pixels):
    projected = rgb.copy()
    # overlay keypoints on the image
    for keypoint_count, pixel in enumerate(candidate_pixels):
        pixel = pixel[1], pixel[0]
        displayed_text = f"{keypoint_count}"
        text_length = len(displayed_text)
        # draw a box
        box_width = 30 + 10 * (text_length - 1)
        box_height = 30
        cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (255, 255, 255), -1)
        cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (0, 0, 0), 2)
        # draw text
        org = (pixel[1] - 7 * (text_length), pixel[0] + 7)
        color = (255, 0, 0)
        cv2.putText(projected, str(keypoint_count), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        keypoint_count += 1
    return projected

def post_processing_viz(result, img_path, check=False):
    
    '''
    视觉大模型输出结果后处理和可视化
    check：是否需要人工看屏幕确认可视化成功，按键继续或退出
    '''

    
    image_source, image = load_image(img_path)
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
    # 起点物体名称
    START_NAME = result['start']
    # 终点物体名称
    END_NAME = result['end']

    TEXT_PROMPT = f"{START_NAME}. {END_NAME}." 

    boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=DEVICE,
    )

    desired_order = [START_NAME, END_NAME]
    desired_order_lower = [category.lower() for category in desired_order]
     
    # 按类别分组索引
    category_indices = defaultdict(list)
    for idx, phrase in enumerate(phrases):
        category_indices[phrase].append(idx)

    # 找到每种类别概率最大的索引
    max_indices = {}
    for category, indices in category_indices.items():
        # 获取对应的 logits 值及其索引
        max_index = max(indices, key=lambda i: logits[i])
        max_indices[category] = max_index


    # 按照指定顺序重排 phrases 和 logits
    ordered_phrases = []
    ordered_logits = []
    ordered_boxes = []
    for category in desired_order_lower:
        if category in max_indices:
            idx = max_indices[category]
            ordered_phrases.append(phrases[idx])
            ordered_logits.append(logits[idx])
            ordered_boxes.append(boxes[idx])

    # 将 ordered_logits 转为 tensor
    ordered_logits = torch.stack(ordered_logits)
    ordered_boxes = torch.stack(ordered_boxes)
   
    annotated_frame = annotate(image_source=image_source, boxes=ordered_boxes, logits=ordered_logits, phrases=ordered_phrases)

    # 后处理
    img_h, img_w, _ = image_source.shape
    # # 起点，左上角像素坐标
    # START_X_MIN = boxes[0][0] * img_w 
    # START_Y_MIN = boxes[0][1] * img_h 
    # # 起点，右下角像素坐标
    # START_X_MAX = boxes[0][2] * img_w
    # START_Y_MAX = boxes[0][3] * img_h 
    # 起点，中心点像素坐标
    # START_X_CENTER = int((START_X_MIN + START_X_MAX) / 2)
    # START_Y_CENTER = int((START_Y_MIN + START_Y_MAX) / 2)
    # # 终点，左上角像素坐标
    # END_X_MIN = boxes[1][0] * img_w 
    # END_Y_MIN = boxes[1][1] * img_h 
    # # 终点，右下角像素坐标
    # END_X_MAX = boxes[1][2] * img_w 
    # END_Y_MAX = boxes[1][3] * img_h 
    # # 终点，中心点像素坐标
    # END_X_CENTER = int((END_X_MIN + END_X_MAX) / 2)
    # END_Y_CENTER = int((END_Y_MIN + END_Y_MAX) / 2)

    # 起点，中心点像素坐标
    START_X_CENTER = ordered_boxes[0][0] * img_w
    START_Y_CENTER = ordered_boxes[0][1] * img_h
    # 终点，中心点像素坐标
    END_X_CENTER = ordered_boxes[1][0] * img_w 
    END_Y_CENTER = ordered_boxes[1][1] * img_h 

    # 保存可视化效果图
    cv2.imwrite('imgs/vl_now_viz.jpg', annotated_frame)

    keypoints = [[int(START_X_CENTER.item()),int(START_Y_CENTER.item())],[int(END_X_CENTER.item()),int(END_Y_CENTER.item())]]
    projected_img = _project_keypoints_to_img(image_source,keypoints)
    projected_img = cv2.cvtColor(projected_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('imgs/projected_img.jpg', projected_img)

    # 在屏幕上展示可视化效果图
    cv2.imshow('vlm', annotated_frame) 
    
    if check:
        print('请确认可视化成功，按c键继续，按q键退出')
        while(True):
            key = cv2.waitKey(10) & 0xFF
            if key == ord('c'): # 按c键继续
                break
            if key == ord('q'): # 按q键退出
                # exit()
                cv2.destroyAllWindows()   # 关闭所有opencv窗口
                # raise NameError('按q退出')
    else:
        if cv2.waitKey(1) & 0xFF == None:
            pass

    return START_X_CENTER, START_Y_CENTER, END_X_CENTER, END_Y_CENTER

