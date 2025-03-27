import argparse
import os
import sys
import time

import cv2

import utilities
from robot import *
from utils_vlm import *

height_safe = 0.3606354296207428
height_start = 0.026069933772087097
height_end_on = 0.07643684327602386
height_end = 0.02989890545606613
height_paper = 0.1107092159986496

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
parser = argparse.ArgumentParser()
args = utilities.parseConnectionArguments(parser)

def vlm_move(PROMPT='帮我乌龙茶放在红色方块上', input_way='keyboard'):
    '''
    input_way：speech语音输入，keyboard键盘输入
    '''
    
    # 机械臂归零
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        move_to_top_view(base)
        gripper_control(base, 0.00)
    
    ## 第一步：完成手眼标定
    print('第一步：完成手眼标定')
    
    ## 第二步：发出指令
    # PROMPT_BACKUP = '帮我把绿色方块放在小猪佩奇上' # 默认指令
    
    # if input_way == 'keyboard':
    #     PROMPT = input('第二步：输入指令')
    #     if PROMPT == '':
    #         PROMPT = PROMPT_BACKUP
    # elif input_way == 'speech':
    #     record() # 录音
    #     PROMPT = speech_recognition() # 语音识别
    print('第二步，给出的指令是：', PROMPT)
    
    ## 第三步：拍摄俯视图
    print('第三步：拍摄俯视图')
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        get_image()
    
    ## 第四步：将图片输入给多模态视觉大模型
    print('第四步：将图片输入给多模态视觉大模型')
    img_path = 'imgs/captured_image.jpg'
    
    n = 1
    while n < 5:
        try:
            print('尝试第 {} 次访问多模态大模型'.format(n))
            result = yi_vision_api(PROMPT, img_path='imgs/captured_image.jpg')
            print('多模态大模型调用成功！')
            print(result)
            break
        except Exception as e:
            print('多模态大模型返回数据结构错误，再尝试一次', e)
            n += 1
    
    ## 第五步：视觉大模型输出结果后处理和可视化
    print('第五步：视觉大模型输出结果后处理和可视化')
    START_X_CENTER, START_Y_CENTER, END_X_CENTER, END_Y_CENTER = post_processing_viz(result, img_path, check=False)
    position = 0.65
    
    ## 第六步：手眼标定转换为机械臂坐标
    print('第六步：手眼标定，将像素坐标转换为机械臂坐标')
    # 起点，机械臂坐标
    START_X_MC, START_Y_MC = eye2hand(START_X_CENTER, START_Y_CENTER)
    # 终点，机械臂坐标
    END_X_MC, END_Y_MC = eye2hand(END_X_CENTER, END_Y_CENTER)
    # print(START_X_CENTER, START_Y_CENTER)
    # print(START_X_MC, START_Y_MC)
    # print(END_X_CENTER, END_Y_CENTER)
    # print(END_X_MC, END_Y_MC)
    
    # ## 第七步：吸泵吸取移动物体
    # print('第七步：吸泵吸取移动物体')
    # pump_move(mc=mc, XY_START=[START_X_MC, START_Y_MC], XY_END=[END_X_MC, END_Y_MC])
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        go_to_base1(base, base_cyclic, START_X_MC, START_Y_MC, height_safe)
        go_to_base1(base, base_cyclic, START_X_MC, START_Y_MC, height_start)
        gripper_control(base, position)
        go_to_base1(base, base_cyclic, START_X_MC, START_Y_MC, height_safe)
        go_to_base1(base, base_cyclic, END_X_MC, END_Y_MC, height_safe)
        go_to_base1(base, base_cyclic, END_X_MC, END_Y_MC, height_end_on)
        gripper_control(base, 0.00)
        move_to_top_view(base)
    
    ## 第八步：收尾
    print('第八步：任务完成')
    

def vlm_move_to(PROMPT='帮我乌龙茶放在红色方块上', input_way='keyboard'):
    '''
    input_way：speech语音输入，keyboard键盘输入
    '''
    
    # 机械臂归零
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        move_to_top_view(base)
        gripper_control(base, 0.00)
    
    ## 第一步：完成手眼标定
    print('第一步：完成手眼标定')
    
    ## 第二步：发出指令
    # PROMPT_BACKUP = '帮我把绿色方块放在小猪佩奇上' # 默认指令
    
    # if input_way == 'keyboard':
    #     PROMPT = input('第二步：输入指令')
    #     if PROMPT == '':
    #         PROMPT = PROMPT_BACKUP
    # elif input_way == 'speech':
    #     record() # 录音
    #     PROMPT = speech_recognition() # 语音识别
    print('第二步，给出的指令是：', PROMPT)
    
    ## 第三步：拍摄俯视图
    print('第三步：拍摄俯视图')
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        get_image()
    
    ## 第四步：将图片输入给多模态视觉大模型
    print('第四步：将图片输入给多模态视觉大模型')
    img_path = 'imgs/captured_image.jpg'
    
    n = 1
    while n < 5:
        try:
            print('尝试第 {} 次访问多模态大模型'.format(n))
            result = yi_vision_api(PROMPT, img_path='imgs/captured_image.jpg')
            print('多模态大模型调用成功！')
            print(result)
            break
        except Exception as e:
            print('多模态大模型返回数据结构错误，再尝试一次', e)
            n += 1
    
    ## 第五步：视觉大模型输出结果后处理和可视化
    print('第五步：视觉大模型输出结果后处理和可视化')
    START_X_CENTER, START_Y_CENTER, END_X_CENTER, END_Y_CENTER = post_processing_viz(result, img_path, check=False)
    position = 0.65
    
    ## 第六步：手眼标定转换为机械臂坐标
    print('第六步：手眼标定，将像素坐标转换为机械臂坐标')
    # 起点，机械臂坐标
    START_X_MC, START_Y_MC = eye2hand(START_X_CENTER, START_Y_CENTER)
    # 终点，机械臂坐标
    END_X_MC, END_Y_MC = eye2hand(END_X_CENTER, END_Y_CENTER)
    # print(START_X_CENTER, START_Y_CENTER)
    # print(START_X_MC, START_Y_MC)
    # print(END_X_CENTER, END_Y_CENTER)
    # print(END_X_MC, END_Y_MC)
    
    # ## 第七步：吸泵吸取移动物体
    # print('第七步：吸泵吸取移动物体')
    # pump_move(mc=mc, XY_START=[START_X_MC, START_Y_MC], XY_END=[END_X_MC, END_Y_MC])
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        go_to_base1(base, base_cyclic, START_X_MC, START_Y_MC, height_safe)
        go_to_base1(base, base_cyclic, START_X_MC, START_Y_MC, height_start)
        gripper_control(base, position)
        go_to_base1(base, base_cyclic, START_X_MC, START_Y_MC, height_safe)
        go_to_base1(base, base_cyclic, END_X_MC, END_Y_MC, height_safe)
        go_to_base1(base, base_cyclic, END_X_MC, END_Y_MC, height_end)
        gripper_control(base, 0.00)
        move_to_top_view(base)
    
    ## 第八步：收尾
    print('第八步：任务完成')


def pumping_paper(PROMPT='帮我抽一张纸', input_way='keyboard'):
    '''
    input_way：speech语音输入，keyboard键盘输入
    '''
    
    # 机械臂归零
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        move_to_top_view(base)
        gripper_control(base, 0.00)
    
    ## 第一步：完成手眼标定
    print('第一步：完成手眼标定')
    
    ## 第二步：发出指令
    print('第二步，给出的指令是：', PROMPT)
    
    ## 第三步：拍摄俯视图
    print('第三步：拍摄俯视图')
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        get_image()
    
    ## 第四步：将图片输入给多模态视觉大模型
    print('第四步：将图片输入给多模态视觉大模型')
    img_path = 'imgs/captured_image.jpg'
    
    n = 1
    while n < 5:
        try:
            print('尝试第 {} 次访问多模态大模型'.format(n))
            result = yi_vision_api(PROMPT, img_path='imgs/captured_image.jpg')
            print('多模态大模型调用成功！')
            print(result)
            break
        except Exception as e:
            print('多模态大模型返回数据结构错误，再尝试一次', e)
            n += 1
    
    ## 第五步：视觉大模型输出结果后处理和可视化
    print('第五步：视觉大模型输出结果后处理和可视化')
    START_X_CENTER, START_Y_CENTER, _, _ = post_processing_viz(result, img_path, check=False)
    position = 1.00
    
    ## 第六步：手眼标定转换为机械臂坐标
    print('第六步：手眼标定，将像素坐标转换为机械臂坐标')
    # 起点，机械臂坐标
    START_X_MC, START_Y_MC = eye2hand(START_X_CENTER, START_Y_CENTER)
    # 终点，机械臂坐标
    # END_X_MC, END_Y_MC = eye2hand(END_X_CENTER, END_Y_CENTER)

    
    # ## 第七步：吸泵吸取移动物体
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        go_to_base1(base, base_cyclic, START_X_MC, START_Y_MC, height_safe)
        go_to_base1(base, base_cyclic, START_X_MC, START_Y_MC, height_paper)
        gripper_control(base, position)
        go_to_base1(base, base_cyclic, START_X_MC, START_Y_MC, height_safe)
        move_to_top_view(base)
    
    ## 第八步：收尾
    print('第八步：任务完成')

    
if __name__ == "__main__":
    vlm_move()   
    
