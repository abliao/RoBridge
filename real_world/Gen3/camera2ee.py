import numpy as np

import numpy as np
import cv2

def pixel_to_ee(u, v, depth, K, R, t):
    """
    将像素坐标 (u, v) 和深度值转换到机械臂末端坐标系。
    
    :param u: 像素 x 坐标
    :param v: 像素 y 坐标
    :param depth: 深度值 (单位: 米)
    :param K: 相机内参矩阵 (3x3)
    :param R: 相机到机械臂末端的旋转矩阵 (3x3)
    :param t: 相机到机械臂末端的平移向量 (3,)
    :return: 机械臂末端坐标系下的 3D 点 (x, y, z)
    """
    # 计算相机坐标
    K_inv = np.linalg.inv(K)
    pixel_homogeneous = np.array([u, v, 1])
    cam_coords = K_inv @ pixel_homogeneous * depth
    
    # print(cam_coords)
    # 变换到机械臂末端坐标系
    ee_coords = R @ cam_coords + t
    if ee_coords[1] > 0:  # 如果 z 坐标小于0，则取绝对值
        ee_coords[1] = ee_coords[1]+ 0.03
    elif ee_coords[1] < 0:  # 如果 z 坐标小于0，则取绝对值
        ee_coords[1] = ee_coords[1]+ 0.02
    return ee_coords

def test_transformation(u, v):
    """
    测试像素坐标转换到机械臂末端坐标系
    """
    # RGB 相机内参
    # K = np.array([[628.8834145,    0.,         322.31764198],
    #             [  0.,         630.77241461, 263.78024913],
    #             [  0.,           0.,           1.        ]], dtype=np.float64)
    
    # K = np.array([[653.682312,   0.,       311.753418],
    #             [  0.,       651.856018, 232.400955],
    #             [  0.,         0.,        1.      ]],dtype=np.float64)
    
    K = np.array([[605.4453735351562,   0.,       332.6232604980469],
                [  0.,       605.1100463867188, 245.16200256347656],
                [  0.,         0.,        1.      ]],dtype=np.float64)
    # K = np.array([[640.52644823,   0.,         321.89076049],
    #                 [  0.,         641.9219399,  255.92716849],
    #                 [  0.,           0.,           1.        ]],dtype=np.float64)
    # 旋转矩阵 (单位矩阵)
    # R = np.eye(3)
    R = np.array([
        [0,  1,  0],  # X_c → Y_e
        [-1, 0,  0],  # Y_c → -X_e
        [0,  0, -1]   # Z_c → -Z_e
    ])
    # R = R.T
    # 平移向量
    # t = np.array([-0.027060, -0.009970, -0.004706])
    t = np.array([-0.05639, 0, 0])
    # t = np.array([0.027060, 0.009970, 0.004706])
    
    # 测试像素坐标和深度值
    # u, v = 960, 540  # 图像中心
    # depth = 1.0  # 假设深度为 1 米
    dist_coeffs = np.zeros(5)
    depth = 0.25 # 0.48+0.08-0.03 0.26
    ee_coords = pixel_to_ee(u, v, depth, K, R, t)
    print("机械臂末端坐标系下的坐标:", ee_coords)
    return ee_coords

