import cv2
import numpy as np
import time

# 设置棋盘格尺寸（角点数）
chessboard_size = (8, 11)  # 9列6行
square_size = 0.020  # 每个棋盘格的边长，单位为米

# 生成棋盘格的三维坐标（世界坐标系）
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size  # 单位为米

# 用于存储对象点和图像点
obj_points = []  # 3D点
img_points = []  # 2D点

# 采集图像并检测角点
def get_image():
    src = "rtsp://admin:admin@192.168.1.10/color"
    cap = cv2.VideoCapture(src)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # 缓冲区大小
    
    # 检查相机是否打开
    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return
    i = 1
    while True:
        print(i)
        ret, frame = cap.read()
        
        # 检查帧是否正确读取
        if not ret or frame is None:
            print("Error: Failed to capture image.")
            break
        if i%100==0:
            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('frame', gray)
            # cv2.imwrite('E:/code/agent_demo/imgs/captured_gray_image.jpg', gray)
            # 查找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            print(ret, corners)
            if ret:
                # 添加对象点和图像点
                obj_points.append(objp)
                img_points.append(corners)
                
                # 在图像中绘制棋盘格角点
                cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
            # time.sleep(3)
            # 显示当前图像
            # cv2.imshow('Chessboard Detection', frame)
            # print(len(obj_points))
            # 如果找到足够的角点，保存图像并退出循环
                cv2.imwrite('E:/code/agent_demo/imgs/captured_gray_image.jpg', frame)
            if len(obj_points) > 30:  # 拍摄足够的图像（这里为20张）
                break
            i+=1
        else:
            i+=1
        # # 按下 'q' 键退出
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()




# 使用收集到的图像点和对象点进行相机内参标定
def calibrate_camera():
    # 假设已经收集了足够的obj_points和img_points
    if len(obj_points) == 0 or len(img_points) == 0:
        print("Error: No points collected.")
        return
    
    # 获取图像的大小
    image_size = (640, 480)  # 根据实际相机图像大小进行设置，或者从采集的图像获取大小
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # image_size = gray.shape[::-1]  # 图像大小（宽，高）

    # 进行相机标定
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)

    if ret:
        print("Camera Matrix (K):")
        print(K)
        print("Distortion Coefficients:")
        print(dist)
        print("Rotation Vectors:")
        print(rvecs)
        print("Translation Vectors:")
        print(tvecs)
    else:
        print("Calibration failed.")

def main():
    import argparse
    import os
    import sys

    # Import the utilities helper module
    import utilities

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        get_image()
        calibrate_camera()


if __name__=="__main__":
    main()
    
    