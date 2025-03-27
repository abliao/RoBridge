import cv2
import numpy as np
import os
import sys

# cap = cv2.VideoCapture(0)

# while(True):
#     ret, frame = cap.read()

#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

def get_image():
    src = "rtsp://admin:admin@192.168.1.10/color"
    # cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
    # 读取一帧
        ret, frame = cap.read()
        
        # 检查帧是否正确读取
        if not ret or frame is None:
            print("Error: Failed to capture image.")
            break

        # 显示帧
        cv2.imshow('Video', frame)
        cv2.imwrite('./captured_image.jpg', frame)
        
        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


def main():
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        get_image()


if __name__ == "__main__":
    main()