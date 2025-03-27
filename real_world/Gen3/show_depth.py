import cv2
import numpy as np

# 读取 16-bit PNG 深度图
depth_image = cv2.imread("E:\code\Gen3_Demo\imgs\captured_image_depth.png", cv2.IMREAD_UNCHANGED)

# 归一化并调整 Gamma
depth_norm = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
depth_gamma = np.power(depth_norm, 0.5) * 255  # Gamma = 0.5 增强对比度
depth_gamma = depth_gamma.astype(np.uint8)

# 保存并显示
cv2.imwrite("E:\code\Gen3_Demo\imgs\captured_image_depth.png", depth_gamma)
cv2.imshow("Depth Visualization", depth_gamma)
cv2.waitKey(0)
cv2.destroyAllWindows()