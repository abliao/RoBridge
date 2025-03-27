import pyrealsense2 as rs
import json
import numpy as np

WIDTH=640
HEIGHT=480
# 相机配置
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 60)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, 60)

profile = pipeline.start(config)
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
# 获取相机内参
intr = color_frame.profile.as_video_stream_profile().intrinsics
camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                     'ppx': intr.ppx, 'ppy': intr.ppy,
                     'height': intr.height, 'width': intr.width,
                     'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                     }
# 保存内参到本地
with open('./intrinsics.json', 'w') as fp:
    json.dump(camera_parameters, fp)
# 图像对齐
align_to = rs.stream.color
align = rs.align(align_to)

frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)

aligned_depth_frame = aligned_frames.get_depth_frame()
# 深度参数，像素坐标系转相机坐标系用到
depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
color_frame = aligned_frames.get_color_frame()

# 深度图
d = np.asanyarray(aligned_depth_frame.get_data())
# 彩色图
image_np = np.asanyarray(color_frame.get_data())
# 输入像素的x和y计算真实距离
x = 320
y = 240
dis = aligned_depth_frame.get_distance(x, y)

camera_coordinate = rs.rs2_deproject_pixel_to_point(intrin=depth_intrin, pixel=[x, y], depth=dis)
print(camera_coordinate)