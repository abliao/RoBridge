import pyrealsense2 as rs
import json
import numpy as np
import open3d as o3d

class Camera():
    """
    Camera class
    Need to be initialized with the width, height, and fps of the camera
    align_frames() should be called every frame !!!
    """
    def __init__(self, WIDTH, HEIGHT, fps, device_index=0):
        # 获取连接的设备
        context = rs.context()
        devices = context.query_devices()
        # 选择要使用的设备索引
        selected_device_serial = devices[device_index].get_info(rs.camera_info.serial_number)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(selected_device_serial)
        self.config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, fps)
        self.profile = self.pipeline.start(self.config)


        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.aligned_frames = None
        self.align_frames()
        color_intrinsics = self.intrinsics
        print("Color Camera Intrinsics:")
        print(f"fx: {color_intrinsics.fx}, fy: {color_intrinsics.fy}")
        print(f"ppx: {color_intrinsics.ppx}, ppy: {color_intrinsics.ppy}")
        print(f"width: {color_intrinsics.width}, height: {color_intrinsics.height}")

    
    def check_frames(self):
        return self.aligned_frames is not None

    # update for every frame
    def align_frames(self):
        frames = self.pipeline.wait_for_frames()
        self.aligned_frames = self.align.process(frames)
        return self.aligned_frames

    def get_rgb_frame(self):
        color_frame = self.aligned_frames.get_color_frame()
        image_np = np.asanyarray(color_frame.get_data())
        return image_np
    
    def get_depth_frame(self):
        depth_frame = self.aligned_frames.get_depth_frame()
        depth = np.asanyarray(depth_frame.get_data())
        return depth

    def pixel_to_world(self, u, v, dis):
        return rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, pixel = [u, v], depth = dis)
    

    def get_point_cloud(self, visualize=False, use_seg=True, seg=None):
        rgb_image = self.get_rgb_frame()  # 获取 RGB 图
        depth_image = self.get_depth_frame()  # 获取深度图
        intrinsics = self.depth_intrinsics  # 相机内参
        depth_scale = self.depth_scale  # 深度缩放比例

        # 获取图像尺寸
        height, width = depth_image.shape

        # 生成像素坐标网格
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # 读取深度信息并转换为实际物理单位（单位：米）
        z = depth_image * depth_scale
        valid_mask = (z > 0) & (z < 5.0)  # 过滤无效深度

        # 计算 3D 坐标
        x = (u - intrinsics.ppx) * z / intrinsics.fx
        y = (v - intrinsics.ppy) * z / intrinsics.fy

        # 仅保留有效的点
        x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

        # 处理颜色
        rgb_image = rgb_image.astype(np.float32) / 255.0  # 归一化
        colors = rgb_image.reshape(-1, 3)[valid_mask.ravel()]  # 仅取有效像素的颜色

        # 组合点云
        points = np.stack((x, y, z), axis=-1)

        # Open3D 可视化
        if visualize:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")

        return points, colors

    def get_pixel_to_ee(self,u,v):
        self.align_frames()
        rgb_image = self.get_rgb_frame()  # 获取 RGB 图
        depth_image = self.get_depth_frame()  # 获取深度图
        intrinsics = self.depth_intrinsics  # 相机内参
        depth_scale = self.depth_scale  # 深度缩放比例

        # 获取图像尺寸
        height, width = depth_image.shape

        # 读取深度信息并转换为实际物理单位（单位：米）
        print('depth',depth_image[v,u], depth_scale,(depth_image * depth_scale).max(),(depth_image * depth_scale).min())
        z = depth_image[v,u] * depth_scale
        valid_mask = (z > 0) & (z < 5.0)  # 过滤无效深度

        # 计算 3D 坐标
        x = (u - intrinsics.ppx) * z / intrinsics.fx
        y = (v - intrinsics.ppy) * z / intrinsics.fy

        return x,y,z

    
    @property
    def intrinsics(self):
        color_frame = self.pipeline.wait_for_frames().get_color_frame()
        return color_frame.profile.as_video_stream_profile().intrinsics
    
    @property
    def depth_intrinsics(self):
        alighed_frames = self.align.process(self.pipeline.wait_for_frames())
        depth_frame = alighed_frames.get_depth_frame()
        return depth_frame.profile.as_video_stream_profile().intrinsics
    @property
    def depth_scale(self):
        return self.profile.get_device().first_depth_sensor().get_depth_scale()
