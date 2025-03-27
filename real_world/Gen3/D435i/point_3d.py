def create_point_cloud(depth_image, color_image, depth_scale, intrinsics, use_seg=True, seg=None):
    points = []
    colors = []
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            z = depth_image[v, u] * depth_scale  # 将深度值换算成实际的距离
            if z <= 0:  # 仅处理有效的点
                points.append([0, 0, 0])
                colors.append([1.0, 0.0, 0.0])  # 归一化颜色值
                continue
            if depth_image[v, u] > 1.0:
                points.append([0, 0, 0])
                colors.append([1.0, 0.0, 0.0])  # 归一化颜色值
                continue
            if use_seg:
                if seg[v, u] > 0:
                    x = (u - intrinsics['ppx']) * z / intrinsics['fx']
                    y = (v - intrinsics['ppy']) * z / intrinsics['fy']
                    points.append([x, y, z])
                    colors.append(color_image[v, u] / 255.0)  # 归一化颜色值
                else:
                    points.append([0, 0, 0])
                    colors.append([1.0, 0.0, 0.0])  # 归一化颜色值
            else:
                x = (u - intrinsics['ppx']) * z / intrinsics['fx']
                y = (v - intrinsics['ppy']) * z / intrinsics['fy']
                points.append([x, y, z])
                colors.append(color_image[v, u] / 255.0)  # 归一化颜色值
    points = np.array(points)
    colors = np.array(colors)
    return points, colors