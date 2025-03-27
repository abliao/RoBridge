import os
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh

qpos = [
    -0.00302714,
    0.22623112,
    -0.00410317,
    -2.61173252,
    -0.01708191,
    2.89053286,
    0.79847054,
    0.04,
    -0.04,
]
eef_xpos = [-0.05662628, -0.10275355, 0.99407789]
eef_xpos = np.array([-0.5, -0.1, 0.912])
pcd = o3d.geometry.PointCloud()
xyz = np.load("pointcloud.npy")
xyz = xyz[xyz[:, -1] > 0.75]
xyz = xyz[xyz[:, -1] < 1.5]
xyz = xyz[xyz[:, 0] > -0.1]
pcd.points = o3d.utility.Vector3dVector(xyz)

import robosuite
from urdfpy import URDF

robot = URDF.load(
    os.path.join(
        robosuite.__file__[: -len("/__init__.py")],
        "/models/assets/bullet_data/panda_description/urdf/panda_arm_hand.urdf",
    )
)
joints = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]
combined = []

t = time.time()
fk = robot.collision_trimesh_fk(dict(zip(joints, qpos)))
print(time.time() - t)
link_fk = robot.link_fk(dict(zip(joints, qpos)))
mesh_eef_xpos = link_fk[robot.links[0]][:3, 3]
print(time.time() - t)
for mesh, pose in fk.items():
    pose[:3, 3] = pose[:3, 3] + (eef_xpos - mesh_eef_xpos)
    transformed = trimesh.transformations.transform_points(mesh.vertices, pose)
    mesh_new = trimesh.Trimesh(transformed, mesh.faces)
    combined.append(mesh_new)
print(time.time() - t)
combined_mesh = trimesh.util.concatenate(combined)
robot_mesh = combined_mesh.as_open3d
print(time.time() - t)
# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(
    o3d.t.geometry.TriangleMesh.from_legacy(robot_mesh)
)  # we do not need the geometry ID for mesh
signed_distance = scene.compute_occupancy(xyz.astype(np.float32))
filtered_xyz = xyz[signed_distance.numpy() == 0]
# pcd.points = o3d.utility.Vector3dVector(filtered_xyz)
# o3d.visualization.draw_geometries([robot_mesh, pcd])
print(time.time() - t)
