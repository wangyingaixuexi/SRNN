from typing import List
from pathlib import Path
import logging
import argparse

import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import torch
from torch import Tensor

from dataset.shapenet import AVAILABLE_CATEGORIES
from utils.geometry import find_KNNs, normalize_vectors


NORMAL_OFFSET_LIMIT = 0.01
TANGENT_OFFSET_LIMIT = 1e-2
N_QUERY_POINTS = 200
N_OFFSET_POINTS = 400

parser = argparse.ArgumentParser()
parser.add_argument('mesh_file', type=str, help='GT mesh')
parser.add_argument('pcd_file', type=str, help='point cloud sampled from the GT mesh')
config = parser.parse_args()

mesh_file = Path(config.mesh_file)
if not mesh_file.exists():
    raise FileNotFoundError(f'file {mesh_file} does not exist')
mesh = o3d.io.read_triangle_mesh(str(mesh_file))
mesh.compute_triangle_normals()
pcd_file = Path(config.pcd_file)
if not pcd_file.exists():
    raise FileNotFoundError(f'file {pcd_file} does not exist')
pcd = o3d.io.read_point_cloud(str(pcd_file))

sampled_points = mesh.sample_points_uniformly(
    number_of_points=N_QUERY_POINTS // 2, use_triangle_normal=True
)
# query points on the surface
query_points = np.asarray(sampled_points.points, dtype=np.float32)
query_points = np.concatenate((query_points, query_points), axis=0)
query_points = torch.tensor(query_points)
# normal vectors from bi-directional surface normals
query_normals = np.asarray(sampled_points.normals, dtype=np.float32)
query_normals = np.concatenate((query_normals, -query_normals), axis=0)
query_normals = torch.tensor(query_normals)
# move the query points along the normals
normal_offset_length = NORMAL_OFFSET_LIMIT * torch.rand((N_QUERY_POINTS,)).unsqueeze(1)
query_points = query_points + normal_offset_length * query_normals
# build a local framework for each query point, t1 and t2 are unit tangent vectors
ref_vector_x = torch.tensor([1, 0, 0], dtype=torch.float32).reshape((1, 3))
ref_vector_y = torch.tensor([0, 1, 0], dtype=torch.float32).reshape((1, 3))
is_parallel_x = torch.abs(torch.sum(query_normals * ref_vector_x, dim=1)) > 0.9
ref_vectors = torch.where(is_parallel_x.unsqueeze(1), ref_vector_y, ref_vector_x)
query_t2 = torch.linalg.cross(query_normals, ref_vectors)
query_t1 = torch.linalg.cross(query_t2, query_normals)

all_offset_queries = []
for _ in range(N_OFFSET_POINTS):
    offset_angle = 2.0 * np.pi * torch.rand((N_QUERY_POINTS,))
    offset_direction = torch.cos(offset_angle).unsqueeze(1) * query_t1 + torch.sin(offset_angle).unsqueeze(1) * query_t2
    offset_length = TANGENT_OFFSET_LIMIT * torch.rand((N_QUERY_POINTS,)).unsqueeze(1)
    offset_queries = query_points + offset_length * offset_direction
    all_offset_queries.append(offset_queries)
all_offset_queries = torch.cat(all_offset_queries, dim=0)

all_offset_queries = o3d.utility.Vector3dVector(all_offset_queries.numpy())
query_pcd = o3d.geometry.PointCloud(all_offset_queries)
o3d.io.write_point_cloud('query.ply', query_pcd)
