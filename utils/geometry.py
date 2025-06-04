from typing import Dict, Any, Optional, Literal
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
import open3d as o3d
import torch
from torch import Tensor
from torch import nn

from .logging import get_predefined_logger


# limit the batch size to avoid VRAM overflow
POINTS_PER_PART = 8192 

logger = get_predefined_logger(__name__)

def generate_query_points(pcd: NDArray, K: int, n_query_points: int) -> NDArray:
    n_points = pcd.shape[0]
    KD_tree = cKDTree(pcd)
    # Use the distance to the 50th nearest neighbor point as σ.
    # Query the (K+1)-th nearest neighbor because the nearest neighbor in
    # KD-Tree is always the query point itself.
    sigma: NDArray = KD_tree.query(pcd, [K + 1])[0]
    if sigma.shape[0] > 1:
        sigma = sigma.squeeze()
    rng = np.random.default_rng()
    query_indices = rng.choice(n_points, size=n_query_points)
    query_points: NDArray = pcd[query_indices] \
                 + 0.25 * np.expand_dims(sigma[query_indices], 1) * rng.normal(size=(n_query_points, 3))
    query_points = query_points.astype(np.float32)
    # logger.info(f'{n_query_points} query points are sampled')
    return query_points

def generate_query_points_in_cube(n_query_points: int) -> NDArray:
    rng = np.random.default_rng()
    return rng.uniform(low=-0.5, high=0.5, size=(n_query_points, 3)).astype(np.float32)

def sample_pointcloud(
        mesh: o3d.geometry.TriangleMesh,
        n_points: int,
        K: int,
        n_query_points: int=int(1e5)
    ) -> Dict[str, Any]:
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_points, init_factor=5)

    logger.info('sample query points around the pointcloud')
    points = np.asarray(pcd.points, dtype=np.float32)

    logger.info(f'compute unsigned distance for each query point')
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    return pcd

def find_KNNs(pcd: Tensor, query_points: Tensor, K: int, pcd_squ_norms_T: Optional[Tensor]=None) -> Tensor:
    """
    Find KNN for each query point from the pointcloud.

    :param pcd: The input pointcloud, a tensor with shape of (N, 3).
    :param query_points: The query points, a tensor with shape of (Q, 3).
    :param K: Number of nearest neighbors to find.
    :param pcd_squ_norms_T: Transposed, squared norms of the input point cloud.
        If the input point cloud is constant across many queries, you can
        compute this value outside and pass it to avoid redundant calculations.
        Value: `(pcd ** 2).sum(axis=1, keepdim=True).T`
    :return: KNNs for each query point, a tensor with shape of (Q, K, 3)
        representing K points for each query point. The KNNs' positions are
        transformed to the local shape of corresponding query point, thus the
        position of query point will not affect the shape of KNN.
    """
    inner = -2 * torch.matmul(query_points, pcd.T)
    query_norms_squ = (query_points ** 2).sum(axis=1, keepdim=True)
    if pcd_squ_norms_T is None:
        pcd_squ_norms_T = (pcd ** 2).sum(axis=1, keepdim=True).T
    # dist_matrix[i, j] is distance between query_points[i] and pcd[j]
    dist_matrix = query_norms_squ + inner + pcd_squ_norms_T 
    # select KNN from pcd for query_points
    _, nn_indices = torch.topk(-dist_matrix, K, sorted=True)
    KNNs = pcd[nn_indices] # (Q, K, 3) tensor
    # transform each point in KNN to the local space of query point by a
    # translation (-query_point) and a mirroring (*-1)
    # by doing so, the query point itself is no more required as input because
    # its position does not matter
    #KNNs = torch.tile(query_points.unsqueeze(1), (1, K, 1)) - KNNs
    KNNs = query_points.unsqueeze(1) - KNNs

    return KNNs

def normalize_vectors(vectors: Tensor) -> Tensor:
    """
    Normalize a batch of vectors.

    :param vectors: A tensor reqpresenting a batch of vectors, its shape must be (..., 3).
    :returns: Normalized vectors.
    """
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    return vectors / norms

@torch.no_grad()
def planar_sample(
    SDF_func: nn.Module,
    density: int,
    axis: Literal['x', 'y', 'z']='x',
    position: float=0.0
) -> NDArray:
    """
    Sample a signed distance function on the specified plane.

    :param SDF_func: A PyTorch `nn.Module` that takes query points (a Qx3 tensor)
        and outputs the corresponding signed distance values (a Qx1 tensor)
    :param density: Resolution of sampling along each edge, e.g. passing 256
        means to sample a 257x257 grid on the specified plane. Here we sample
        `density + 1` points because we want to sample on corners of `density`
        grids, so 256 grids have 257 corner points.
    :param axis: Which axis the plane will be vertical to.
    :param position: Position along the specified axis, e.g. passing `axis='x'`
        and `position=0.5` means to sample on the x=0.5 plane.
    :returns: Signed distance values as a (density + 1)x(density + 1) array
    """
    device = next(SDF_func.parameters()).device
    is_training = SDF_func.training
    SDF_func.eval()

    v = density + 1
    dots = np.linspace(-0.5, 0.5, v, dtype=np.float32)
    t1, t2 = np.meshgrid(dots, dots)
    axis_values = np.full_like(t1, position, dtype=np.float32)
    coord = []
    if axis == 'x':
        coord = [axis_values, t1, t2]
    elif axis == 'y':
        coord = [t1, axis_values, t2]
    else:
        coord = [t1, t2, axis_values]
    grid_points = np.dstack(coord).reshape(v * v, 3)

    grid_points = torch.tensor(grid_points, dtype=torch.float32, device=device)
    grid_parts = list(torch.split(grid_points, POINTS_PER_PART))
    SDF = tuple(map(lambda part: SDF_func(part), grid_parts))
    SDF = torch.cat(SDF).reshape(v, v).cpu().numpy()

    if is_training:
        SDF_func.train()
    return SDF

@torch.no_grad()
def cubic_sample(
    SDF_func: nn.Module,
    density: int,
    clip_min: Tensor=torch.tensor([-0.5, -0.5, -0.5]),
    clip_max: Tensor=torch.tensor([0.5, 0.5, 0.5])
) -> NDArray:
    """
    Sample a signed distance function within the unit cube (between -0.5 and 0.5).
    Values on the sampled points outside the bounding box specified by `clip_min`
    and `clip_max` are truncated to 1.

    :param SDF_func: A PyTorch `nn.Module` that takes Q query points (a Qx3 tensor)
        and outputs the corresponding signed distance values (a Qx1 tensor)
    :param density: Resolution of Marching Cubes, e.g. passing 256 to
        sample a 257x257x257 grid. Here we sample `density + 1` points because
        `density` is resolution of Marching Cubes (i.e. grids) instead of points.
    :param clip_min: The lower-left-far corner of the bounding box
    :param clip_max: The upper-right-near corner of the bounding box
    :returns: Signed distance values as a VxVxV tensor, where V is `density + 1`.
    """
    device = next(SDF_func.parameters()).device
    is_training = SDF_func.training
    SDF_func.eval()

    clip_min = clip_min.to(dtype=torch.float32, device=device)
    clip_max = clip_max.to(dtype=torch.float32, device=device)
    def compute_truncated_SDF(points: Tensor) -> Tensor:
        distances = SDF_func(points).squeeze()
        mask = (points < clip_min) | (points > clip_max)
        point_wise_mask = mask.any(dim=1)
        distances[point_wise_mask] = 1.0
        return distances
    v = density + 1
    dots = np.linspace(-0.5, 0.5, v, dtype=np.float32)
    grid_points = torch.tensor(list(product(dots, repeat=3)), dtype=torch.float32, device=device)
    grid_parts = torch.split(grid_points, POINTS_PER_PART)
    SDF = tuple(map(lambda part: compute_truncated_SDF(part), grid_parts))
    SDF = torch.cat(SDF).reshape(v, v, v).cpu().numpy()

    if is_training:
        SDF_func.train()
    return SDF
