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
from model.prior import NearestPointPredictor, OnSurfaceDecisionFunction
from utils.logging import get_predefined_logger, get_timestamp, LoggingConfig
from utils.geometry import find_KNNs, normalize_vectors



logger = get_predefined_logger(__name__)
timestamp = get_timestamp()
LoggingConfig.set_level(logging.DEBUG)
device = torch.device('cpu')
NORMAL_OFFSET_LIMIT = 0.001
TANGENT_OFFSET_LIMIT = 1e-3

@torch.no_grad
def validate_category(category: str, config: argparse.Namespace) -> None:
    logger.info(f'start validation on category {category}')
    category_ID = AVAILABLE_CATEGORIES[category]
    mesh_path = Path(config.shapenet_path) / category_ID
    pcd_path = Path(config.shapenet_pcd_path) / category_ID

    ODF_func = NearestPointPredictor(config.K).to(device)
    #ODF_func = OnSurfaceDecisionFunction(config.K).to(device)
    state_dict = torch.load(config.odf_model, map_location=device)
    ODF_func.load_state_dict(state_dict)
    ODF_func.eval()
    # the error metric
    euclidean_dist = torch.nn.PairwiseDistance(p=2, eps=0).to(device)
    # traverse the category
    with (pcd_path / 'subtest.txt').open('r') as f:
        lines = f.readlines()
        full_IDs: List[str] = list(map(lambda line: line.strip(), lines))
    dist: List[NDArray] = []
    tangent_offset: List[NDArray] = []
    error: List[NDArray] = []
    for full_ID in full_IDs:
        logger.info(f'start processing model {full_ID}')
        # load files
        mesh_file = mesh_path / full_ID / 'model.obj'
        mesh = o3d.io.read_triangle_mesh(str(mesh_file))
        mesh.compute_triangle_normals()
        pcd_file = pcd_path / 'test' / f'{full_ID}.ply'
        pcd = o3d.io.read_point_cloud(str(pcd_file))
        # generate query points by moving sampled points along the normal direction sllightly
        pcd = np.asarray(pcd.points, dtype=np.float32)
        sampled_points = mesh.sample_points_uniformly(
            number_of_points=config.n_query_points // 2, use_triangle_normal=True
        )
        # query points on the surface
        query_points = np.asarray(sampled_points.points, dtype=np.float32)
        query_points = np.concatenate((query_points, query_points), axis=0)
        query_points = torch.tensor(query_points, device=device)
        # normal vectors from bi-directional surface normals
        query_normals = np.asarray(sampled_points.normals, dtype=np.float32)
        query_normals = np.concatenate((query_normals, -query_normals), axis=0)
        query_normals = torch.tensor(query_normals, device=device)
        # move the query points along the normals
        normal_offset_length = NORMAL_OFFSET_LIMIT * torch.rand((config.n_query_points,), device=device).unsqueeze(1)
        query_points = query_points + normal_offset_length * query_normals
        # build a local framework for each query point, t1 and t2 are unit tangent vectors
        ref_vector_x = torch.tensor([1, 0, 0], dtype=torch.float32, device=device).reshape((1, 3))
        ref_vector_y = torch.tensor([0, 1, 0], dtype=torch.float32, device=device).reshape((1, 3))
        is_parallel_x = torch.abs(torch.sum(query_normals * ref_vector_x, dim=1)) > 0.9
        ref_vectors = torch.where(is_parallel_x.unsqueeze(1), ref_vector_y, ref_vector_x)
        query_t2 = torch.linalg.cross(query_normals, ref_vectors)
        query_t1 = torch.linalg.cross(query_t2, query_normals)
        # compute GT
        pcd = torch.tensor(pcd, device=device)
        pcd_squ_norms_T = (pcd ** 2).sum(axis=1, keepdim=True).T
        nn_in_pcd = find_KNNs(pcd, query_points, 1, pcd_squ_norms_T).squeeze()
        dist_to_nn = torch.linalg.vector_norm(nn_in_pcd, dim=1).cpu().numpy()
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        gt = scene.compute_closest_points(
            o3d.core.Tensor(query_points.cpu().numpy(), dtype=o3d.core.Dtype.Float32)
        )['points'].numpy()
        gt = torch.tensor(gt, device=device)
        # prediction
        average_error = torch.zeros((config.n_query_points,), dtype=torch.float32, device=device)
        for _ in range(config.n_offset_points):
            offset_angle = 2.0 * np.pi * torch.rand((config.n_query_points,), device=device)
            offset_direction = torch.cos(offset_angle).unsqueeze(1) * query_t1 + torch.sin(offset_angle).unsqueeze(1) * query_t2
            offset_length = TANGENT_OFFSET_LIMIT * torch.rand((config.n_query_points,), device=device).unsqueeze(1)
            offset_queries = query_points + offset_length * offset_direction
            offset_gt = gt + offset_length * offset_direction

            #predictions = offset_queries - find_KNNs(pcd, offset_queries, 1, pcd_squ_norms_T).squeeze()
            #prediction_error = euclidean_dist(predictions, offset_gt)

            #offset_queries = torch.tensor(offset_queries, device=device, requires_grad=True)
            #offset_knns = find_KNNs(pcd, offset_queries, config.K, pcd_squ_norms_T)
            #predictions = ODF_func(offset_knns).squeeze()
            #directions = torch.autograd.grad(torch.sum(predictions), offset_queries, retain_graph=True)[0]
            #directions = normalize_vectors(directions)
            #predictions = offset_queries - predictions.unsqueeze(1) * directions
            #prediction_error = euclidean_dist(predictions, offset_gt).detach()

            offset_knns = find_KNNs(pcd, offset_queries, config.K, pcd_squ_norms_T)
            predictions = offset_queries - ODF_func(offset_knns)
            prediction_error = euclidean_dist(predictions, offset_gt)

            average_error += prediction_error
        average_error /= config.n_offset_points

        dist.append(dist_to_nn)
        error.append(average_error.cpu().numpy())
        logger.info(f'average tangent error: {error[-1].mean()}')
        logger.info('done')

    dist = np.concatenate(dist)
    error = np.concatenate(error)
    result_path = Path('results/tangent-error') / f'{category_ID}.npz'
    np.savez(result_path, dist=dist, error=error)
    logger.info(f'results of category {category} saved')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior-model', help='Path to the pre-trained prior network weight file')
    parser.add_argument('-N', '--n-query-points', type=int, default=int(1e4), help='Number of query points in total')
    parser.add_argument('-O', '--n-offset-points', type=int, default=10, help='Number of offset points per query point')
    parser.add_argument('-K', type=int, default=100, help='Number of points in KNN')
    parser.add_argument('--gpu', default=None, help='CUDA device index')
    parser.add_argument(
        'shapenet_path',
        help='Path to the ShapeNetCore.v1 dataset'
    )
    parser.add_argument(
        'shapenet_pcd_path',
        help='Path to point cloud data sampled from the ShapeNetCore dataset'
    )
    config = parser.parse_args()
    if config.gpu is not None:
        device = torch.device(f'cuda:{config.gpu}')
    LoggingConfig.set_file(f'log/validations/{timestamp}.log')

    for category in AVAILABLE_CATEGORIES.keys():
        validate_category(category, config)

if __name__ == '__main__':
    main()
