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
from utils.geometry import generate_query_points, generate_query_points_in_cube, find_KNNs



logger = get_predefined_logger(__name__)
timestamp = get_timestamp()
LoggingConfig.set_level(logging.DEBUG)
device = torch.device('cpu')
QUERY_POINTS_PER_BATCH = 10000

def validate_category(category: str, config: argparse.Namespace) -> None:
    logger.info(f'start validatation on category {category}')
    category_ID = AVAILABLE_CATEGORIES[category]
    mesh_path = Path(config.shapenet_path) / category_ID
    pcd_path = Path(config.shapenet_pcd_path) / category_ID
    # load the ODF network
    #ODF_func = NearestPointPredictor(config.K).to(device)
    ODF_func = OnSurfaceDecisionFunction(config.K).to(device)
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
    error: List[NDArray] = []
    for full_ID in full_IDs:
        logger.info(f'start processing model {full_ID}')
        # load files
        mesh_file = mesh_path / full_ID / 'model.obj'
        mesh = o3d.io.read_triangle_mesh(str(mesh_file))
        pcd_file = pcd_path / 'test' / f'{full_ID}.ply'
        pcd = o3d.io.read_point_cloud(str(pcd_file))
        # generate query points and KNNs
        pcd = np.asarray(pcd.points, dtype=np.float32)
        #query_points = generate_query_points(pcd, config.K, config.n_query_points)
        query_points = generate_query_points_in_cube(config.n_query_points)
        query_points_gpu = torch.tensor(query_points, device=device, requires_grad=True)
        query_batches = torch.split(query_points_gpu, QUERY_POINTS_PER_BATCH)
        logger.debug(f'{query_points.shape[0]} query points are split into {len(query_batches)} batches')
        pcd = torch.tensor(pcd, device=device)
        pcd_squ_norms_T = (pcd ** 2).sum(axis=1, keepdim=True).T
        knn_batches = tuple(map(lambda q: find_KNNs(pcd, q, config.K, pcd_squ_norms_T), query_batches))
        logger.debug('query points & KNNs generated')
        logger.debug(f'each KNN batch has a shape of {knn_batches[0].shape}')
        # compute the distance to nearest neighbor in the point cloud
        nn_batches = tuple(map(lambda q: find_KNNs(pcd, q, 1, pcd_squ_norms_T).squeeze(), query_batches))
        dist_to_nn = torch.linalg.vector_norm(torch.cat(nn_batches), dim=1).detach().cpu().numpy()
        dist.append(dist_to_nn)
        logger.debug('distance from query points to point cloud computed')
        logger.debug(f'first three dist: {dist_to_nn[:3]}')
        logger.debug(f'average dist: {np.mean(dist_to_nn)}')
        # compute GT
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        gt = scene.compute_closest_points(
            o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
        )['points'].numpy()
        gt = torch.tensor(gt, device=device)
        logger.debug('NNs on mesh computed')
        #predictions = tuple(map(lambda knn: ODF_func(knn), knn_batches)) # Nearest Neighbor
        #predictions = torch.cat(predictions)
        #predictions = query_points_gpu - predictions
        #prediction_error = euclidean_dist(predictions, gt)

        #gt = torch.linalg.vector_norm(query_points_gpu - gt, dim=1)
        predictions = tuple(map(lambda knn: ODF_func(knn).squeeze(), knn_batches)) # On-Surface Decision
        predictions = torch.cat(predictions)
        directions = torch.autograd.grad(torch.sum(predictions), query_points_gpu, retain_graph=True)[0]
        directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
        predictions = query_points_gpu - predictions.unsqueeze(1) * directions
        #prediction_error = torch.abs(predictions - gt)
        prediction_error = euclidean_dist(predictions, gt)

        #predictions = torch.cat(nn_batches) # Neural-Pull
        #predictions = query_points_gpu - predictions
        #prediction_error = euclidean_dist(predictions, gt)

        error.append(prediction_error.detach().cpu().numpy())
        logger.debug(f'first three errors: {prediction_error[:3]}')
        logger.debug(f'average error: {torch.mean(prediction_error)}')
        logger.info(f'done')
    # save the results
    dist = np.concatenate(dist)
    error = np.concatenate(error)
    result_path = Path('results/error-dist') / f'{category_ID}.npz'
    np.savez(result_path, dist=dist, error=error)
    logger.info('dist & error saved')

def main():
    parser = argparse.ArgumentParser(description='Sampling query points on each category, saving distance and prediction error')
    parser.add_argument('--prior-model', help='Path to the pre-trained prior network weight file')
    parser.add_argument('-N', '--n-query-points', type=int, default=int(1e5), help='Number of query points sampled from an object')
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
    if config.n_query_points % QUERY_POINTS_PER_BATCH != 0:
        raise RuntimeError(f'Argument -N (--n-query-points) must be divisible by {QUERY_POINTS_PER_BATCH}')
    if config.gpu is not None:
        device = torch.device(f'cuda:{config.gpu}')
    LoggingConfig.set_file(f'log/validations/{timestamp}.log')

    for category in AVAILABLE_CATEGORIES.keys():
        validate_category(category, config)

if __name__ == '__main__':
    main()
