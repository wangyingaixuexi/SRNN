from typing import List, Dict
import logging
from pathlib import Path
import argparse
from csv import DictWriter
import shutil

import numpy as np
from numpy.typing import NDArray
import torch
import open3d as o3d
from tqdm import tqdm
import matplotlib as mpl

from dataset.shapenet import ShapeNetCore, AVAILABLE_CATEGORIES
from dataset.shapenet_pcd import ShapeNetCorePointcloud
from utils.chamfer_distance import ChamferDistance
from utils.f_score import fscore
from utils.logging import get_predefined_logger, get_timestamp, LoggingConfig



logger = get_predefined_logger(__name__)
LoggingConfig.set_level(logging.INFO)
cmap = mpl.colormaps['coolwarm']

@torch.no_grad()
def evaluate_category(category: str, config: argparse.Namespace) -> List[Dict[str, float]]:
    """
    Evaluate metrics on one category.

    :param category: Category name
    :param config: Command-line arguments
    :returns: Metrics of each object (as a dict)
    """
    calc_CD = ChamferDistance().to('cuda')
    calc_CD.eval()
    category_ID = AVAILABLE_CATEGORIES[category]
    mesh_dataset = ShapeNetCore(config.shapenet_path, category)
    results_path = Path(config.results_path) / category_ID
    result_files = list(results_path.glob('*.obj'))
    result_files.sort()
    n_results = len(result_files)
    if config.nc_vis_path is not None:
        vis_path = Path(config.nc_vis_path) / category_ID
        if vis_path.exists():
            shutil.rmtree(vis_path)
        vis_path.mkdir()
    metrics = []
    for i, result_file in enumerate(tqdm(result_files, desc='Evaluating', unit='objects')):
        full_ID = result_file.stem
        # Load the reconstructed surface and the GT surface, sample `config.n_points` points from each mesh
        mesh_res: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(result_file.as_posix())
        pcd_res = mesh_res.sample_points_uniformly(number_of_points=config.n_points)
        pcd_res = np.asarray(pcd_res.points, dtype=np.float32)
        pcd_res = torch.tensor(pcd_res, device='cuda')
        mesh_GT = mesh_dataset.find(full_ID)
        pcd_GT = mesh_GT.sample_points_uniformly(number_of_points=config.n_points)
        pcd_GT = np.asarray(pcd_GT.points, dtype=np.float32)
        pcd_GT = torch.tensor(pcd_GT, device='cuda')

        # Compute CD and F1-Score
        dist1, dist2 = calc_CD(pcd_GT.unsqueeze(0), pcd_res.unsqueeze(0))
        f_score, _, _ = fscore(dist1, dist2)
        f_score = f_score.squeeze().item()
        dist1, dist2 = torch.sqrt(dist1), torch.sqrt(dist2)
        CD = (torch.mean(dist1) + torch.mean(dist2)) / 2
        CD = CD.item()

        # Sample `4*config.n_points` points from the reconstructed surface with corresponding face
        # normals as point normals
        mesh_res.compute_triangle_normals()
        pcd_res = mesh_res.sample_points_uniformly(number_of_points=4*config.n_points, use_triangle_normal=True)
        # Query the closest points on the GT surface and get face normals on the GT mesh
        query_points = o3d.core.Tensor(np.asarray(pcd_res.points, dtype=np.float32))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_GT))
        query_result = scene.compute_closest_points(query_points)
        # Do not use primitive_normals provided by Open3D because because some of them has NaN components
        indices = query_result['primitive_ids'].numpy()
        mesh_GT.compute_triangle_normals()
        all_GT_normals = np.asarray(mesh_GT.triangle_normals, dtype=np.float32)
        normals_GT = all_GT_normals[indices]
        normals_res = np.asarray(pcd_res.normals, dtype=np.float32)
        # Use bi-directional normals for the GT mesh, i.e. the dot product is always positive
        normal_consistency = np.abs((normals_GT * normals_res).sum(axis=1))
        if config.nc_vis_path is not None:
            nc_colors = cmap(1.0 - normal_consistency)
            pcd_res.colors = o3d.utility.Vector3dVector(nc_colors[:, :3]) # drop the alpha channel
            pcd_res.normals.clear() # we do not need point normals when visualizing NC
            vis_pcd_path = Path(config.nc_vis_path) / category_ID / f'{full_ID}.ply'
            o3d.io.write_point_cloud(vis_pcd_path.as_posix(), pcd_res)
        normal_consistency = normal_consistency.mean()

        metrics.append(
            {
                'full ID': full_ID,
                'CD': CD,
                'F-Score': f_score,
                'NC': normal_consistency
            }
        )

        logger.info(f'{category} [{i + 1}/{n_results}] {result_file.stem}, CD: {CD:.5f}, F-Score: {f_score:.5f}')

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--n-points', type=int, default=300000, help='Number of sampling points when calculating CD')
    parser.add_argument(
        '-vis', '--nc-vis-path',
        help='Path to save point clouds visualizing normal consistency',
        default=None
    )
    parser.add_argument(
        'shapenet_path',
        help='Path to the ShapeNetCore.v1 dataset'
    )
    parser.add_argument(
        'results_path',
        help='Path to the reconstruction results'
    )
    config = parser.parse_args()
    results_path = Path(config.results_path)
    LoggingConfig.set_file(Path(f'log/evaluation-{results_path.stem}.log'))
    LoggingConfig.disable_console_output()
    
    if config.nc_vis_path is not None:
        vis_path = Path(config.nc_vis_path)
        if vis_path.exists():
            shutil.rmtree(vis_path)
        vis_path.mkdir()

    avg_rows = []
    for category in AVAILABLE_CATEGORIES.keys():
        metrics = evaluate_category(category, config)
        metrics_path = results_path / AVAILABLE_CATEGORIES[category] / 'metrics.csv'
        with open(metrics_path, 'w', newline='') as metrics_file:
            writer = DictWriter(metrics_file, metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)
        avg_CD = sum([row['CD'] for row in metrics]) / len(metrics)
        avg_f_score = sum([row['F-Score'] for row in metrics]) / len(metrics)
        avg_NC = sum([row['NC'] for row in metrics]) / len(metrics)
        logger.info(f'category {category}:')
        logger.info(f'average CD: {avg_CD}')
        logger.info(f'average F-Score: {avg_f_score}')
        logger.info(f'average NC: {avg_NC}')
        avg_rows.append(
            {
                'category ID': AVAILABLE_CATEGORIES[category],
                'category name': category,
                'CD': avg_CD,
                'F-Score': avg_f_score,
                'NC': avg_NC
            }
        )
    stat_path = Path(config.results_path) / 'stats.csv'
    with open(stat_path, 'w', newline='') as stat_file:
        writer = DictWriter(stat_file, avg_rows[0].keys())
        writer.writeheader()
        writer.writerows(avg_rows)

if __name__ == '__main__':
    main()
