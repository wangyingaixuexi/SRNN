from typing import List, Tuple, Literal
from pathlib import Path
from itertools import product
from collections import OrderedDict
import argparse
import copy
import logging

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
import open3d as o3d
import torch
from torch import Tensor
import mcubes
import tqdm
import matplotlib
import matplotlib.pyplot as plt

from model.sdf import SignedDistancePredictor
from model.prior import NearestPointPredictor
from utils.geometry import generate_query_points, find_KNNs, planar_sample, cubic_sample
from reconstruct import show_slice, extract_surface


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
# Number of points per part when the whole tensor (points) is too large to be processed at once.
POINTS_PER_PART = 8192
def enable_tensorboard_logging():
    from torch.utils.tensorboard import SummaryWriter
    global writer
    writer = SummaryWriter(log_dir='log/reconstruction')

def reconstruct(
    pcd: NDArray,
    config: argparse.Namespace
) -> Tuple[o3d.geometry.TriangleMesh, NDArray, NDArray]:
    """
    Optimize the distance function (network) to fit the input point cloud.

    :param pcd: N×3 array representing N points.
    :param config: Command-line arguments.
    :returns: An extracted iso-surface (as mesh), the loss values during optimization,
        and the rotation matrix.
    """
    device = config.device
    prior = NearestPointPredictor(config.K).to(device)
    state_dict = torch.load(config.prior_model, map_location=device)
    prior.load_state_dict(state_dict)
    prior.eval()

    SDF_func = SignedDistancePredictor(
        radius=0.5, init_alpha=config.init_alpha, max_batches=config.max_batches, n_dim_hidden=256
    ).to(device)
    SDF_func.train()
    optimizer = torch.optim.Adam(SDF_func.parameters(), lr=config.learning_rate)
    euclidean_dist = torch.nn.PairwiseDistance(p=2, eps=0).to(device)
    rng = np.random.default_rng()
    rotation = Rotation.from_euler('ZYX', rng.uniform(low=0.0, high=180.0, size=(3,)), degrees=True)
    pcd = rotation.apply(pcd).astype(np.float32)
    while abs(pcd.max()) > 1 or abs(pcd.min()) > 1:
        rotation = Rotation.from_euler('ZYX', rng.uniform(low=0.0, high=180.0, size=(3,)), degrees=True)
        pcd = rotation.apply(pcd).astype(np.float32)

    # Sample all query points at once, faster than sampling query points within each batch.
    # The total number of query points must be enough, e.g. more than 10^6. Less sampling will lead
    # to very poor reconstruction quality.
    all_query_points = generate_query_points(pcd, config.K, int(5e6))
    pcd_tensor = torch.tensor(pcd, device=device)
    pcd_squ_norms_T = (pcd_tensor ** 2).sum(axis=1, keepdim=True).T.detach().requires_grad_(False)

    # For saving the best state of the distance function (network).
    loss_values = np.zeros((config.max_batches,), dtype=np.float32)
    lowest_loss = 10.0
    best_state = OrderedDict()

    generator = range(config.max_batches)
    if config.progress_bar:
        generator = tqdm.tqdm(generator, desc='Reconstructing', unit='iter')
    for batch_no in generator:
        # Randomly select some query points
        indices = rng.choice(all_query_points.shape[0], config.n_query_points, replace=False)
        query_points = torch.tensor(all_query_points[indices], device=device, requires_grad=True)
        query_points.retain_grad()
        query_points_KNNs = find_KNNs(pcd_tensor, query_points, config.K, pcd_squ_norms_T)

        optimizer.zero_grad()

        SDF = SDF_func(query_points).clone()
        # Use PyTorch's auto grad to calculate the gradient of SDF with
        # respect to each query point.
        sum_SDF = torch.sum(SDF) # Make SDF a scalar because we cannot diff a tensor directly
        projection_dir = torch.autograd.grad(sum_SDF, query_points, retain_graph=True)[0]
        norms = torch.linalg.vector_norm(projection_dir, dim=1, keepdim=True)
        projection_dir = projection_dir / norms
        # Project query points according to SDF and directions
        projected_points = query_points - SDF * projection_dir

        # The Nearest Neighbor Prior predicts the projection targets
        NNs = prior(query_points_KNNs)
        loss_proj = torch.mean(euclidean_dist(SDF * projection_dir, NNs))
        # Eikonal regularization loss
        loss_reg = torch.mean(torch.abs(norms - 1))
        # No special meaning, just to make the two losses at the same order of magnitude.
        loss_total = 100 * loss_proj + 1 * loss_reg
        loss_total.backward()
        optimizer.step()

        if config.tensorboard:
            writer.add_scalars(
                'loss',
                {
                    'Projection (100×)': 100 * loss_proj,
                    'Eikonal': loss_reg,
                    'Total': loss_total
                },
                batch_no
            )
            writer.add_scalar('alpha', SDF_func.alpha, batch_no)
            if (batch_no + 1) % 500 == 0:
                slice_x = show_slice(SDF_func, config, 'x')
                slice_y = show_slice(SDF_func, config, 'y')
                slice_z = show_slice(SDF_func, config, 'z')
                writer.add_figure('slices/x', slice_x, batch_no)
                writer.add_figure('slices/y', slice_y, batch_no)
                writer.add_figure('slices/z', slice_z, batch_no)
        if config.save_progress and (batch_no + 1) % config.batches_per_save == 0:
            mesh = extract_surface(pcd, SDF_func, config.iso_value, config)
            index = (batch_no + 1) // config.batches_per_save
            o3d.io.write_triangle_mesh(f'results/progress/{index:03}.obj', mesh)
        if (batch_no + 1) % 500 == 0 and loss_total.item() < lowest_loss:
            best_state = copy.deepcopy(SDF_func.state_dict())
            lowest_loss = loss_total.item()
        loss_values[batch_no] = loss_total.item()

    SDF_func.load_state_dict(best_state)
    mesh = extract_surface(pcd, SDF_func, config.iso_value, config)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    vertices = rotation.apply(vertices, inverse=True)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh, loss_values, rotation.as_matrix()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', '--prior-model', required=True, help='The pre-trained prior network weight file')
    parser.add_argument('-K', type=int, default=100, help='Number of points in KNN')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('-A', '--init-alpha', type=float, default=0.3, help='Initial alpha value for controlling weights of PE components')
    parser.add_argument('--max-batches', type=int, default=30000, help='Number of batches to optimize the distance function')
    parser.add_argument('-Q', '--n-query-points', type=int, default=4096, help='Number of query points per batch')
    parser.add_argument('-V', '--n-voxels', type=int, default=256, help='Precision of SDF grid')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index')
    parser.add_argument('--iso-value', type=float, default=0.003, help='The iso-surface to extract from the distance function')
    parser.add_argument('--tensorboard', action='store_true', help='Log loss values to a tensorboard file')
    parser.add_argument('--save-progress', action='store_true', help='Extract and save surface at regular intervals during optimization')
    parser.add_argument('--batches-per-save', default=1000, type=int, help='The interval for saving intermediate surfaces, valid when being used with option --save-progress')
    parser.add_argument('--progress-bar', action='store_true', help='Show a progress bar indicating the optimization progress')
    parser.add_argument('ply_file', help='The input point cloud (must be PLY format)')
    parser.add_argument('mesh_file', help='Filename to save the reconstructed mesh (can be any format supported by Open3D)')

    config = parser.parse_args()
    ply_file_path = Path(config.ply_file).absolute()
    if not ply_file_path.exists():
        raise FileNotFoundError(f'The PLY file {ply_file_path} does not exist')
    save_path = Path(config.mesh_file).absolute()
    if not save_path.parent.exists():
        raise FileNotFoundError(f'Directory {save_path.parent} does not exist')
    loss_save_path = save_path.with_suffix('.npy')
    rotation_save_path = save_path.parent / f'{save_path.stem}-rotation.npy'
    if config.tensorboard:
        enable_tensorboard_logging()
    config.device = torch.device(f'cuda:{config.gpu}')

    pcd = o3d.io.read_point_cloud(ply_file_path.as_posix())
    pcd = np.asarray(pcd.points, dtype=np.float32)
    mesh, loss_values, rotation = reconstruct(pcd, config)
    o3d.io.write_triangle_mesh(save_path.as_posix(), mesh)
    np.save(loss_save_path, loss_values)
    np.save(rotation_save_path, rotation)

    if config.tensorboard:
        writer.flush()
        writer.close()
    logging.shutdown()

if __name__ == '__main__':
    main()
