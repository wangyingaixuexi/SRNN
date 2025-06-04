from typing import List, Tuple, Literal
from pathlib import Path
from itertools import product
from collections import OrderedDict
import argparse
import copy
import logging

import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import torch
from torch import Tensor
import mcubes
import tqdm
import matplotlib
import matplotlib.pyplot as plt

from model.sdf import SignedDistancePredictor
from model.prior import OnSurfaceDecisionFunction
from utils.geometry import generate_query_points, find_KNNs, planar_sample, cubic_sample


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

@torch.no_grad()
def show_slice(
    SDF_func: SignedDistancePredictor, config: argparse.Namespace,
    axis: Literal['x', 'y', 'z']='x', position: float=0.0
) -> matplotlib.figure.Figure:
    """
    Slice the distance function with a plane vertical to an axis.

    :param SDF_func: The distance function (network)
    :param config: Command-line arguments
    :param axis: Which axis the slice plane will be vertical to
    :param position: Position of the slice plane along the specified axis
    :returns: The contour plot of the slice
    """
    SDF = planar_sample(SDF_func, config.n_voxels, axis, position)
    v = config.n_voxels + 1
    dots = np.linspace(-0.5, 0.5, v)
    t1, t2 = np.meshgrid(dots, dots)

    fig = plt.figure()
    ax = fig.add_subplot()
    color_norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)

    contour_set = ax.contourf(t1, t2, SDF, levels=np.linspace(-1, 1, 30), cmap='coolwarm')
    ax.contour(t1, t2, SDF, levels=[config.iso_value], colors='k')
    fig.colorbar(contour_set, ax=ax)
    fig.tight_layout()
    return fig

@torch.no_grad()
def extract_surface(
    pcd: NDArray, SDF_func: SignedDistancePredictor, iso_value: float, config: argparse.Namespace
) -> o3d.geometry.TriangleMesh:
    """
    Extract iso-surface from the distance function.

    :param pcd: N×3 array representing N points.
    :param SDF_func: The distance function (network).
    :param iso_value: Specify which iso-surface you want to extract.
    :param config: Command-line arguments.
    :returns: A triangle mesh.
    """
    min_point = torch.tensor(pcd.min(axis=0) - 0.02)
    max_point = torch.tensor(pcd.max(axis=0) + 0.02)
    SDF = cubic_sample(SDF_func, config.n_voxels, min_point, max_point)

    # Perform MC, scale grid points into a unit cube
    vertices, triangles = mcubes.marching_cubes(SDF, iso_value)
    vertices /= config.n_voxels
    vertices -= 0.5
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh.compute_vertex_normals()
    return mesh

def reconstruct(
    pcd: NDArray,
    config: argparse.Namespace
) -> Tuple[o3d.geometry.TriangleMesh, NDArray]:
    """
    Optimize the distance function (network) to fit the input point cloud.

    :param pcd: N×3 array representing N points.
    :param config: Command-line arguments.
    :returns: An extracted iso-surface (as mesh) and loss values during optimization.
    """
    device = config.device
    prior = OnSurfaceDecisionFunction(config.K).to(device)
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

    # Sample all query points at once, faster than sampling query points within each batch.
    # The total number of query points must be enough, e.g. more than 10^6. Less sampling will lead
    # to very poor reconstruction quality.
    all_query_points = generate_query_points(pcd, config.K, int(5e6))
    pcd_tensor = torch.tensor(pcd, device=device)
    pcd_squ_norms_T = (pcd_tensor ** 2).sum(axis=1, keepdim=True).T.detach().requires_grad_(False)

    # For saving the best state of the distance function (network).
    loss_values = np.zeros((config.max_batches,), dtype=np.float32)
    lowest_loss = 1.0
    best_state = OrderedDict()

    generator = range(config.max_batches)
    if config.progress_bar:
        generator = tqdm.tqdm(generator, desc='Reconstructing', unit='iter')
    for batch_no in generator:
        # Randomly select some query points
        indices = rng.choice(all_query_points.shape[0], config.n_query_points, replace=False)
        query_points = torch.tensor(all_query_points[indices], device=device, requires_grad=True)
        query_points.retain_grad()

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
        projected_KNNs = find_KNNs(pcd_tensor, projected_points, config.K, pcd_squ_norms_T)

        dist = prior(projected_KNNs)
        loss_odf = torch.mean(dist)
        loss_reg = torch.mean(torch.abs(SDF))
        # No special meaning, just to make the two losses at the same order of magnitude.
        loss_total = loss_odf + 0.2 * loss_reg
        loss_total.backward()
        optimizer.step()

        if config.tensorboard:
            writer.add_scalars(
                'loss',
                {
                    'ODF': loss_odf,
                    'Reg': loss_reg,
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
    return mesh, loss_values

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
    if config.tensorboard:
        enable_tensorboard_logging()
    config.device = torch.device(f'cuda:{config.gpu}')

    pcd = o3d.io.read_point_cloud(ply_file_path.as_posix())
    pcd = np.asarray(pcd.points, dtype=np.float32)
    mesh, loss_values = reconstruct(pcd, config)
    o3d.io.write_triangle_mesh(save_path.as_posix(), mesh)
    np.save(loss_save_path, loss_values)

    if config.tensorboard:
        writer.flush()
        writer.close()
    logging.shutdown()

if __name__ == '__main__':
    main()
