from pathlib import Path
import logging
import argparse
import sqlite3
from io import BytesIO
import pdb

import open3d as o3d
import numpy as np
from numpy.typing import NDArray
import torch
from tqdm import tqdm, trange

from dataset.shapenet import ShapeNetCore, AVAILABLE_CATEGORIES
from utils.geometry import generate_query_points, find_KNNs
from utils.logging import get_predefined_logger, LoggingConfig
import utils.database



logger = get_predefined_logger(__name__)
LoggingConfig.set_level(logging.INFO)

next_train_id = 0
next_test_id = 0


def sample_prior_category(
    dataset: ShapeNetCore,
    db: sqlite3.Connection,
    config: argparse.Namespace,
    title: str,
    train: bool
) -> None:
    """
    Sample (query point, KNN, nearest neighbor) on one category.

    :param dataset: The ShapeNetCore dataset object
    :param db: SQLite database connection object
    :param config: Command-line arguments
    :param title: Title of the progress bar
    :param train: True for sampling training set or False for sampling test set
    """
    global next_train_id, next_test_id
    cursor = db.cursor()
    query_points_table = 'query_points_{}'.format('train' if train else 'test')
    for i in trange(len(dataset), desc=title, unit='object'):
        full_ID, mesh = dataset[i]
        logger.info(f'sample from object {full_ID}')
        pcd = mesh.sample_points_poisson_disk(number_of_points=config.n_pcd_points, init_factor=5)
        pcd = np.asarray(pcd.points, dtype=np.float32)

        query_points = generate_query_points(pcd, config.K, config.n_query_points)
        KNNs = find_KNNs(torch.tensor(pcd), torch.tensor(query_points), config.K).numpy()
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        result = scene.compute_closest_points(
            o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
        )
        nearest_points = result['points'].numpy()
        for j in range(config.n_query_points):
            cursor.execute(
                f'INSERT INTO {query_points_table} (query_point, knn, nearest_point) VALUES (?, ?, ?)',
                (query_points[j], KNNs[j], nearest_points[j])
            )

        next_id = next_train_id if train else next_test_id
        db.commit()
        logger.info('all results has been written into the database')
        if train:
            next_train_id += config.n_query_points
        else:
            next_test_id += config.n_query_points
    cursor.close()

def shuffle_prior_data(db_tmp: sqlite3.Connection, db_target: sqlite3.Connection) -> None:
    """
    Shuffle training set of the prior network by randomly reading the sampled
    database and re-inserting items into another database.

    :param db_tmp: The database sampled by `sample_prior_category`
    :param db_target: The final database
    """
    cursor_read = db_tmp.cursor()
    cursor_write = db_target.cursor()

    logger.info(f'start shuffling training data')
    n_items = cursor_read.execute(
        'SELECT count(id) FROM query_points_train'
    ).fetchone()[0]
    logger.info(f'training data contains {n_items} pairs of KNN and unsigned distance')
    indices = np.arange(1, n_items + 1) # SQLite's PK starts from 1, not 0
    rng = np.random.default_rng()
    rng.shuffle(indices)
    logger.info('indices shuffled')
    commit_counter = 0
    for index in tqdm(indices, desc='Shuffling', unit='rows'):
        query_point, KNN, nearest_point = cursor_read.execute(
            'SELECT query_point, knn, nearest_point FROM query_points_train WHERE id = ?',
            (int(index), )
        ).fetchone()
        cursor_write.execute(
            'INSERT INTO query_points_train (query_point, knn, nearest_point) VALUES (?, ?, ?)',
            (query_point, KNN, nearest_point)
        )
        commit_counter += 1
        if commit_counter == 1000:
            db_target.commit()
            commit_counter = 0
    db_target.commit()
    commit_counter = 0
    logger.info('done')

    logger.info(f'start copying test data')
    cursor_read.execute(
        'INSERT INTO target_db.query_points_test SELECT * FROM main.query_points_test'
    )
    db_tmp.commit()
    logger.info('done')

    cursor_read.close()
    cursor_write.close()

def sample_reconstruction_category(
    dataset: ShapeNetCore, config: argparse.Namespace, category_path: Path, train: bool
) -> None:
    """
    Sample pointclouds uniformly on the surface.

    :param dataset: The ShapeNetCore datase object
    :param config: Command-line arguments
    :param category_path: Path to save pointclouds in this category
    :param train: True for sampling training set or False for sampling test set
    """
    title = f'Sampling {dataset.category} (reconstruction)'
    full_IDs: List[str] = []
    for i in trange(len(dataset), desc=title, unit='object'):
        full_ID, mesh = dataset[i]
        full_IDs.append(full_ID)
        pcd_path = category_path / f'{full_ID}.ply'
        if pcd_path.exists():
            logger.warning(f'sampling of object {full_ID} already exists, skip it')
            continue
        logger.info(f'sample from object {full_ID}')
        pcd = mesh.sample_points_poisson_disk(number_of_points=config.n_pcd_points, init_factor=5)

        o3d.io.write_point_cloud(pcd_path.as_posix(), pcd)
        logger.info(f'point cloud has been saved to {pcd_path.name}')

def sample_prior_data(source_path: Path, target_path: Path, config: argparse.Namespace):
    tmp_database_path = target_path / f'prior-{config.n_pcd_points}-{config.K}-tmp.db'
    database_path = target_path / f'prior-{config.n_pcd_points}-{config.K}.db'
    if database_path.exists():
        if database_path.is_file():
            logger.warning(f'The {database_path} already exists, rename it')
            database_path.rename(database_path.parent / f'{database_path.name}.bak')
            # database_path.unlink()
        else:
            raise RuntimeError(f'The {database_path} already exists and is not a file')
    db_tmp = sqlite3.connect(tmp_database_path)
    db_target = sqlite3.connect(database_path)
    try:
        with open(Path(__file__).parent / 'dataset/on_surface_metadata.sql', 'r') as f:
            schema = f.read()
            if not config.shuffle_only:
                db_tmp.executescript(schema)
                db_tmp.commit()
            db_target.executescript(schema)
            db_target.commit()
        if not config.shuffle_only:
            for category in AVAILABLE_CATEGORIES.keys():
                category_ID = AVAILABLE_CATEGORIES[category]
                logger.info(f'start sampling from {category} ({category_ID})')
                train_set = ShapeNetCore(source_path, category=category, train=True)
                test_set = ShapeNetCore(source_path, category=category, train=False)

                sample_prior_category(train_set, db_tmp, config, f'Sampling {category} (train)', True)
                sample_prior_category(test_set, db_tmp, config, f'Sampling {category} (test)', False)
        db_tmp.execute(f'ATTACH DATABASE \'{str(database_path)}\' AS target_db')
        shuffle_prior_data(db_tmp, db_target)
        db_tmp.execute('DETACH DATABASE \'target_db\'')
    finally:
        db_tmp.close()
        db_target.close()

def sample_reconstruction_data(source_path: Path, target_path: Path, config):
    for category in AVAILABLE_CATEGORIES.keys():
        category_ID = AVAILABLE_CATEGORIES[category]
        category_path = target_path / category_ID
        category_path.mkdir(mode=0o755, exist_ok=True)
        logger.info(f'start sampling from {category} ({category_ID})')
        train_set = ShapeNetCore(source_path, category=category, train=True)
        test_set = ShapeNetCore(source_path, category=category, train=False)

        sample_reconstruction_category(train_set, config, category_path / 'train', True)
        sample_reconstruction_category(test_set, config, category_path / 'test', False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--n_pcd_points', type=int, default=2048, help='Number of points in a pointcloud')
    parser.add_argument('-Q', '--n_query_points', type=int, default=2048, help='Number of query points per object, only useful when sampling training data for the prior network')
    parser.add_argument('-K', type=int, default=100, help='Number of points in KNN')
    parser.add_argument('-p', '--prior', action='store_true', help='Sample training data for prior network')
    parser.add_argument(
        '--shuffle-only', action='store_true', default=False,
        help='Only shuffle the training data for prior network (sampled data required, useful if a previous sampling task exited with an error)'
    )
    parser.add_argument('-r', '--reconstruction', action='store_true', help='Sample pointclouds for surface reconstruction')
    parser.add_argument(
        'shapenet_path',
        help='Path to the ShapeNetCore dataset (with all texture images removed)'
    )
    parser.add_argument(
        'shapenet_pcd_path',
        help='Path to point cloud data sampled from the ShapeNetCore dataset'
    )
    config = parser.parse_args()
    LoggingConfig.set_file(Path('log/sampling.log').absolute())

    logger.info('configurations:')
    for key, value in vars(config).items():
        logger.info(f'{key}={value}')
    if config.prior or config.reconstruction:
        # here we disable the output to stdout because we want to see prograss bar
        # instead of log messages on the console. The log messages are in the log file.
        LoggingConfig.disable_console_output()
    source_path = Path(config.shapenet_path).absolute()
    target_path = Path(config.shapenet_pcd_path).absolute()
    target_path.mkdir(mode=0o755, parents=True, exist_ok=True)
    if config.prior:
        sample_prior_data(source_path, target_path, config)
    if config.reconstruction:
        sample_reconstruction_data(source_path, target_path, config)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        raise e
    finally:
        logging.shutdown()
