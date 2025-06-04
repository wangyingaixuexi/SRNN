from pathlib import Path
from typing import List
import logging
import argparse

import numpy as np
import torch

from shapenet import AVAILABLE_CATEGORIES
from shapenet_pcd import ShapeNetCorePointcloud
from utils.chamfer_distance import ChamferDistance
from utils.logging import get_predefined_logger, get_timestamp, LoggingConfig


calc_CD = ChamferDistance().cuda()

logger = get_predefined_logger(__name__)
timestamp = get_timestamp()
LoggingConfig.set_level(logging.INFO)

def sample_category(testset: ShapeNetCorePointcloud, object_per_category: int) -> List[str]:
    full_IDs = []
    pcds = []
    n_pcds = len(testset)
    logger.info(f'{n_pcds} objects found in the test set')
    for full_ID, pcd in testset:
        full_IDs.append(full_ID)
        pcds.append(pcd.cuda())

    dist = np.zeros((n_pcds, n_pcds), dtype=np.float32)
    for i, pcd in enumerate(pcds):
        dist[i, i] = 0
        for j in range(i + 1, n_pcds):
            dist1, dist2 = calc_CD(pcd.unsqueeze(0), pcds[j].unsqueeze(0))
            dist[i, j] = (torch.mean(dist1) + torch.mean(dist2)).item()
            dist[j, i] = dist[i, j]
    logger.info(f'pair wise distance matrix obtained, average CD: {np.average(dist)}')

    dist_to_selection = np.sum(dist, axis=1)
    selected_indices = []
    for i in range(object_per_category):
        selected_index = np.argmax(dist_to_selection)
        logger.info(f'selected object {i + 1}: {full_IDs[i]}, distance: {dist_to_selection[selected_index]}')
        selected_indices.append(selected_index)
        dist_to_selection = np.minimum(dist_to_selection, dist[selected_index])
    selected_IDs = [full_IDs[i] for i in selected_indices]
    return selected_IDs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='Name of the subset')
    parser.add_argument('-N', '--n-objects', type=int, default=100, help='Number of objects per category')
    parser.add_argument('shapenet_pcd_path', help='Path to the ShapeNetCore pointcloud dataset')
    config = parser.parse_args()
    LoggingConfig.set_file(f'log/subset-{config.name}.log')

    pcd_dataset_path = Path(config.shapenet_pcd_path)
    for category in AVAILABLE_CATEGORIES:
        logger.info(f'start sampling from category {category}')
        subset_list = pcd_dataset_path / AVAILABLE_CATEGORIES[category] / f'{config.name}.txt'
        logger.info(f'write the full ID list to {subset_list}')
        testset = ShapeNetCorePointcloud(pcd_dataset_path, category, shuffle=False, train=False)
        selected_IDs = sample_category(testset)
        with subset_list.open('w') as f:
            f.writelines([full_ID + '\n' for full_ID in selected_IDs])
        logger.info('done')
    logging.shutdown()

if __name__ == '__main__':
    main()
