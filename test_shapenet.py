from typing import List
from pathlib import Path
from subprocess import run
from multiprocessing import Queue, Process
import logging
import time
import argparse
import os

import numpy as np
from numpy.typing import NDArray

from dataset.shapenet import AVAILABLE_CATEGORIES
from utils.logging import get_predefined_logger, get_timestamp, LoggingConfig



timestamp = get_timestamp()
logs = Queue()
END_MESSAGE = 'all reconstruction tasks finished'

def work(
    gpu_index: str, config: argparse.Namespace, other_args: List,
    tasks: Queue, logs: Queue, save_path: Path
) -> None:
    """
    Fetch reconstruction tasks (point cloud files) from the task queue and
    execute reconstruct.py to reconstruct surfaces.

    :param gpu_index: CUDA devices index
    :param config: Command-line arguments used for this test script
    :param other_args: Other command-line arguments passed to reconstruct.py
    :param tasks: The task queue containing point cloud files (as Path objects)
    :param logs: The logging message queue
    :param save_path: Directory to save reconstructed meshes
    :raises CalledProcessError: If any reconstruction task failed with an Exception
    """
    while not tasks.empty():
        pcd_path = tasks.get()
        full_ID = pcd_path.stem
        logs.put((logging.INFO, f'start reconstructing object {full_ID}'))
        #cmd = ['python3', 'reconstruct.py'] + other_args + [
        cmd = ['python3', 'ablation/rotation.py'] + other_args + [
            '--gpu', gpu_index,
            str(pcd_path),
            str(save_path / f'{full_ID}.obj')
        ]
        #run(cmd, check=True)
        env = os.environ.copy()
        env['PYTHONPATH'] = '.'
        run(cmd, env=env, check=True)
        logs.put((logging.INFO, f'{full_ID} has been reconstructed'))

def print_logs(log_queue: Queue) -> None:
    """
    Collect logging messages from other processes and print them to screen & file.
    Put `(logging.INFO, END_MESSAGE)` into the queue to terminate this process.

    :param logs: The logging message queue
    """
    logger = get_predefined_logger(__name__)
    LoggingConfig.set_file(Path(f'log/reconstruction/{timestamp}.log'))
    LoggingConfig.set_level(logging.DEBUG)
    while True:
        while not log_queue.empty():
            level, msg = log_queue.get()
            logger.log(level, msg)
            if msg == END_MESSAGE:
                logging.shutdown()
                return
        time.sleep(5)

def reconstruct_category(category: str, config: argparse.Namespace, other_args: List) -> None:
    """
    Preforming reconstruction on a whole category in ShapeNetCore.

    :param category: Category name
    :param config: Command-line arguments for this test script
    :param other_args: Other command-line arguments passed to reconstruct.py
    """
    logs.put((logging.INFO, f'start reconstructing category {category}'))
    category_ID = AVAILABLE_CATEGORIES[category]
    pcd_dataset_path = Path(config.shapenet_pcd_path) / category_ID / 'test'
    full_IDs: List[str] = []
    with open(pcd_dataset_path.parent / f'{config.name}.txt', 'r') as f:
        full_IDs = f.readlines()
    full_IDs = [full_ID.strip() for full_ID in full_IDs]
    rng = np.random.default_rng()
    rng.shuffle(full_IDs)
    save_path = Path(f'results/{timestamp}/{category_ID}')
    save_path.mkdir(parents=True)
    tasks = Queue()
    for i in range(len(full_IDs)):
        tasks.put(pcd_dataset_path / f'{full_IDs[i]}.ply')
    gpu_indices = config.gpu.split(',')
    
    workers = []
    for index in gpu_indices:
        for i in range(config.tasks_per_gpu):
            worker = Process(target=work, args=(index, config, other_args, tasks, logs, save_path))
            worker.start()
            workers.append(worker)
    for worker in workers:
        worker.join()
    logs.put((logging.INFO, 'done'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='Name of the subset')
    parser.add_argument('--gpu', default='0', help='Comma-separated list of CUDA device indices')
    parser.add_argument('--tasks-per-gpu', type=int, default=1, help='Number of tasks running on each GPU')
    parser.add_argument(
        'shapenet_pcd_path',
        help='Path to point cloud data sampled from the ShapeNetCore dataset'
    )
    config, other_args = parser.parse_known_args()

    logger_process = Process(target=print_logs, args=(logs, ))
    logger_process.start()

    try:
        logs.put((logging.INFO, 'test script configurations:'))
        for key, value in vars(config).items():
            logs.put((logging.INFO, f'{key}={value}'))
        logs.put((logging.INFO, 'reconstruction configurations:'))
        logs.put((logging.INFO, str(other_args)))

        for category in AVAILABLE_CATEGORIES.keys():
            reconstruct_category(category, config, other_args)
    finally:
        logs.put((logging.INFO, END_MESSAGE))
        logger_process.join()

if __name__ == '__main__':
    main()
