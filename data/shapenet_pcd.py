from typing import Union, Tuple, Optional, NamedTuple
from pathlib import Path
import csv
import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import torch
from torch import Tensor

from .shapenet import AVAILABLE_CATEGORIES


class ShapeNetCorePointcloud():
    """
    This dataset loads pointclouds uniformly sampled from meshes in
    ShapeNetCore.v1. The point cloud dataset is more simple than the original
    mesh dataset, with each category stored as a directory containing some
    .ply files. Each PLY file is named as its object ID ('fullId' in the
    ShapeNetCore metadata files).

    NOTE: This class does not inherit torch.data.Dataset, and should not be used
    by a PyTorch DataLoader.
    """

    def __init__(
            self,
            pcd_dataset_path: Union[str, Path],
            category: str,
            shuffle=False,
            train=False
        ) -> None:
        """
        :param pcd_dataset_path: Path to the sampled pointclouds, transformation
            parameters and query points.
        :param category: Name of the specified category of None. If a name is
            passed, only objects in the specified category will be loaded. If
            None, all available categories will be loaded.
        :param shuffle: Whether to shuffle the data items.
        """
        self.pcd_dataset_path = Path(pcd_dataset_path)
        if not (self.pcd_dataset_path.exists() and self.pcd_dataset_path.is_dir()):
            raise FileNotFoundError(
                "Path to the pointcloud dataset must exist and must be a directory"
            )

        if category not in AVAILABLE_CATEGORIES.keys():
            raise NameError(
                f"Unknown category name: {category}"
            )
        category_ID = AVAILABLE_CATEGORIES[category]
        self.pcd_dataset_path = self.pcd_dataset_path / category_ID
        self.pcd_dataset_path /= ('train' if train else 'test')

        self.full_IDs: List[str] = []
        for dirname in self.pcd_dataset_path.iterdir():
            full_ID = dirname.stem
            self.full_IDs.append(full_ID)
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(self.full_IDs)

    def __len__(self) -> int:
        return len(self.full_IDs)

    def __getitem__(
            self, index: int
        ) -> Tuple[str, Tensor]:
        """
        :return: An object's full_ID and its point cloud representation (as a Nx3 tensor).
        """
        full_ID = self.full_IDs[index]
        pcd_path = self.pcd_dataset_path / f'{full_ID}.ply'

        pcd = o3d.io.read_point_cloud(pcd_path.as_posix())
        pcd = np.asarray(pcd.points, dtype=np.float32)

        return (full_ID, torch.tensor(pcd))
