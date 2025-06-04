from typing import Tuple, Union
import sqlite3
from pathlib import Path
from io import BytesIO
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

import utils.database


class PriorDataset(Dataset):
    """
    Dataset used for training and testing the prior network.

    This class wraps an SQLite database, which contains 2 tables: query_points_train
    and query_points_test. Each row of the database table contains the query point
    itself, together with its KNN sampled from the ground truth point cloud.
    The query_points_train table is shuffled, query points from different
    objects (and in different categories) are shuffled randomly. The query_points_test
    is ordered so query points from the same object are placed sequentially.
    """
    def __init__(self, database_path: Union[str, Path], train: bool, K: int=100) -> None:
        """
        Initialize an prior database.

        :param database_path: Path to the SQLite database file.
        :param train: Load training data or testing data.
        :param K: Number of points in each KNN.
        """
        self.database_path = Path(database_path)
        if not self.database_path.exists():
            raise FileNotFoundError('The database file does not exist')

        self.db = sqlite3.connect(
            self.database_path.as_posix(),
            detect_types=sqlite3.PARSE_COLNAMES
        )
        self.db.row_factory = sqlite3.Row
        self.cursor = self.db.cursor()
        if train:
            self.length = self.cursor.execute(
                'SELECT max(id) FROM query_points_train'
            ).fetchone()[0]
            self.query_points_table = 'query_points_train'
        else:
            self.length = self.cursor.execute(
                'SELECT max(id) FROM query_points_test'
            ).fetchone()[0]
            self.query_points_table = 'query_points_test'
        self.K = K
    
    def __del__(self):
        """
        Close the database connection during destruction.
        """
        self.cursor.close()
        self.db.close()

    def __len__(self) -> int:
        """
        :returns: Number of query points available in the database table.
        """
        return self.length
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Fetch a query point's KNN and GT nearest neighbor from database.

        :returns: KNN as a Kx3 tensor, nearest neighbor as a (3,) tensor.
        """
        # result = self.cursor.execute(
        #     f'SELECT knn as "knn [NDArray]", label FROM {self.query_points_table} WHERE id = ?',
        #     (index + 1,)
        # ).fetchone()
        result = self.cursor.execute(
            f'SELECT query_point as "query_point [NDArray]", knn as "knn [NDArray]", nearest_point as "nearest_point [NDArray]" FROM {self.query_points_table} WHERE id = ?',
            (index + 1,)
        ).fetchone()
        KNN = torch.tensor(result['knn'][0:self.K, :])
        # label = result['label']
        # return KNN, label
        nearest_point = torch.tensor(result['query_point'] - result['nearest_point'])
        return KNN, nearest_point
