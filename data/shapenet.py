from typing import Union, List, Tuple, Optional
from pathlib import Path
import csv
import open3d as o3d
import torch



AVAILABLE_CATEGORIES = {
    'airplane': '02691156',
    'bench': '02828884',
    'cabinet': '02933112',
    'car': '02958343',
    'chair': '03001627',
    'table': '04379243',
    'vessel': '04530566'
}

class ShapeNetCore():
    """
    This dataset class assume the ShapeNetCore.v1 dataset is provided as a set of
    directories, with each category placed in a directory named as the category
    ID. A set of .csv tables should be provided along side the directories, each
    of them contains the metadata of the corresponding category.

    This dataset can be iterated over indices to obtain a tuple (fullId, mesh),
    where fullId is the string ID given in the ShapeNet Core dataset and mesh is
    a open3d TriangleMesh object.

    Because there are some texture images in ShapeNetCore.v1 have wrong
    extensions (e.g. a PNG image has a .jpg extension), the Open3D API
    read_triangle_mesh will fail on loading the textured mesh. So we have to
    remove all texture images from the dataset before reading.

    NOTE: This class does not inherit torch.data.Dataset and should not be used
    by a PyTorch DataLoader.
    """

    def __init__(
            self,
            dataset_path: Union[str, Path],
            category: str,
            train=True
        ):
        """
        :param dataset_path: Path to the ShapeNet Core v1 dataset. The path must
            exist and must be a directory, otherwise it will raise a
            FileNotFoundError.
        :param category: Name of the specified category of None. Only objects in
            the specified category will be loaded.
        :param train: Load training data or testing data.
        """
        self.dataset_path = Path(dataset_path)
        if not (self.dataset_path.exists() and self.dataset_path.is_dir()):
            raise FileNotFoundError(
                "Path to the ShapeNet Core dataset must exist and must be a directory"
            )

        if category not in AVAILABLE_CATEGORIES.keys():
            raise NameError(
                f"Unknown category name: {category}"
            )
        self.category = category
        category_ID: str = AVAILABLE_CATEGORIES[category]
        self.dataset_path = self.dataset_path / category_ID
        self.full_IDs: List[str] = []
        metadata_path = self.dataset_path
        if train:
            metadata_path = metadata_path / 'train.txt'
        else:
            metadata_path = metadata_path / 'test.txt'
        with metadata_path.open('r') as metadata_file:
            lines = metadata_file.readlines()
        self.full_IDs = list(map(lambda line: line.strip(), lines))

    def __len__(self) -> int:
        return len(self.full_IDs)

    def __getitem__(self, index: int) -> Tuple[str, o3d.geometry.TriangleMesh]:
        """
        :return: fullId and a mesh (as an open3d.geometry.TriangleMesh object).
        """
        mesh_path = (self.dataset_path / self.full_IDs[index] / 'model.obj').absolute()
        mesh = o3d.io.read_triangle_mesh(mesh_path.as_posix())
        return self.full_IDs[index], mesh

    def find(self, full_ID) -> o3d.geometry.TriangleMesh:
        """
        This method allows user to randomly access meshes within the category.
        During evaluation we have to iterate over a subset of the category (the test set)
        and find the corresponding ground truth mesh for calculating metrics, so random
        access is useful.
        """
        mesh_path = (self.dataset_path / full_ID / 'model.obj').absolute()
        if not mesh_path.exists():
            raise FileNotFoundError(f'{mesh_path} does not exist')
        mesh = o3d.io.read_triangle_mesh(mesh_path.as_posix())
        return mesh

