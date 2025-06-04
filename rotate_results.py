from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from dataset.shapenet import AVAILABLE_CATEGORIES


result_path = Path('results/ablation-rotation')
for category_name, category_id in AVAILABLE_CATEGORIES.items():
    category_path = result_path / category_id
    for file in category_path.iterdir():
        #if file.stem.find('rotation') == -1:
        #    continue
        #print(f'load {file.stem}')
        #rotation_matrix = np.load(file)
        #rotation = Rotation.from_matrix(rotation_matrix)
        #object_id = file.stem.split('-')[0]
        #object_file = category_path / f'{object_id}.obj'
        #mesh = o3d.io.read_triangle_mesh(str(object_file))
        #vertices = np.asarray(mesh.vertices)
        #vertices = rotation.apply(vertices)
        #mesh.vertices = o3d.utility.Vector3dVector(vertices)
        #target_file = category_path / f'{object_id}-rotated.obj'
        #o3d.io.write_triangle_mesh(str(target_file), mesh)
        #print(f'{target_file.stem} saved')
        if file.suffix != '.obj':
            continue
        if file.stem.find('rotated') != -1:
            continue
        print(f'load {file.stem}')
        mesh = o3d.io.read_triangle_mesh(str(file))
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(file), mesh)
        print(f'{file.stem} saved')
