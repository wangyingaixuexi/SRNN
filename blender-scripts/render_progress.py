import argparse
import shutil
from pathlib import Path
from itertools import product

import bpy
import numpy as np

import prepare
import camera_ops
import load


def main() -> None:
    parser = prepare.get_basic_parser(
        """
        Render the mesh sequence saved by reconstruct.py, the rendered images
        will be saved in *images* subdirectory in the result-dir
        """
    )
    config = parser.parse_args()
    print(config)

    result_dir = Path(config.result_dir)
    image_dir = result_dir / 'images'
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir()
    prepare.init_scene((config.size, config.size), render_engine=config.engine)
    prepare.init_lights(prepare.detail_augmentation_lights)
    prepare.create_surface_material('surface', (0.165, 0.564, 0.921))

    camera_location = np.array([0.7359, 0.6926, 0.4958])
    camera_ops.set_viewpoint(tuple(camera_location), distance_factor=1.8)
    material = bpy.data.materials['surface']
    for obj_file in result_dir.iterdir():
        if obj_file.is_dir():
            continue
        try:
            mesh = load.load_single_mesh(obj_file, material)
            camera_ops.track_object(mesh)
        except RuntimeError as e:
            print(e)
        index = obj_file.stem
        bpy.context.scene.render.filepath = str(image_dir / f'{index}.png')
        bpy.ops.render.render(write_still=True)
        prepare.clear_imported_objects(['surface'])

if __name__ == '__main__':
    main()
