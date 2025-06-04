from pathlib import Path
from itertools import product
import argparse
import shutil

import bpy
import numpy as np

import prepare
import camera_ops
import load


def render_category(mesh_path: Path, image_path: Path) -> None:
    camera_location = np.array([0.7359, 0.6926, 0.4958])
    material = bpy.data.materials['surface']
    for obj_file in mesh_path.iterdir():
        if obj_file.suffix != '.obj':
            continue
        full_ID = obj_file.stem
        try:
            mesh = load.load_single_mesh(obj_file, material)
        except RuntimeError as e:
            print(e)
            continue
        camera_ops.track_object(mesh)
        for view_index, sign in enumerate(product((1, -1), repeat=3)):
            t = np.array(sign)
            camera_ops.set_viewpoint(tuple(camera_location * t), distance_factor=1.4)
            bpy.context.scene.render.filepath = str(image_path / f'{full_ID}-{view_index}.png')
            bpy.ops.render.render(write_still=True)
            break
        prepare.clear_imported_objects(['surface'])

def main() -> None:
    parser = prepare.get_basic_parser('Render all results saved by test_shapenet.py')
    parser.add_argument(
        '--shading', type=str, choices=['color', 'normal'], default='color',
        help='How to shade the surface, color for PBR shading, normal for normal shading, default: %(default)s'
    )
    config = parser.parse_args()

    result_dir = Path(config.result_dir)
    image_dir = result_dir / 'images'
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir()
    prepare.init_scene((config.size, config.size), render_engine=config.engine)
    prepare.init_lights(prepare.detail_augmentation_lights)
    if config.shading == 'color':
        prepare.create_surface_material('surface', (0.165, 0.564, 0.921))
    else:
        prepare.create_normal_material('surface')

    for directory in result_dir.iterdir():
        if not directory.is_dir() or directory == image_dir:
            continue
        category_ID = directory.stem
        category_images = image_dir / category_ID
        category_images.mkdir()
        render_category(directory, category_images)

if __name__ == '__main__':
    main()
