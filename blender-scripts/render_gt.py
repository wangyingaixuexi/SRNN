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
    with (mesh_path / 'subtest.txt').open('r') as f:
        full_IDs = f.readlines()
    full_IDs = [line.strip() for line in full_IDs]
    for full_ID in full_IDs:
        obj_file = mesh_path / full_ID / 'model.obj'
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
        prepare.clear_imported_objects(['surface'])

def main() -> None:
    parser = argparse.ArgumentParser(description='Render GT meshes')
    parser.add_argument(
        '--shading', type=str, choices=['color', 'normal', 'orientation'], default='color',
        help='How to shade the surface, color for PBR shading, normal for normal shading, orientation for orientation indication'
    )
    parser.add_argument('gt_dir', type=str, help='Path to the ground truth mesh dataset')
    parser.add_argument('image_dir', type=str, help='Directory for rendered images')
    config = parser.parse_args()
    gt_dir = Path(config.gt_dir)
    image_dir = Path(config.image_dir)
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir()
    prepare.init_scene((640, 640), render_engine='eevee')
    prepare.init_lights(prepare.detail_augmentation_lights)
    if config.shading == 'color':
        prepare.create_surface_material('surface', (0.794, 0.489, 0.243))
    elif config.shading == 'normal':
        prepare.create_normal_material('surface')
    elif config.shading == 'orientation':
        prepare.create_orientation_material('surface')
        bpy.context.scene.eevee.use_shadows = False
    else:
        raise RuntimeError(f'the shading parameter {config.shading} is not supported')

    for directory in gt_dir.iterdir():
        if not directory.is_dir():
            continue
        category_ID = directory.stem
        category_images = image_dir / category_ID
        category_images.mkdir()
        render_category(directory, category_images)

if __name__ == '__main__':
    main()
