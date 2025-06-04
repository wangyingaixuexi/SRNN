from pathlib import Path
from itertools import product
import argparse
import shutil

import bpy
from bpy.types import GeometryNodeTree

import numpy as np

import prepare
import camera_ops
import load



def render_category(pcd_path: Path, image_path: Path, modifier: GeometryNodeTree) -> None:
    camera_location = np.array([0.7359, 0.6926, 0.4958])
    material = bpy.data.materials['point cloud']
    for ply_file in pcd_path.iterdir():
        if ply_file.suffix != '.ply':
            continue
        full_ID = ply_file.stem
        try:
            pcd = load.load_point_cloud(ply_file)
        except RuntimeError as e:
            print(e)
            continue
        load.apply_modifier(pcd, modifier)
        camera_ops.track_object(pcd)
        for view_index, sign in enumerate(product((1, -1), repeat=2)):
            t = np.array([sign[0], sign[1], 1])
            camera_ops.set_viewpoint(tuple(camera_location * t), distance_factor=1.4)
            bpy.context.scene.render.filepath = str(image_path / f'{full_ID}-{view_index}.png')
            bpy.ops.render.render(write_still=True)
            break
        prepare.clear_imported_objects(['point cloud'])

def main() -> None:
    parser = argparse.ArgumentParser(description='Visualizing Normal Consistency (NC) of reconstructed surfaces')
    parser.add_argument('-r', '--radius', type=float, default=0.001, help='Radius of each point')
    parser.add_argument('pcd_dir', type=str, help='Path to the point cloud results')
    parser.add_argument('image_dir', type=str, help='Directory for rendered images')
    config = parser.parse_args()

    pcd_dir = Path(config.pcd_dir)
    image_dir = Path(config.image_dir)
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir()
    prepare.init_scene((1024, 1024), render_engine='eevee')
    prepare.init_lights(prepare.detail_augmentation_lights, energy_factor=2.0)
    bpy.context.scene.eevee.use_shadows = False
    material = prepare.create_point_color_material('point cloud')
    modifier = prepare.create_pointcloud_modifier(material, radius=config.radius, subdivisions=1)

    for directory in pcd_dir.iterdir():
        if not directory.is_dir() or directory == image_dir:
            continue
        category_ID = directory.stem
        category_images = image_dir / category_ID
        category_images.mkdir()
        render_category(directory, category_images, modifier)

if __name__ == '__main__':
    main()
