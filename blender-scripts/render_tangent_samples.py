import argparse
from pathlib import Path

import bpy
import numpy as np

import prepare
import camera_ops
import load


def main() -> None:
    parser = argparse.ArgumentParser(description='Render a mesh with some disks sampled for tangent error metric')
    parser.add_argument('gt_file', type=str, help='Path to the GT mesh file')
    parser.add_argument('query_file', type=str, help='Path to the query points file')
    config = parser.parse_args()

    prepare.init_scene((2048, 2048), render_engine='eevee')
    prepare.init_lights(prepare.detail_augmentation_lights, energy_factor=2.0)
    prepare.create_transparent_material('surface', (0.794, 0.489, 0.243, 0.3))
    material = prepare.create_surface_material('point cloud', (0.165, 0.564, 0.921))
    modifier = prepare.create_pointcloud_modifier(material, radius=0.001, subdivisions=2)

    pcd = load.load_point_cloud(config.query_file)
    load.apply_modifier(pcd, modifier)
    material = bpy.data.materials['surface']
    mesh = load.load_single_mesh(config.gt_file, material)
    camera_location = np.array([0.7359, 0.6926, 0.4958])
    camera_ops.track_object(mesh)
    camera_ops.set_viewpoint(tuple(camera_location), distance_factor=1.4)
    bpy.context.scene.render.filepath = 'query.png'
    bpy.ops.render.render(write_still=True)

if __name__ == '__main__':
    main()
