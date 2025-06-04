# built-in packages
import argparse
from typing import List, Tuple, Dict, Any, Literal
# blender packages
import bpy
from bpy.types import (
    Scene, Material, Object, GeometryNodeTree
)
from mathutils import Vector, Euler
# third-party packages
import numpy as np


# all parameters are obtained through blender GUI
# the unit of angle is radians as blender API default setting
shadow_augmentation_lights = [
    {
        'name': 'sun light 1', 'type': 'SUN',
        'location': np.array([3.638, 1.674, 4.329]), 'energy': 5.0, 'angle': 0.199
    },
    {
        'name': 'sun light 2', 'type': 'SUN',
        'location': np.array([0.449, -3.534, 1.797]), 'energy': 1.83, 'angle': 0.009
    },
    {
        'name': 'point light 1', 'type': 'POINT',
        'location': np.array([-2.163, -0.381, -2.685]), 'energy': 500
    }
]
detail_augmentation_lights = [
    {
        'name': 'sun', 'type': 'SUN',
        'energy': 15.0, 'angle': 0.105,
        'location': np.array([-0.562, -1.031, 0.636]), 'rotation': Euler((-1.016, 0.58, 2.972), 'XYZ')
    },
    {
        'name': 'area top', 'type': 'AREA',
        'energy': 20.0, 'size': 1.0,
        'location': np.array([0, 0, 1.396]), 'rotation': Euler((0, 0, 0), 'XYZ')
    },
    {
        'name': 'area front', 'type': 'AREA',
        'energy': 8.0, 'size': 1.0,
        'location': np.array([0, 1.228, 0]), 'rotation': Euler((-1.571, 0, 0), 'XYZ')
    }
]

def get_basic_parser(description: str) -> argparse.ArgumentParser:
    """
    Create a basic command-line argument parser. The following arguments are
    added:
    - `--size`: image size
    - `--engine`: Blender render engine
    - `result-dir`: path to the results
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--size', type=int, default=640,
        help='Size of the rendered image (in px), default: %(default)s'
    )
    parser.add_argument(
        '--engine', type=str, choices=['eevee', 'cycles'], default='cycles',
        help='The render engine in Blender, by default %(default)s will be used'
    )
    parser.add_argument('result_dir', type=str, help='The directory storing reconstructed mesh')
    return parser

def init_scene(image_resolution: Tuple[int, int], render_engine: Literal['eevee', 'cycles']) -> None:
    """
    Initialize a scene with following configurations:

    - Color mode: RGBA (8 bit for each channel)
    - File format: PNG
    - Transparent background
    - One camera, no other objects

    :param image_resolution: Image resolution (in pixel) represented as a tuple (width, height).
    :param render_engine: The render engine to use, can be eevee or cycles.
    """
    # the bpy.context module is usually read-only, so we access the current scene through bpy.data
    scene_name: str = bpy.context.scene.name
    scene: Scene = bpy.data.scenes[scene_name]
    if render_engine == 'eevee':
        scene.render.engine = 'BLENDER_EEVEE'
    else:
        scene.render.engine = 'CYCLES'
        preferences = bpy.context.preferences.addons['cycles'].preferences
        preferences.get_devices()
        # Enable CUDA acceleration if available
        if len(preferences.get_devices_for_type('CUDA')) > 0:
            # Use all NVIDIA GPUs
            for device in preferences.devices:
                if 'NVIDIA' in device.name:
                    device.use = True
            preferences.compute_device_type = 'CUDA'
            bpy.context.scene.cycles.device = 'GPU'
    # output image settings
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = image_resolution[0]
    scene.render.resolution_y = image_resolution[1]
    scene.render.film_transparent = True # transparent background
    # remove the default cube and lights created by blender
    for obj in bpy.data.objects:
        if obj.name != 'Camera':
            bpy.data.objects.remove(obj)

def _create_material_base(name: str) -> Material:
    """
    Create an empty material which only has an output node.

    :param name: Name of this material.
    :return: Reference to the newly created material in `bpy.data.materials`.
    """
    bpy.data.materials.new(name=name)
    material: Material = bpy.data.materials[name]
    material.use_nodes = True
    nodes: bpy_prop_collection = material.node_tree.nodes
    links: bpy_prop_collection = material.node_tree.links
    # remove the default Principle BSDF node in the material's node tree
    for node in nodes:
        if node.type != 'OUTPUT_MATERIAL':
            nodes.remove(node)
    return material

def _create_diffuse_material_base(name: str, color: Tuple[float, float, float], roughness: float) -> Material:
    """
    Create a material with basic attributes, which needs to be configured before
    being used for rendering. This function will turn on the Shading Nodes and
    create a Diffuse BSDF node, but will not connect it to the output node.

    :param name: Name of this material.
    :param color: Diffuse color in RGB format, value of each channel should be between 0 and 1.
    :param roughness: Roughness of the surface (between 0 and 1).
    :return: Reference to the newly created material in `bpy.data.materials`.
    """
    color = (color[0], color[1], color[2], 1.0)
    material: Material = _create_material_base(name)
    nodes: bpy_prop_collection = material.node_tree.nodes
    links: bpy_prop_collection = material.node_tree.links
    # add a Diffuse BSDF node
    BSDF_node = nodes.new('ShaderNodeBsdfDiffuse')
    BSDF_node.inputs['Color'].default_value = color
    BSDF_node.inputs['Roughness'].default_value = roughness
    # Use the true normal
    geometry_input = nodes.new('ShaderNodeNewGeometry')
    links.new(geometry_input.outputs['True Normal'], BSDF_node.inputs['Normal'])
    return material

def create_surface_material(
    name: str,
    color: Tuple[float, float, float],
    roughness: float=0.5
) -> Material:
    """
    Create a material for diffuse surface (i.e. no specular color).

    :param name: Name of this material.
    :param color: Diffuse color in RGB format, value of each channel should be between 0 and 1.
    :param roughness: Roughness of the surface (between 0 and 1).
    :return: Reference of the newly created material in `bpy.data.materials`.
    """
    material = _create_diffuse_material_base(name, color, roughness)
    nodes: bpy_prop_collection = material.node_tree.nodes
    links: bpy_prop_collection = material.node_tree.links
    links.new(nodes['Diffuse BSDF'].outputs['BSDF'], nodes['Material Output'].inputs['Surface'])
    return material

def create_transparent_material(
    name: str,
    color: Tuple[float, float, float, float],
    roughness: float=0.5
) -> Material:
    """
    Create a material for transparent surface.
    
    :param name: Name of this material.
    :param color: Diffuse color in RGBA format, value of each channel should be between 0 and 1.
    :param roughness: Roughness of the surface (between 0 and 1).
    :return: Reference of the newly created material in `bpy.data.materials`.
    """
    material = _create_diffuse_material_base(name, (color[0], color[1], color[2]), roughness)
    alpha = color[3]
    nodes: bpy_prop_collection = material.node_tree.nodes
    links: bpy_prop_collection = material.node_tree.links

    BSDF_node = nodes['Diffuse BSDF']
    output_node = nodes['Material Output']
    # for a transparent material, create a Mix Shader node and enable color
    # blending
    transparent_node = nodes.new('ShaderNodeBsdfTransparent')
    mix_node = nodes.new('ShaderNodeMixShader')
    mix_node.inputs['Fac'].default_value = 0.5

    # here we have to use index instead of key to access the 'Shader' input
    # of a Mix Shader node, because there are two input slots with the same
    # name 'Shader' and we need to use both of them
    links.new(BSDF_node.outputs['BSDF'], mix_node.inputs[1])
    links.new(transparent_node.outputs['BSDF'], mix_node.inputs[2])
    links.new(mix_node.outputs['Shader'], output_node.inputs['Surface'])

    material.blend_method = 'BLEND'
    material.shadow_method = 'CLIP'

    return material

def create_normal_material(name: str) -> Material:
    """
    Create a material implementing normal shading (normal vector as color),
    which is not affected by any light.

    :param name: Name of this material.
    """
    material: Material = _create_material_base(name)
    nodes: bpy_prop_collection = material.node_tree.nodes
    links: bpy_prop_collection = material.node_tree.links

    mapping_node = nodes.new('ShaderNodeMapping')
    # Blender apply scaling before translation, so we construct the following
    # transformation to map range [-1, 1] to [0, 1] (i.e. range of normal vector
    # to range of color).
    mapping_node.vector_type = 'POINT' # treat normal vector as point for transformation
    mapping_node.inputs['Location'].default_value = (0.5, 0.5, 0.5)
    mapping_node.inputs['Scale'].default_value = (0.5, 0.5, 0.5)
    geometry_input = nodes.new('ShaderNodeNewGeometry')
    links.new(geometry_input.outputs['True Normal'], mapping_node.inputs['Vector'])
    bg_node = nodes.new('ShaderNodeBackground')
    links.new(mapping_node.outputs['Vector'], bg_node.inputs['Color'])
    output_node = nodes['Material Output']
    links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

def create_orientation_material(name: str) -> Material:
    """
    Create a material indicating face orientation with color. Font faces are blue and back faces are
    red. A diffuse BSDF material is mixed with this binary indicator material in order to make the
    edges more distinguishable.
    
    When using this material, you may want to turn off shadows because they may affect visualization
    of orientation. It is recommended to use the EEVEE renderer, and set
    `bpy.context.scene.eevee.use_shadows = False`.

    :param name: Name of this material.
    """
    material: Material = _create_material_base(name)
    nodes: bpy_prop_collection = material.node_tree.nodes
    links: bpy_prop_collection = material.node_tree.links

    geometry_input = nodes.new('ShaderNodeNewGeometry')
    # Invert the 'Backfacing' input, i.e. 1 to 0 and 0 to 1.
    invertor_1 = nodes.new('ShaderNodeMath') # -0.5
    invertor_1.operation = 'SUBTRACT'
    links.new(geometry_input.outputs['Backfacing'], invertor_1.inputs[0])
    invertor_1.inputs[1].default_value = 0.5
    invertor_2 = nodes.new('ShaderNodeMath') # *-1
    invertor_2.operation = 'MULTIPLY'
    links.new(invertor_1.outputs['Value'], invertor_2.inputs[0])
    invertor_2.inputs[1].default_value = -1.0
    invertor_3 = nodes.new('ShaderNodeMath') # +0.5
    invertor_3.operation = 'ADD'
    links.new(invertor_2.outputs['Value'], invertor_3.inputs[0])
    invertor_3.inputs[1].default_value = 0.5
    # Combine the Backfacing and its inversion together.
    combinator = nodes.new('ShaderNodeCombineColor')
    links.new(geometry_input.outputs['Backfacing'], combinator.inputs['Red'])
    combinator.inputs['Green'].default_value = 0
    links.new(invertor_3.outputs['Value'], combinator.inputs['Blue'])
    # Add a pure-white diffuse BSDF node
    base_color = nodes.new('ShaderNodeBsdfDiffuse')
    base_color.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    base_color.inputs['Roughness'].default_value = 0.0
    links.new(geometry_input.outputs['True Normal'], base_color.inputs['Normal'])
    # Mix orientation indicator material and base color
    mix_node = nodes.new('ShaderNodeMixShader')
    links.new(combinator.outputs['Color'], mix_node.inputs[1]) # 1st 'shader' input slot
    links.new(base_color.outputs['BSDF'], mix_node.inputs[2]) # 2nd 'shader' input slot
    mix_node.inputs['Fac'].default_value = 0.15
    output_node = nodes['Material Output']
    links.new(mix_node.outputs['Shader'], output_node.inputs['Surface'])

def create_point_color_material(name: str) -> Material:
    """
    Create a material for rendering point cloud with per-point color.
    Because points are treated as vertices by PLY format, A PLY point cloud file
    will be imported as a vertex-only mesh by Blender. You should use a geometry
    node modifier to instantiate something on each vertex before rendering.

    :param name: Name of this material.
    :return: Reference of the newly created material in `bpy.data.materials`.
    """
    material = _create_material_base(name)
    nodes: bpy_prop_collection = material.node_tree.nodes
    links: bpy_prop_collection = material.node_tree.links

    input_node = nodes.new('ShaderNodeAttribute')
    # After applying the geometry node on a point cloud, each point will be
    # replaced by an ico sphere instance.
    input_node.attribute_type = 'INSTANCER'
    # Blender's PLY importer import the point color attribute as `Col` attribute
    input_node.attribute_name = 'Col'
    BSDF_node = nodes.new('ShaderNodeBsdfDiffuse')
    output_node = nodes['Material Output']

    links.new(input_node.outputs['Color'], BSDF_node.inputs['Color'])
    links.new(BSDF_node.outputs['BSDF'], output_node.inputs['Surface'])
    return material

def init_lights(light_params: List[Dict[str, Any]], energy_factor: float=1.0, distance_factor: float=1.0) -> None:
    """
    Set lights for rendering. By default, this function will place

    - two sun lights above the object
    - one point light below the object

    With the default distance_factor, the object is assumed to be normalized,
    i.e. it can be enclosed by a unit cube centered at (0, 0, 0), all coordinates
    are in [-0.5, -0.5].

    :param light_params: A list containing all lights' parameters. See the pre-defined
        light parameters in this file for more details.
    :param energy_factor: The factor controlling light intensity, larger for brighter
        images.
    :param distance_factor: The factor controlling the distance from each light source
        to the origin. With a larger (or smaller) factor you can render bigger (or
        smaller) objects.
    """
    for param in light_params:
        light = bpy.data.lights.new(name=param['name'], type=param['type'])
        light.energy = param['energy'] * energy_factor
        if param['type'] == 'SUN':
            light.angle = param['angle']
        elif param['type'] == 'AREA':
            light.size = param['size']
        light_obj = bpy.data.objects.new(name=param['name'], object_data=light)
        light_obj.location = param['location'] * distance_factor
        light_obj.rotation_euler = param['rotation']
        bpy.context.collection.objects.link(light_obj)

def create_pointcloud_modifier(
    sphere_material: Material,
    radius: float=0.005,
    subdivisions: int=3
) -> GeometryNodeTree:
    """
    Create the geometry nodes as a modifier for point clouds. This modifier will
    expand each point to an ico sphere for rendering.

    :param material: Material used by the spheres.
    :param radius: Radius of each sphere.
    :param subdivisions: How much times the ico sphere is subdivided. More subdivisions
        produce smoother sphere and longer rendering time.
    :return: Reference to the newly created Geometry Node.
    """
    # create a node group and enable it as a geometry modifier
    geom_nodes = bpy.data.node_groups.new('pointcloud modifier', 'GeometryNodeTree')
    geom_nodes.is_modifier = True
    nodes = geom_nodes.nodes
    links = geom_nodes.links
    interface = geom_nodes.interface
    interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

    # create all node with their properties set
    input_node = nodes.new('NodeGroupInput')
    output_node = nodes.new('NodeGroupOutput')
    mesh_to_points_node = nodes.new('GeometryNodeMeshToPoints')
    mesh_to_points_node.mode = 'VERTICES'
    ico_sphere_node = nodes.new('GeometryNodeMeshIcoSphere')
    ico_sphere_node.inputs['Radius'].default_value = radius
    ico_sphere_node.inputs['Subdivisions'].default_value = subdivisions # control the smoothness of the ico sphere
    instance_node = nodes.new('GeometryNodeInstanceOnPoints')
    material_node = nodes.new('GeometryNodeReplaceMaterial')
    # only set the New slot of the Replace Material node because we actually
    # use it to set the material of output instances (spheres), the Old slot
    # is not used.
    material_node.inputs['New'].default_value = sphere_material

    # link the nodes
    links.new(input_node.outputs['Geometry'], mesh_to_points_node.inputs['Mesh'])
    # the PLY file are imported as mesh, so we need to replace each vertex in
    # the mesh with a point, then we will have a real point cloud in Blender
    links.new(mesh_to_points_node.outputs['Points'], instance_node.inputs['Points'])
    # use the pre-defined ico sphere as the template instance. with the
    # Instance On Points node we can instantiate an instance at each point in
    # the point cloud
    links.new(ico_sphere_node.outputs['Mesh'], instance_node.inputs['Instance'])
    links.new(instance_node.outputs['Instances'], material_node.inputs['Geometry'])
    links.new(material_node.outputs['Geometry'], output_node.inputs['Geometry'])

    return geom_nodes

def clear_imported_objects(
    protected_material_names: List[str]
) -> None:
    """
    Remove the imported mesh and point cloud from the current scene, together
    with the materials automatically created by Blender when importing a mesh.

    :param protected_material_names: The materials you do not want to remove.
    """
    for obj in bpy.data.objects:
        # after application of geometry nodes, the point cloud data will also
        # be mesh
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj)

    for material in bpy.data.materials:
        if material.name not in protected_material_names:
            bpy.data.materials.remove(material)
