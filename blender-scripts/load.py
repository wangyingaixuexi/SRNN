# built-in packages
from typing import List, Tuple, Union
import os
from pathlib import Path
# blender packages
import bpy
from bpy.types import (
    Scene, Material, Object, GeometryNodeTree
)
from mathutils import Vector, Euler
# third-party packages
import numpy as np


def load_single_mesh(obj_file_path: Union[str, os.PathLike], material: Material) -> Object:
    file_path = Path(obj_file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f'Please check if {file_path} exists, maybe you have to use absolute path instead of relative path'
        )

    previous_names = list(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=str(file_path), forward_axis='NEGATIVE_Z', up_axis='Y')
    mesh = None
    res = bpy.ops.object.select_all(action='DESELECT') # un-select all objects
    for name in bpy.data.objects.keys():
        if name not in previous_names:
            mesh = bpy.data.objects[name]
            # Activate the imported object and remove all redundant material slots.
            # Only one slot is kept and the specified material will be linked to it.
            n_slots = len(mesh.material_slots)
            bpy.context.view_layer.objects.active = mesh
            mesh.active_material_index = 0 # set the active **material slot** to 0
            while len(mesh.material_slots) > 1:
                bpy.ops.object.material_slot_remove()
            mesh.active_material = material
    if mesh is None:
        raise RuntimeError('The specified file contains nothing')
    return mesh

def load_point_cloud(ply_file_path: Union[str, os.PathLike]) -> Object:
    file_path = Path(ply_file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f'Please check if {file_path} exists, maybe you have to use absolute path instead of relative path'
        )

    previous_names = list(bpy.data.objects.keys())
    bpy.ops.wm.ply_import(filepath=str(file_path), forward_axis='NEGATIVE_Z', up_axis='Y')
    pcd = None
    for name in bpy.data.objects.keys():
        if name not in previous_names:
            pcd = bpy.data.objects[name]
    if pcd is None:
        raise RuntimeError('The specified file contains nothing')
    
    return pcd

def apply_modifier(point_cloud: Object, modifier: GeometryNodeTree) -> None:
    current_modifier = point_cloud.modifiers.new('modifier', 'NODES')
    current_modifier.node_group = modifier
