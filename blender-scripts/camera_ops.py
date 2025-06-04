from typing import Tuple
# blender packages
import bpy
from bpy.types import (
    Object
)
# third-party packages
import numpy as np



def set_viewpoint(
    location: Tuple[float, float, float]=(0.7359, -0.6926, 0.4958),
    distance_factor:float=1.2,
    camera_name: str='Camera'
) -> None:
    """
    Set the camera's position. The default location is suitable for rendering
    an normalized object (whose all coordinates fall in [-0.5, 0.5]).

    :param radius_factor: Control the distance from (0, 0, 0) to the viewpoint.
        A larger factor lets the camera be farther to the origin.
    :param camera_name: Name of the camera you want to apply tracking constraint.
        By default, the name of automatically created camera is "Camera".
    """
    camera_obj: Object = bpy.data.objects[camera_name]
    # the location is obtained through GUI
    camera_obj.location = np.array(location) * distance_factor

def track_object(obj: Object, camera_name: str='Camera') -> None:
    """
    Let the camera track the specified object's center.
    By setting the tracking constraint, you can easily make the camera orient to
    the target object we want to render. This is less flexible but easier than
    setting the rotation manually.

    :param camera_name: Name of the camera you want to apply tracking constraint.
        By default, the name of automatically created camera is "Camera".
    """
    camera: Object = bpy.data.objects[camera_name]
    # the Track To constraint can keep the up direction of the camera better
    # than the Damp Track constraint, allowing placing the camera in the half-
    # space where x < 0
    camera.constraints.new('TRACK_TO')
    constraint = camera.constraints['Track To']
    constraint.target = obj
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
