"""Original capture/retarget world, with migration of saved half-turn cameras.

Preview and export retain the native source and character placement. Version 1
introduced an output-only half-turn; version 2 restores upstream coordinates.
The API remains shared so source, character and camera paths cannot diverge.
"""
from copy import deepcopy
import numpy as np


ROTATION = np.eye(3)
TRANSFORM = np.eye(4)
ROTATION.setflags(write=False)
TRANSFORM.setflags(write=False)
HEADING_VERSION = 2


def output_transform(placement=None):
    """Map a native take into the common output world, in metres/Y-up."""
    return TRANSFORM.copy() if placement is None else TRANSFORM @ np.asarray(placement, dtype=float)


def output_points(points):
    return np.asarray(points) @ ROTATION.T


def source_camera(eye, basis):
    """Equivalent camera for drawing native coordinates in the output world."""
    return np.asarray(eye) @ ROTATION, np.asarray(basis) @ ROTATION


def character_transform(take, kind=None):
    """All characters use upstream take placement, including captured PMX."""
    return take.get('transform')


def upgrade_saved_camera(camera):
    """Restore version-1 cameras to native coordinates without editing input."""
    result = deepcopy(camera)
    version = result.get('heading_version')
    if version == HEADING_VERSION:
        return result
    if version not in (None, 1):
        raise ValueError('Unsupported saved camera heading version.')
    if version == 1:
        inverse = np.diag([-1., 1., -1.])
        for key in ('scene_center', 'target', 'pan'):
            if key in result:
                result[key] = (np.asarray(result[key]) @ inverse).tolist()
        if 'yaw' in result:
            result['yaw'] = float(result['yaw']) - np.pi
    result['heading_version'] = HEADING_VERSION
    return result
