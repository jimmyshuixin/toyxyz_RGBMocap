"""Captured motion's output world: +Z forward (Blender/PMX -Y).

Native inference, contact caches and shared-scene placements retain their
original coordinates. Apply this proper half-turn once at presentation/export
boundaries, after shared placement. Rotating the whole world preserves turns,
trajectories and interactions instead of independently reorienting each actor.
"""
from copy import deepcopy
import numpy as np


ROTATION = np.diag([-1., 1., -1.])
TRANSFORM = np.diag([-1., 1., -1., 1.])
ROTATION.setflags(write=False)
TRANSFORM.setflags(write=False)
HEADING_VERSION = 1


def output_transform(placement=None):
    """Map a native take into the common output world, in metres/Y-up."""
    return TRANSFORM.copy() if placement is None else TRANSFORM @ np.asarray(placement, dtype=float)


def output_points(points):
    return np.asarray(points) @ ROTATION.T


def source_camera(eye, basis):
    """Equivalent camera for drawing native coordinates in the output world."""
    return np.asarray(eye) @ ROTATION, np.asarray(basis) @ ROTATION


def character_transform(take):
    """Captured bodies turn with their source; rest/face-only rigs stay native."""
    placement = take.get('transform')
    return output_transform(placement) if take.get('source') else placement


def upgrade_saved_camera(camera):
    """Copy a saved orbit camera into the output world without changing its view.

    Unmarked cameras predate the output half-turn. Transport both the look-at
    position and orbit offset so an old distant actor stays in the same frame.
    The returned marker makes repeated restore/save operations idempotent; the
    supplied project metadata is never modified or written here.
    """
    result = deepcopy(camera)
    if 'heading_version' in result:
        if result['heading_version'] != HEADING_VERSION:
            raise ValueError('Unsupported saved camera heading version.')
        return result
    for key in ('scene_center', 'target', 'pan'):
        if key in result:
            result[key] = output_points(result[key]).tolist()
    if 'yaw' in result:
        result['yaw'] = float(result['yaw']) + np.pi
    result['heading_version'] = HEADING_VERSION
    return result
