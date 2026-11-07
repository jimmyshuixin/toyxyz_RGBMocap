"""Run inside Blender against a separately supplied, licensed PMX fixture.

The folder must contain a real import and outputs from
pmx_two_spine_motion_fixture.py. This exercises RGB MoCap's actual bake/preview
and custom VMD exporter, then imports that VMD into a fresh native PMX rig.
It does not run AI inference or the native Windows MMD application.

blender -b --factory-startup --disable-autoexec --python-exit-code 1 \
  --python tests/blender_pmx_two_spine_integration.py -- \
  --folder /absolute/validation/folder --pmx /absolute/model.pmx
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import addon_utils
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/character_import'))

from mocap.retarget.mapping import A, VIRTUAL_SPINE1, real_bone_mapping
from mocap.retarget.blender_io import main as character_operation
from mmd_tools.core import vmd
from mmd_tools.core.vmd.importer import VMDImporter
from mmd_tools.utils import makePmxBoneMap


def topology(arm):
    """Static bones and base mesh data, independent of object placement/pose."""
    bones = {b.name: {'parent': b.parent.name if b.parent else None,
                      'rest': np.array(b.matrix_local),
                      'head': np.array(b.head_local), 'tail': np.array(b.tail_local)}
             for b in arm.data.bones}
    meshes = {}
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        co = np.empty(len(obj.data.vertices) * 3, np.float32)
        obj.data.vertices.foreach_get('co', co)
        weights = [(v.index, obj.vertex_groups[g.group].name, float(g.weight))
                   for v in obj.data.vertices for g in v.groups]
        meshes[obj.name] = (co, weights)
    return bones, meshes


def assert_topology_same(first, second):
    first_bones, first_meshes = first
    second_bones, second_meshes = second
    assert first_bones.keys() == second_bones.keys(), 'Native bone names changed'
    assert first_meshes.keys() == second_meshes.keys(), 'Native meshes changed'
    for name in first_bones:
        assert first_bones[name]['parent'] == second_bones[name]['parent'], name
        for key in ('rest', 'head', 'tail'):
            np.testing.assert_allclose(first_bones[name][key], second_bones[name][key],
                                       rtol=0, atol=1e-7, err_msg=f'{name}/{key}')
    for name in first_meshes:
        np.testing.assert_array_equal(first_meshes[name][0], second_meshes[name][0])
        assert first_meshes[name][1] == second_meshes[name][1], name


def mesh_positions():
    result = {}
    graph = bpy.context.evaluated_depsgraph_get()
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            co = np.empty(len(mesh.vertices) * 3, np.float64)
            mesh.vertices.foreach_get('co', co)
            co = co.reshape(-1, 3)
            transform = np.array(evaluated.matrix_world)
            result[obj.name] = (co @ transform[:3, :3].T + transform[:3, 3]) @ A
        finally:
            evaluated.to_mesh_clear()
    return result


def sample(arm, mapping, frames, mesh_frames):
    poses, meshes = {}, {}
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        poses[frame] = {name: np.array(arm.matrix_world @ arm.pose.bones[name].matrix)
                        for name in mapping}
        if frame in mesh_frames:
            meshes[frame] = mesh_positions()
    return poses, meshes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--pmx', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    folder, model = args.folder.resolve(), args.pmx.resolve()
    start = time.perf_counter()
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    addon_utils.enable('mmd_tools', default_set=False, persistent=False)

    rig = json.loads((folder / 'pmx-rig.json').read_text(encoding='utf-8'))
    with np.load(folder / 'motion-pmx-stance.npz') as data:
        names, points, world = data['names'], data['points'], data['world']
    mapping = real_bone_mapping(names, rig.get('body_profile'))
    frame_count = len(points)
    frames = list(range(1, frame_count + 1))
    mesh_frames = sorted({1, frame_count, 1 + frame_count // 4,
                          1 + frame_count // 2, 1 + 3 * frame_count // 4})

    # This is the real app entry point, including bone-driven viewport creation.
    character_operation(['bake', str(folder), 'pmx'])
    arm = bpy.data.objects[rig['armature']]
    original_topology = topology(arm)
    assert VIRTUAL_SPINE1 not in arm.data.bones
    keyed = json.loads((folder / 'bake.json').read_text(encoding='utf-8'))['keyed_bones']
    assert VIRTUAL_SPINE1 not in keyed
    assert all(name in arm.data.bones for name in keyed)
    baked, baked_meshes = sample(arm, mapping, frames, mesh_frames)

    rest_rotations = {name: np.array(arm.matrix_world @ arm.data.bones[name].matrix_local)[:3, :3]
                      for name in mapping}
    position_errors, rotation_errors = [], []
    for frame in frames:
        for name, slot in mapping.items():
            matrix = baked[frame][name]
            position_errors.append(float(np.linalg.norm(matrix[:3, 3] @ A - points[frame - 1, slot])))
            desired = A @ world[frame - 1, slot] @ A.T @ rest_rotations[name]
            rotation_errors.append(float(np.abs(matrix[:3, :3] - desired).max()))
    max_position = max(position_errors)
    max_rotation = max(rotation_errors)
    assert max_position < 5e-5, ('Baked real bones differ from target FK (m)', max_position)
    assert max_rotation < 5e-5, ('Baked real orientations differ', max_rotation)

    target = folder / 'validation.vmd'
    character_operation(['export', str(folder), str(folder), str(target), 'null',
                         model.stem, json.dumps({'format': 'vmd'})])
    motion = vmd.File()
    motion.load(filepath=str(target))
    track_names = sorted(motion.boneAnimation.keys())
    assert VIRTUAL_SPINE1 not in track_names
    if rig['body_profile']['virtual_slots']:
        assert '上半身1' not in track_names
        assert all(name in track_names for name in ('上半身', '上半身2'))

    # Fresh native import is intentional: the VMD root already contains world
    # placement, so importing it onto RGB's translated rest object double-shifts it.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable('mmd_tools', default_set=False, persistent=False)
    bpy.ops.mmd_tools.import_model(filepath=str(model), types={'MESH', 'ARMATURE'},
                                   scale=.08, log_level='ERROR')
    arm = next(obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE')
    assert_topology_same(original_topology, topology(arm))
    native_names = set(makePmxBoneMap(arm))
    assert set(track_names) <= native_names
    VMDImporter(str(target), scale=.08, bone_mapper=makePmxBoneMap,
                frame_margin=0).assign(arm)
    restored, restored_meshes = sample(arm, mapping, frames, mesh_frames)
    matrix_error = max(float(np.abs(baked[f][name] - restored[f][name]).max())
                       for f in frames for name in mapping)
    mesh_error = max(float(np.linalg.norm(baked_meshes[f][name] - restored_meshes[f][name], axis=1).max())
                     for f in mesh_frames for name in baked_meshes[f])
    assert matrix_error < 5e-5, ('VMD bone round-trip mismatch', matrix_error)
    assert mesh_error < 1e-4, ('VMD evaluated mesh round-trip mismatch (m)', mesh_error)
    assert hashlib.sha256(model.read_bytes()).hexdigest() == model_hash

    record = dict(result='PASS', blender=bpy.app.version_string, model=model.name,
                  model_sha256=model_hash, profile=rig['body_profile']['id'],
                  frames=frame_count, mesh_sample_frames=mesh_frames,
                  real_body_bindings=len(mapping), native_bone_count=len(arm.data.bones),
                  native_bones_bind_mesh_weights_unchanged=True,
                  canonical_to_blender_position_max_m=max_position,
                  canonical_to_blender_rotation_component_max=max_rotation,
                  vmd_roundtrip_matrix_component_max=matrix_error,
                  vmd_roundtrip_evaluated_vertex_max_m=mesh_error,
                  vmd_tracks=track_names, seconds=time.perf_counter() - start,
                  scope='Controlled three-spine motion; real RGB contact pipeline outputs, '
                        'Blender bake/preview, RGB VMD export, mmd_tools fresh-native-PMX reimport. '
                        'No AI inference, Windows GUI, or native MMD playback tested.')
    (folder / 'blender-integration-validation.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PMX_INTEGRATION_VALIDATION ' + json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
