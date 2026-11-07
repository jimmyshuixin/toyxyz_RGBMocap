"""Real Blender PMX bake, viewport skinning and VMD round-trip measurements.

Supply an isolated import folder containing prepared motion, never user projects.
This checks the renderer's LBS data independently against evaluated Blender meshes.
It does not run inference, the GUI or the native MMD application.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import addon_utils
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/character_import'))
sys.path.insert(0, str(ROOT / 'tests'))

from blender_pmx_two_spine_integration import topology, assert_topology_same, mesh_positions
from mocap.retarget.mapping import A, PMX
from mocap.retarget.blender_io import main as operation
from mmd_tools.core import vmd
from mmd_tools.core.vmd.importer import VMDImporter
from mmd_tools.utils import makePmxBoneMap


def preview_positions(folder, frame):
    with np.load(folder / 'preview.npz', allow_pickle=False) as data:
        vertices, rows, influences = data['vertices'], data['rows'], data['weights']
    frames = np.load(folder / 'preview-frames.npy', mmap_mode='r')
    try:
        matrices = frames[frame - 1].astype(np.float64)
        owners = np.repeat(np.arange(len(vertices)), rows[:, 1])
        assert np.array_equal(rows[:, 0], np.r_[0, np.cumsum(rows[:-1, 1])])
        transformed = np.einsum('nij,nj->ni', matrices[influences[:, 0].astype(int), :3, :3], vertices[owners])
        transformed += matrices[influences[:, 0].astype(int), :3, 3]
        result = np.zeros_like(vertices, dtype=np.float64)
        np.add.at(result, owners, transformed * influences[:, 1, None])
        return result
    finally:
        frames._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--pmx', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    folder, pmx = args.folder.resolve(), args.pmx.resolve()
    started = time.perf_counter()
    report = dict(status='failed', model=str(pmx), blender=bpy.app.version_string,
                  ai_inference_performed=False, native_mmd_performed=False)
    digest = hashlib.sha256(pmx.read_bytes()).hexdigest()
    try:
        addon_utils.enable('mmd_tools', default_set=False, persistent=False)
        rig = json.loads((folder / 'pmx-rig.json').read_text(encoding='utf-8'))
        with np.load(folder / 'motion-pmx-stance.npz', allow_pickle=False) as data:
            names = data['names'].tolist()
            assert names == PMX, 'Import a PMX supported by the upstream 22-bone mapping'
            mapping = {name: slot for slot, name in enumerate(names)}
            expected_points, expected_world = data['points'], data['world']
        count = len(expected_points)
        frames = sorted({1, count, 1 + count // 4, 1 + count // 2, 1 + 3 * count // 4})
        operation(['bake', str(folder), 'pmx'])
        arm = bpy.data.objects[rig['armature']]
        original_topology = topology(arm)
        baked, meshes = {}, {}
        position, rotation, preview = {}, {}, {}
        for frame in frames:
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            baked[frame] = {name: np.array(arm.matrix_world @ arm.pose.bones[name].matrix) for name in mapping}
            meshes[frame] = mesh_positions()
            evaluated = np.concatenate(list(meshes[frame].values()))
            preview[frame] = float(np.linalg.norm(preview_positions(folder, frame) - evaluated, axis=1).max())
            for name, slot in mapping.items():
                matrix = baked[frame][name]
                position[name] = max(position.get(name, 0.), float(np.linalg.norm(matrix[:3, 3] @ A - expected_points[frame - 1, slot])))
                rest = np.array(arm.matrix_world @ arm.data.bones[name].matrix_local)[:3, :3]
                desired = A @ expected_world[frame - 1, slot] @ A.T @ rest
                rotation[name] = max(rotation.get(name, 0.), float(np.abs(matrix[:3, :3] - desired).max()))
        report.update(frames=count, sample_frames=frames, profile=rig.get('body_profile'),
                      position_max_m=max(position.values()), rotation_component_max=max(rotation.values()),
                      preview_vertex_max_m=max(preview.values()), position_per_bone_m=position,
                      rotation_per_bone=rotation, preview_per_frame_m=preview)
        target = folder / 'motion-audit.vmd'
        operation(['export', str(folder), str(folder), str(target), 'null', pmx.stem, json.dumps({'format': 'vmd'})])
        motion = vmd.File()
        motion.load(filepath=str(target))
        bpy.ops.wm.read_factory_settings(use_empty=True)
        addon_utils.enable('mmd_tools', default_set=False, persistent=False)
        # Match the application import: UV morphs also create vertex groups,
        # so omitting MORPHS would falsely report changed native skin data.
        bpy.ops.mmd_tools.import_model(filepath=str(pmx), types={'MESH', 'ARMATURE', 'MORPHS'}, scale=.08, log_level='ERROR')
        arm = next(obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE')
        assert_topology_same(original_topology, topology(arm))
        assert set(motion.boneAnimation) <= set(makePmxBoneMap(arm))
        VMDImporter(str(target), scale=.08, bone_mapper=makePmxBoneMap, frame_margin=0).assign(arm)
        roundtrip_bones, roundtrip_meshes = {}, {}
        for frame in frames:
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            restored = mesh_positions()
            for name in mapping:
                error = np.abs(np.array(arm.matrix_world @ arm.pose.bones[name].matrix) - baked[frame][name]).max()
                roundtrip_bones[name] = max(roundtrip_bones.get(name, 0.), float(error))
            for name, vertices in restored.items():
                error = np.linalg.norm(vertices - meshes[frame][name], axis=1).max()
                roundtrip_meshes[name] = max(roundtrip_meshes.get(name, 0.), float(error))
        report.update(vmd_bone_component_max=max(roundtrip_bones.values()),
                      vmd_vertex_max_m=max(roundtrip_meshes.values()),
                      vmd_bones=roundtrip_bones, vmd_meshes_m=roundtrip_meshes,
                      original_model_unchanged=hashlib.sha256(pmx.read_bytes()).hexdigest() == digest,
                      bind_topology_weights_unchanged=True)
        report['status'] = 'passed' if (report['position_max_m'] < 5e-5 and report['rotation_component_max'] < 5e-5
            and report['preview_vertex_max_m'] < 1e-4 and report['vmd_bone_component_max'] < 5e-5
            and report['vmd_vertex_max_m'] < 1e-4 and report['original_model_unchanged']) else 'failed'
    except Exception as error:
        report.update(exception=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    report.update(seconds=time.perf_counter() - started, model_sha256=digest)
    (folder / 'motion-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PMX_MOTION_AUDIT ' + json.dumps({k: v for k, v in report.items() if k not in
        {'profile', 'traceback', 'position_per_bone_m', 'rotation_per_bone', 'vmd_bones', 'vmd_meshes_m'}}, ensure_ascii=False), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
