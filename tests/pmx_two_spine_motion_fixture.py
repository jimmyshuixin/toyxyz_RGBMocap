"""Run the real CPU retarget pipeline on an explicitly synthetic torso motion.

Usage:
    python tests/pmx_two_spine_motion_fixture.py --folder /path/to/pmx-import

Repeat --folder to validate several separately imported characters. The input
folder must contain RGB MoCap's canonical, contact-geometry, rig and rest-blend
assets. Only generated fixture/retarget outputs are written. No PMX coordinates,
user model, licensed body model, AI weights or inferred motion are in this script.
The generated source-rig files also support the application's real Blender bake.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mocap.bone_overlay import reference_frame  # noqa: E402
from mocap.retarget import pipeline  # noqa: E402
from mocap.retarget.mapping import NAMES, PARENTS  # noqa: E402


SOURCE_FILES = (
    'motion-source.npz', 'motion-source-neutral.npz',
    'motion-source-contacts.npz', 'motion-neutral-calibration.npz',
    'motion-evidence.npz', 'sources.json',
    'source-rig-asset.npz', 'source-rig-frames.npy',
)
IMPORT_FILES = (
    'pmx-canonical.npz', 'pmx-contact-geometry.npz',
    'pmx-rig.json', 'pmx-rest.blend', 'import.json',
)
FIXTURE_KIND = 'synthetic_three_spine_source_v1'


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_fk(local, rest, root):
    """Independent source FK using homogeneous transforms, not solver.fk."""
    count = len(local)
    pose = np.empty((count, 22, 4, 4), dtype=np.float64)
    for slot, parent in enumerate(PARENTS):
        transform = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
        transform[:, :3, :3] = local[:, slot]
        transform[:, :3, 3] = root if parent < 0 else rest[slot] - rest[parent]
        pose[:, slot] = transform if parent < 0 else pose[:, parent] @ transform
    return pose


def local_angles(local):
    return np.degrees(Rotation.from_matrix(local.reshape(-1, 3, 3)).magnitude()).reshape(len(local), 22)


def axis_rotation(axis, degrees):
    vectors = np.zeros((len(degrees), 3))
    vectors[:, axis] = np.radians(degrees)
    return Rotation.from_rotvec(vectors).as_matrix()


def write_fixture(folder, count, fps):
    marker = folder / 'motion-fixture.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')).get('fixture_kind') != FIXTURE_KIND:
            raise ValueError('This folder contains a different motion fixture.')
    elif any((folder / name).exists() for name in SOURCE_FILES):
        raise ValueError('Source motion files already exist; use a separate validation import folder.')
    for name in IMPORT_FILES:
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)
    with np.load(folder / 'pmx-canonical.npz', allow_pickle=False) as canonical:
        target_rest = canonical['rest'].copy()
        virtual = pipeline._target_virtual_slots(folder, 'pmx', canonical)
    if target_rest.shape != (22, 3) or not np.isfinite(target_rest).all():
        raise ValueError('The target rest data is not a finite 22-joint skeleton.')
    rest = target_rest.copy()
    if len(virtual):
        rest[6] = (rest[3] + rest[9]) / 2
    torso_lengths = np.linalg.norm(rest[[6, 9]] - rest[[3, 6]], axis=1)
    if not np.all(torso_lengths > 1e-6):
        raise ValueError('The synthetic source must contain two nonzero torso links.')

    phase = np.linspace(0, 2 * np.pi, count)
    local = np.broadcast_to(np.eye(3), (count, 22, 3, 3)).copy()
    # Separate, noncommuting source joints: bend, twist, and lateral bend.
    local[:, 3] = axis_rotation(0, 14 * np.sin(phase))
    local[:, 6] = axis_rotation(1, 20 * np.sin(phase))
    local[:, 9] = axis_rotation(2, 10 * np.sin(2 * phase))
    local[:, 0] = axis_rotation(1, 4 * np.sin(phase))
    # Bent elbows supply reliable calibration if a later validation opts into
    # hand solving. This fixture never requests hands to contact the floor.
    local[:, 18] = axis_rotation(2, 75 + 8 * np.sin(phase))
    local[:, 19] = axis_rotation(2, -75 - 8 * np.sin(phase))
    root = np.broadcast_to(rest[0], (count, 3)).copy()
    source_frames = source_fk(local, rest, root)
    world = source_frames[:, :, :3, :3].copy()
    points = source_frames[:, :, :3, 3].copy()
    active = np.zeros((count, 4), dtype=bool)
    left_stop = round(count * .75)
    right_start = round(count * .25)
    active[:left_stop, 0] = True
    active[right_start:, 1] = True

    with np.load(folder / 'pmx-contact-geometry.npz', allow_pickle=False) as geometry:
        vertices = np.concatenate([geometry[f'foot_v{side}'] for side in (0, 1)])
        weights = np.concatenate([geometry[f'foot_w{side}'] for side in (0, 1)])
        if weights.shape != (len(vertices), 22):
            raise ValueError('The imported foot geometry needs 22-column proxy weights.')
        if len(virtual):
            for key in ('foot_w0', 'foot_w1', 'palm_w0', 'palm_w1'):
                if np.any(geometry[key][:, virtual] != 0):
                    raise ValueError('A virtual target slot must not have skin weights.')
    source_eligible = []
    for side in (0, 1):
        eligible = ((weights[:, [7 + side, 10 + side]].sum(1) > .7)
                    & (vertices[:, 1] < rest[7 + side, 1] + .02))
        source_eligible.append(int(eligible.sum()))
        if eligible.sum() < 6:
            raise ValueError('Not enough imported sole geometry for source foot calibration.')

    basis = [reference_frame(rest[knee] - rest[hip]) for hip, knee in ((1, 4), (2, 5))]
    evidence = np.stack([world[:, hip] @ basis[side] for side, hip in enumerate((1, 2))], axis=1)
    neutral_frames = np.stack(basis)
    frame_axes = []
    ends = []
    display_children = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11,
                        9: 12, 12: 15, 13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21}
    for slot, parent in enumerate(PARENTS):
        direction = rest[display_children[slot]] - rest[slot] if slot in display_children else (rest[slot] - rest[parent]) * .25
        frame_axes.append(reference_frame(direction))
        ends.append(direction)
    frame_axes = np.stack(frame_axes)
    metadata = {
        'fixture_kind': FIXTURE_KIND, 'synthetic_motion': True,
        'ai_inference_performed': False, 'frames': count, 'fps': fps,
        'source_names': NAMES, 'target_virtual_slots': virtual.tolist(),
        'source_torso_link_lengths_m': torso_lengths.tolist(),
        'peak_local_torso_angles_deg': local_angles(local)[:, [3, 6, 9]].max(axis=0).tolist(),
        'foot_contact_intervals': [[0, 0, left_stop], [1, right_start, count]],
        'contact_frames_per_part': active.sum(axis=0).tolist(),
        'source_sole_vertices_per_side': source_eligible,
        'source_surface': 'Imported foot contact patches only; no licensed parametric body model.',
    }
    marker.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    np.savez(folder / 'motion-source.npz', names=np.asarray(NAMES), parents=PARENTS,
             rest=rest, points=points, world=world, local=local,
             origin=np.zeros(3), fps=fps)
    np.savez(folder / 'motion-source-neutral.npz', vertices=vertices, weights=weights,
             rest=rest, faces=np.empty((0, 3), np.int64), native_world=world)
    np.savez(folder / 'motion-source-contacts.npz', active=active, fps=fps)
    np.savez(folder / 'motion-neutral-calibration.npz', neutral=np.broadcast_to(np.eye(3), (22, 3, 3)).copy())
    np.savez(folder / 'motion-evidence.npz', frames=evidence, neutral_frames=neutral_frames)
    np.savez(folder / 'source-rig-asset.npz', rest=rest, ends=np.asarray(ends),
             fbx_rest_frames=frame_axes, fbx_display_frames=frame_axes.copy(),
             names=np.asarray(NAMES), parents=PARENTS)
    np.save(folder / 'source-rig-frames.npy', source_frames)
    info = dict(names=NAMES, parents=PARENTS.tolist(), fps=fps, frames=count,
                asset_file='source-rig-asset.npz', frames_file='source-rig-frames.npy',
                synthetic_motion=True, fixture_kind=FIXTURE_KIND)
    (folder / 'sources.json').write_text(json.dumps([
        dict(label='motion', folder=str(folder.resolve()), info=info)
    ], indent=2), encoding='utf-8')
    return metadata


def validate_result(folder, count, method, virtual):
    path = folder / f'motion-pmx-{method}.npz'
    with np.load(path, allow_pickle=False) as result:
        arrays = {key: result[key] for key in ('rest', 'root', 'world', 'local', 'points')}
        if arrays['world'].shape != (count, 22, 3, 3) or arrays['points'].shape != (count, 22, 3):
            raise AssertionError(f'{method}: incomplete output sequence')
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise AssertionError(f'{method}: nonfinite output')
        expected = np.linalg.norm(arrays['rest'][1:] - arrays['rest'][PARENTS[1:]], axis=1)
        actual = np.linalg.norm(arrays['points'][:, 1:] - arrays['points'][:, PARENTS[1:]], axis=2)
        length_error = float(np.max(np.abs(actual - expected)))
        if length_error >= 1e-8:
            raise AssertionError(f'{method}: fixed lengths changed by {length_error}')
        rotations = arrays['world']
        orthogonality = float(np.max(np.abs(rotations.swapaxes(-1, -2) @ rotations - np.eye(3))))
        determinants = np.linalg.det(rotations)
        if orthogonality > 1e-8 or not np.allclose(determinants, 1, rtol=0, atol=1e-8):
            raise AssertionError(f'{method}: invalid rotation matrices')
        virtual_error = 0.0
        if virtual:
            if not np.array_equal(arrays['rest'][6], arrays['rest'][9]):
                raise AssertionError(f'{method}: virtual rest point changed')
            virtual_error = float(np.max(np.abs(arrays['points'][:, 6] - arrays['points'][:, 9])))
            if virtual_error > 1e-12:
                raise AssertionError(f'{method}: virtual pivot is no longer coincident')
        return dict(file=path.name, frames=count, all_finite=True,
                    fixed_length_max_error_m=length_error,
                    rotation_orthogonality_max_error=orthogonality,
                    rotation_determinant_min=float(determinants.min()),
                    rotation_determinant_max=float(determinants.max()),
                    virtual_point_max_error_m=virtual_error)


def run_validation(folder, count, fps):
    folder = folder.resolve()
    report = dict(status='failed', folder=str(folder), synthetic_motion=True,
                  ai_inference_performed=False, blender_validation_performed=False,
                  native_mmd_validation_performed=False)
    started = time.perf_counter()
    try:
        before = {name: fingerprint(folder / name) for name in IMPORT_FILES}
        report['fixture'] = write_fixture(folder, count, fps)
        print(f'FIXTURE READY {folder} ({count} frames)', flush=True)
        log_path = folder / 'full-pipeline-validation.log'
        with log_path.open('w', encoding='utf-8') as log, redirect_stdout(log):
            report['pipeline_diagnostic'] = pipeline.run(folder, 'pmx')
        report['solver_records'] = [json.loads(line.split(' ', 2)[2])
                                    for line in log_path.read_text(encoding='utf-8').splitlines()
                                    if line.startswith('READY pmx ')]
        virtual = report['fixture']['target_virtual_slots']
        report['outputs'] = [validate_result(folder, count, method, virtual)
                             for method in ('rotation', 'stance')]
        after = {name: fingerprint(folder / name) for name in IMPORT_FILES}
        report['import_asset_sha256'] = before
        report['import_assets_unchanged'] = before == after
        if before != after:
            raise AssertionError('An imported character asset changed during validation')
        report['status'] = 'passed'
    except Exception as error:
        report['exception'] = f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
    report['elapsed_seconds'] = time.perf_counter() - started
    output = folder / 'full-pipeline-validation.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report.get(key) for key in ('folder', 'status', 'elapsed_seconds', 'exception')}), flush=True)
    print(f'REPORT {output}', flush=True)
    return report['status'] == 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, action='append', required=True)
    parser.add_argument('--frames', type=int, default=61)
    parser.add_argument('--fps', type=float, default=30.0)
    args = parser.parse_args()
    if args.frames < 45 or args.fps != 30:
        parser.error('Use at least 45 frames at 30 fps for this controlled fixture.')
    results = [run_validation(folder, args.frames, args.fps) for folder in args.folder]
    return 0 if all(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
