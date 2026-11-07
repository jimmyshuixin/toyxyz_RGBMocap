"""CPU regression tests for selective NPZ reads in long-take retargeting.

Uses real NPZ/NPY files and the actual surface/bake arithmetic. Small model and
Blender API stand-ins avoid requiring weights, CUDA, or a Blender installation.
"""
from collections import Counter
from contextlib import ExitStack, contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mocap.retarget as retarget
from mocap import mesh_data
from mocap.retarget import source
from mocap.retarget.mapping import NAMES, PARENTS, PMX


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


class ArrayReads:
    """Observe real archive reads/handles, and clean up after failed assertions."""
    def __init__(self):
        self.reads = Counter()
        self.archives = []
        self.mappings = []

    def __enter__(self):
        self.stack = ExitStack()
        original_load = np.load
        original_getitem = np.lib.npyio.NpzFile.__getitem__

        def load(path, *args, **kwargs):
            result = original_load(path, *args, **kwargs)
            if isinstance(result, np.lib.npyio.NpzFile):
                self.archives.append((Path(path).name, result))
            elif isinstance(result, np.memmap):
                self.mappings.append((Path(path).name, result))
            return result

        def getitem(archive, key):
            self.reads[Path(archive.zip.filename).name, key] += 1
            return original_getitem(archive, key)

        self.stack.enter_context(patch.object(np, 'load', side_effect=load))
        self.stack.enter_context(patch.object(np.lib.npyio.NpzFile, '__getitem__', getitem))
        return self

    def __exit__(self, *exc):
        self.stack.close()
        for _, archive in self.archives:
            archive.close()
        for _, mapping in self.mappings:
            if not mapping._mmap.closed:
                mapping._mmap.close()

    def assert_closed(self, case, names=None):
        for name, archive in self.archives:
            if names is None or name in names:
                case.assertIsNone(archive.zip, name)
        for name, mapping in self.mappings:
            if names is None or name in names:
                case.assertTrue(mapping._mmap.closed, name)


def source_fixture(folder, count):
    project = folder / 'project'
    output = folder / 'output'
    models = folder / 'models'
    project.mkdir()
    output.mkdir()
    body_path = models / 'body_models/smplx/SMPLX_NEUTRAL.npz'
    body_path.parent.mkdir(parents=True)
    vertices = np.zeros((22, 3), dtype=np.float32)
    vertices[:, 0] = np.arange(22) * .1
    vertices[:, 1] = np.arange(22) * .01
    weights = np.eye(22, 55, dtype=np.float32)
    np.savez(body_path, v_template=vertices, shapedirs=np.zeros((22, 3, 1)),
             J_regressor=np.eye(22), weights=weights, f=np.array([[0, 1, 2]]))
    packed = np.concatenate((vertices.ravel(), np.zeros(22 * 3 * (20 + 486)),
                             weights.ravel())).astype(np.float32)
    # Loading the entire archive instead of selected fields would reject this member.
    unused = np.array([{'unused': True}], dtype=object)
    np.savez(project / 'mesh-asset.npz', packed=packed, vertex_count=22, unused=unused)
    mesh_frames = np.zeros((count, 1389), dtype=np.float32)
    mesh_frames[:, :880] = np.tile(np.eye(4, dtype=np.float32), (55, 1, 1)).ravel()
    mesh_frames[:, -3:] = np.arange(count)[:, None] * np.array([.1, .02, -.03])
    np.save(project / 'mesh-frames.npy', mesh_frames)
    motion = np.tile(np.eye(4), (count, 22, 1, 1))
    motion[:, :, :3, 3] = vertices + np.arange(count)[:, None, None] * [.1, 0., .2]
    np.save(project / 'rig-frames.npy', motion)
    np.savez(project / 'rig-asset.npz', rest=vertices,
             fbx_rest_frames=np.tile(np.eye(3), (22, 1, 1)),
             fbx_display_frames=np.tile(np.eye(3), (22, 1, 1)))
    (project / 'result.pt').write_bytes(b'controlled tensor-loader fixture')
    manifests = {
        'project.json': {'pipeline': {'model': 'exmr'}},
        'initial_mixamo.json': {'fps': 30., 'names': NAMES,
                               'frames_file': 'rig-frames.npy', 'asset_file': 'rig-asset.npz'},
        'initial_mesh.json': {'asset': 'mesh-asset.npz', 'frames_file': 'mesh-frames.npy'},
        'initial_correction.json': {'mode': 'multi_level', 'floor': -.1},
    }
    for name, value in manifests.items():
        (project / name).write_text(json.dumps(value), encoding='utf-8')
    params = {}
    for name, array in [('betas', np.zeros((count, 1))),
                        ('global_orient', np.zeros((count, 3))),
                        ('body_pose', np.zeros((count, 21, 3)))]:
        params[name] = SimpleNamespace(shape=array.shape, numpy=lambda a=array: a)
    surface = module('mocap.retarget.surface', endpoints=lambda *args: np.zeros((8, 4, 3)))
    stubs = {
        'torch': module('torch', Tensor=type('Tensor', (), {}), set_num_threads=lambda n: None,
                        load=lambda *args, **kwargs: {'smpl_params_global': params}),
        'mocap.paths': module('mocap.paths', MODELS=models),
        'mocap.motion_tools': module('mocap.motion_tools', RESULTS={'initial': ('result.pt',)}),
        'mocap.project_storage': module('mocap.project_storage', resolve_reference=lambda p, r: p / r),
        'mocap.source_surface': module('mocap.source_surface', native_endpoints=None,
            smplx_endpoints=lambda p, k, n: (np.zeros((n, 8, 4, 3)), {})),
        'mocap.surface_roll': module('mocap.surface_roll',
            confidence_solve=lambda points, world, *args: (world.copy(),)),
        'mocap.bone_overlay': module('mocap.bone_overlay', reference_frame=lambda v: np.eye(3)),
        'mocap.retarget.surface': surface,
        'mocap.retarget.solver': module('mocap.retarget.solver', local_from_world=lambda w: w.copy()),
    }
    expected_heights = vertices[[7, 8], 1] + mesh_frames[:, -2, None] + .1
    return project, output, stubs, expected_heights, motion


@contextmanager
def source_stubs(stubs):
    with patch.dict(sys.modules, stubs), patch.object(retarget, 'surface', stubs['mocap.retarget.surface'], create=True):
        yield


class Matrix:
    """The few mathutils operations used by bake(), backed by NumPy/SciPy."""
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float).copy()

    def __array__(self, dtype=None, copy=None):
        result = np.asarray(self.values, dtype=dtype)
        return result.copy() if copy else result

    def __matmul__(self, other):
        array = np.asarray(other)
        if self.values.shape == (4, 4) and array.shape == (3,):
            return (self.values @ np.r_[array, 1.])[:3]
        result = self.values @ array
        return Matrix(result) if isinstance(other, Matrix) else result

    def identity(self):
        self.values[:] = np.eye(len(self.values))

    def to_quaternion(self):
        xyzw = Rotation.from_matrix(self.values[:3, :3]).as_quat()
        return SimpleNamespace(normalized=lambda: xyzw[[3, 0, 1, 2]])


class Bones(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            return next(b for b in self if b.name == key)
        return super().__getitem__(key)


def bake_fixture(folder, count):
    source_folder = folder / 'source'
    source_folder.mkdir()
    bones = Bones()

    def bone(name, parent=None):
        head = np.array([len(bones) * .02, 0., 0.])
        if name in ('全ての親', '腰', PMX[0]):
            head[:] = 0.
        matrix = np.eye(4)
        matrix[:3, 3] = head
        result = SimpleNamespace(name=name, parent=parent, head_local=head,
            tail_local=head + [0., 1., 0.], matrix_local=Matrix(matrix), length=1.)
        bones.append(result)
        return result

    root = bone('全ての親')
    waist = bone('腰', root)
    mapped = []
    for i, name in enumerate(PMX):
        mapped.append(bone(name, waist if i == 0 else mapped[PARENTS[i]]))
    source_names = list(NAMES)
    finger_names = []
    for side, suffix, hand in [('Left', '.L', 20), ('Right', '.R', 21)]:
        parent = mapped[hand]
        for index in range(1, 4):
            source_names.append(side + 'HandIndex' + str(index))
            name = '人指' + '０１２３'[index] + suffix
            parent = bone(name, parent)
            finger_names.append(name)
    pose = Bones()
    for b in bones:
        pose.append(SimpleNamespace(name=b.name, matrix_basis=Matrix(np.eye(4)), constraints=[],
            keyframe_insert=lambda *args, **kwargs: None,
            path_from_id=lambda prop, name=b.name: f'pose.bones["{name}"].{prop}'))
    arm = SimpleNamespace(matrix_world=Matrix(np.eye(4)), data=SimpleNamespace(bones=bones),
        pose=SimpleNamespace(bones=pose), animation_data=SimpleNamespace(action=SimpleNamespace(fcurves=[])),
        animation_data_clear=lambda: None)
    scene = SimpleNamespace(render=SimpleNamespace(), frame_set=lambda n: None)
    bpy = module('bpy', data=SimpleNamespace(objects={'Armature': arm}),
        context=SimpleNamespace(scene=scene), ops=SimpleNamespace(wm=SimpleNamespace(open_mainfile=lambda **kw: None)))
    theta = np.linspace(170., 190., count)
    body_angles = theta[:, None] + 3. * np.arange(22)
    world = Rotation.from_euler('y', body_angles.reshape(-1, 1), degrees=True).as_matrix().reshape(count, 22, 3, 3)
    trajectory = np.arange(count)[:, None] * [.12, .02, -.04] + [1., 2., 3.]
    unused = np.array([{'unused': True}], dtype=object)
    np.savez(folder / 'motion-pmx-stance.npz', root=trajectory, names=PMX, world=world, unused=unused)
    np.savez(folder / 'motion-pmx-rotation.npz', world=np.tile(np.eye(3), (count, 22, 1, 1)), unused=unused)
    np.savez(folder / 'motion-source.npz', fps=30., unused=unused)
    rest = np.zeros((len(source_names), 3))
    rest[:, 2] = -np.arange(len(source_names))
    ends = np.tile([0., 0., -1.], (len(source_names), 1))
    np.savez(source_folder / 'source-asset.npz', rest=rest, ends=ends, unused=unused)
    frames = np.tile(np.eye(4), (count, len(source_names), 1, 1))
    finger_angles = []
    for index in range(6):
        angles = np.linspace(5. + index, 25. + index * 2, count)
        finger_angles.append(angles)
        frames[:, 22 + index, :3, :3] = Rotation.from_euler('x', angles[:, None], degrees=True).as_matrix()
    np.save(source_folder / 'source-frames.npy', frames)
    (folder / 'pmx-rig.json').write_text(json.dumps({'armature': 'Armature'}), encoding='utf-8')
    (folder / 'sources.json').write_text(json.dumps([{'label': 'motion', 'folder': str(source_folder),
        'info': {'names': source_names, 'frames_file': 'source-frames.npy', 'asset_file': 'source-asset.npz'}}]), encoding='utf-8')
    spec = importlib.util.spec_from_file_location('mocap.retarget._test_blender_bake', ROOT / 'mocap/retarget/blender_bake.py')
    bake = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'bpy': bpy, 'mathutils': module('mathutils', Matrix=Matrix)}):
        spec.loader.exec_module(bake)
    bake.OUT = folder
    recorded = {}
    bake.key_values = lambda arm, path, values: recorded.__setitem__(path, values.copy())
    bake.save_blend = lambda path: None
    return bake, recorded, scene, trajectory, theta, finger_names, finger_angles


class RetargetArrayReadTests(unittest.TestCase):
    def test_source_static_asset_is_read_once_and_heights_are_unchanged(self):
        for count in (4, 17):
            with self.subTest(frames=count), tempfile.TemporaryDirectory() as temporary:
                project, output, stubs, heights, frames = source_fixture(Path(temporary), count)
                with source_stubs(stubs), ArrayReads() as observed:
                    receipt = source.prepare(project, 'initial', output)
                    self.assertEqual(observed.reads['mesh-asset.npz', 'packed'], 1)
                    self.assertEqual(observed.reads['mesh-asset.npz', 'vertex_count'], 1)
                    self.assertNotIn(('mesh-asset.npz', 'unused'), observed.reads)
                    observed.assert_closed(self)
                self.assertEqual(receipt['frames'], count)
                self.assertEqual(receipt['fps'], 30.)
                with np.load(output / 'motion-source.npz') as actual:
                    np.testing.assert_allclose(actual['support_height'], heights, rtol=0, atol=1e-7)
                    np.testing.assert_array_equal(actual['world'], frames[:, :, :3, :3])

    def test_source_sampling_error_closes_mesh_archive_and_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            project, output, stubs, _, _ = source_fixture(Path(temporary), 4)
            original = mesh_data.reference_vertices
            calls = 0

            def interrupted(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('surface sampling interrupted')
                return original(*args)

            with source_stubs(stubs), ArrayReads() as observed, patch.object(mesh_data, 'reference_vertices', interrupted):
                with self.assertRaisesRegex(RuntimeError, 'surface sampling interrupted'):
                    source.prepare(project, 'initial', output)
                observed.assert_closed(self, {'mesh-asset.npz', 'mesh-frames.npy'})
                self.assertEqual(observed.reads['mesh-asset.npz', 'packed'], 1)

    def test_bake_reads_selected_fields_once_and_preserves_motion(self):
        for count in (4, 17):
            with self.subTest(frames=count), tempfile.TemporaryDirectory() as temporary:
                bake, keyed, scene, trajectory, theta, fingers, angles = bake_fixture(Path(temporary), count)
                with ArrayReads() as observed:
                    record = bake.key_values

                    def record_after_close(*args):
                        observed.assert_closed(self, {'source-frames.npy'})
                        record(*args)

                    bake.key_values = record_after_close
                    _, names = bake.bake('pmx', 'motion', 'stance')
                    for filename, keys in [('motion-pmx-stance.npz', ('root', 'names', 'world')),
                                           ('motion-pmx-rotation.npz', ('world',)),
                                           ('source-asset.npz', ('rest', 'ends')),
                                           ('motion-source.npz', ('fps',))]:
                        for key in keys:
                            self.assertEqual(observed.reads[filename, key], 1, (filename, key))
                    self.assertFalse(any(key == 'unused' for filename, key in observed.reads))
                    observed.assert_closed(self)
                np.testing.assert_array_equal(keyed['pose.bones["全ての親"].location'], trajectory @ bake.A.T)
                waist = keyed['pose.bones["腰"].rotation_quaternion']
                np.testing.assert_allclose(Rotation.from_quat(waist[:, [1, 2, 3, 0]]).as_matrix(),
                    Rotation.from_euler('z', theta[:, None], degrees=True).as_matrix(), atol=1e-12)
                self.assertTrue(np.all(np.sum(waist[:-1] * waist[1:], axis=1) >= 0.))
                for index, name in enumerate(PMX):
                    values = keyed[f'pose.bones["{name}"].rotation_quaternion']
                    relative = 0. if index == 0 else 3. * (index - PARENTS[index])
                    expected = Rotation.from_euler('z', np.full((count, 1), relative), degrees=True).as_matrix()
                    np.testing.assert_allclose(Rotation.from_quat(values[:, [1, 2, 3, 0]]).as_matrix(), expected, atol=1e-12)
                for index, name in enumerate(fingers):
                    values = keyed[f'pose.bones["{name}"].rotation_quaternion']
                    expected = angles[index] if index % 3 == 0 else angles[index] - angles[index - 1]
                    np.testing.assert_allclose(Rotation.from_quat(values[:, [1, 2, 3, 0]]).as_matrix(),
                        Rotation.from_euler('x', expected[:, None], degrees=True).as_matrix(), atol=1e-12)
                    self.assertIn(name, names)
                self.assertEqual((scene.frame_start, scene.frame_end), (1, count))
                self.assertEqual(scene.render.fps / scene.render.fps_base, 30.)

    def test_bake_finger_failure_closes_all_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            bake, *_ = bake_fixture(Path(temporary), 4)
            original = bake.align
            calls = 0

            def interrupted(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('finger bake interrupted')
                return original(*args)

            with ArrayReads() as observed, patch.object(bake, 'align', interrupted):
                with self.assertRaisesRegex(RuntimeError, 'finger bake interrupted'):
                    bake.bake('pmx', 'motion', 'stance')
                observed.assert_closed(self)


if __name__ == '__main__':
    unittest.main()
