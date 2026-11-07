"""CPU checks for native PMX bones at the geometry/bake boundary.

Uses synthetic meshes and real NPZ files. Blender API stand-ins exercise the
production weight registration and parent/rest-space bake arithmetic.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'tests'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_longvideo_retarget import ArrayReads, Bones, Matrix, bake_fixture, module
from mocap.retarget.mapping import A, PARENTS, PMX, VIRTUAL_SPINE1


def two_spine_bindings():
    names = list(PMX)
    names[6] = VIRTUAL_SPINE1
    profile = {'version': 1, 'id': 'pmx_two_spine_v1',
               'slot_bones': [None if i == 6 else name for i, name in enumerate(names)],
               'virtual_slots': [6]}
    return names, profile


class Vertices(list):
    def foreach_get(self, prop, target):
        assert prop == 'co'
        target[:] = np.array([v.co for v in self]).ravel()


def geometry_fixture(two_spine):
    names, profile = two_spine_bindings() if two_spine else (list(PMX), None)
    rest = np.zeros((22, 3))
    rest[:, 1] = 1.
    for slot, height in ((3, 1.2), (6, 1.4), (9, 1.5), (12, 1.65), (15, 1.8)):
        rest[slot, 1] = height
    for side, sign in ((0, -1.), (1, 1.)):
        rest[1 + side] = [sign * .12, .9, 0.]
        rest[4 + side] = [sign * .12, .5, 0.]
        rest[7 + side] = [sign * .12, .1, 0.]
        rest[10 + side] = [sign * .12, .05, .2]
        rest[20 + side] = [sign * .55, 1.2, 0.]
    if two_spine:
        rest[6] = rest[9]
    bones, by_slot = Bones(), {}
    for slot, name in enumerate(names):
        if two_spine and slot == 6:
            continue
        parent = int(PARENTS[slot])
        if two_spine and parent == 6:
            parent = 3
        bone = SimpleNamespace(name=name, parent=by_slot.get(parent), head_local=rest[slot] @ A.T)
        bones.append(bone)
        by_slot[slot] = bone
    vertices, groups = Vertices(), []

    def vertex(position, weights):
        memberships = []
        for name, weight in weights:
            if name not in groups:
                groups.append(name)
            memberships.append(SimpleNamespace(group=groups.index(name), weight=weight))
        vertices.append(SimpleNamespace(index=len(vertices), co=position @ A.T, groups=memberships))

    for side, suffix in enumerate(('.L', '.R')):
        aliases = [('足D' + suffix, .1), ('ひざD' + suffix, .2),
                   ('足首D' + suffix, .3), ('足先EX' + suffix, .4)]
        # Their actual parents deliberately differ from their canonical aliases.
        for name, _ in aliases:
            bones.append(SimpleNamespace(name=name, parent=by_slot[0], head_local=np.zeros(3)))
        for x in (-.02, 0., .02):
            for z in (-.06, -.03, .03, .09, .15):
                vertex(np.array([rest[7 + side, 0] + x, 0., z]), aliases)
        for angle in np.linspace(0., 2 * np.pi, 8, endpoint=False):
            position = rest[20 + side] + [.03 * np.cos(angle), 0., .03 * np.sin(angle)]
            vertex(position, [(names[20 + side], 1.)])
    mesh = SimpleNamespace(type='MESH', name='mesh', matrix_world=Matrix(np.eye(4)),
                           data=SimpleNamespace(vertices=vertices),
                           vertex_groups=[SimpleNamespace(name=name) for name in groups])
    arm = SimpleNamespace(type='ARMATURE', matrix_world=Matrix(np.eye(4)), data=SimpleNamespace(bones=bones))
    bpy = module('bpy', context=SimpleNamespace(scene=SimpleNamespace(objects=[arm, mesh])))
    spec = importlib.util.spec_from_file_location('mocap.retarget._test_pmx_geometry',
                                                 ROOT / 'mocap/retarget/blender_geometry.py')
    geometry = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'bpy': bpy}):
        spec.loader.exec_module(geometry)
    rig = {'body_profile': profile} if profile is not None else {}
    return geometry, rig, arm, names, rest


def two_spine_bake_fixture(folder, count=5):
    fixture = bake_fixture(folder, count)
    bake = fixture[0]
    arm = bake.bpy.data.objects['Armature']
    removed = arm.data.bones[PMX[6]]
    for bone in arm.data.bones:
        if bone.parent is removed:
            bone.parent = removed.parent
    arm.data.bones[:] = [bone for bone in arm.data.bones if bone is not removed]
    arm.pose.bones[:] = [bone for bone in arm.pose.bones if bone.name != PMX[6]]
    names, profile = two_spine_bindings()
    path = folder / 'motion-pmx-stance.npz'
    with np.load(path) as data:
        root, world = data['root'], data['world']
    # Noncommuting rotations distinguish real-parent composition from dropping
    # one source rotation or adding angles. Slot 6 must not be keyed at all.
    world[:, 3] = Rotation.from_euler('x', np.linspace(10., 40., count)[:, None], degrees=True).as_matrix()
    world[:, 6] = Rotation.from_euler('z', np.linspace(-150., 130., count)[:, None], degrees=True).as_matrix()
    world[:, 9] = Rotation.from_euler('yz', np.column_stack((np.linspace(-25., 30., count),
                                                           np.linspace(35., -15., count))), degrees=True).as_matrix()
    np.savez(path, root=root, world=world, names=names)
    (folder / 'pmx-rig.json').write_text(json.dumps({'armature': 'Armature', 'body_profile': profile}), encoding='utf-8')
    chest = arm.data.bones[PMX[9]]
    chest_rest = Rotation.from_euler('xyz', [23., -17., 41.], degrees=True).as_matrix()
    rest_matrix = np.asarray(chest.matrix_local).copy()
    rest_matrix[:3, :3] = chest_rest
    chest.matrix_local = Matrix(rest_matrix)
    return fixture, arm, world, chest_rest


class PmxNativeBoneBoundaryTests(unittest.TestCase):
    def test_contact_weights_keep_native_aliases_and_zero_virtual_column(self):
        for two_spine in (False, True):
            with self.subTest(two_spine=two_spine), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                geometry, rig, arm, names, rest = geometry_fixture(two_spine)
                geometry.register(folder, 'pmx', rig, arm, names, rest)
                with np.load(folder / 'pmx-contact-geometry.npz') as registered:
                    for side in (0, 1):
                        expected = np.zeros(22)
                        expected[[1 + side, 4 + side, 7 + side, 10 + side]] = [.1, .2, .3, .4]
                        foot = registered[f'foot_w{side}']
                        self.assertEqual(len(foot), 15)
                        np.testing.assert_allclose(foot, np.tile(expected, (len(foot), 1)))
                        palm = registered[f'palm_w{side}']
                        self.assertEqual(len(palm), 8)
                        np.testing.assert_array_equal(palm[:, 20 + side], 1.)
                        for weights in (foot, palm):
                            np.testing.assert_allclose(weights.sum(1), 1.)
                            np.testing.assert_array_equal(weights[:, 6], 0.)

    def test_bake_uses_actual_parent_and_rest_axes_without_virtual_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture, arm, world, chest_rest = two_spine_bake_fixture(Path(temporary))
            bake, keyed, scene, _, _, fingers, _ = fixture
            with ArrayReads() as observed:
                _, names = bake.bake('pmx', 'motion', 'stance')
                observed.assert_closed(self)
                for member in ('root', 'names', 'world'):
                    self.assertEqual(observed.reads['motion-pmx-stance.npz', member], 1)
            self.assertEqual(set(names), (set(PMX) - {PMX[6]}) | set(fingers) | {'腰', '全ての親'})
            self.assertTrue(set(names).issubset({bone.name for bone in arm.data.bones}))
            self.assertFalse(any(VIRTUAL_SPINE1 in path or PMX[6] in path for path in keyed))
            chest_keys = keyed[f'pose.bones["{PMX[9]}"].rotation_quaternion']
            local = Rotation.from_quat(chest_keys[:, [1, 2, 3, 0]]).as_matrix()
            # Evaluate the native two-bone chain independently of the canonical
            # parent table: UpperBody -> UpperBody2, with UpperBody2's rest axes.
            actual = (A @ world[:, 3] @ A.T) @ chest_rest @ local
            desired = (A @ world[:, 9] @ A.T) @ chest_rest
            np.testing.assert_allclose(actual, desired, atol=1e-12)
            self.assertEqual((scene.frame_start, scene.frame_end), (1, len(world)))
            self.assertEqual(scene.render.fps / scene.render.fps_base, 30.)

    def test_geometry_and_bake_reject_missing_real_bones(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            geometry, rig, arm, names, rest = geometry_fixture(True)
            arm.data.bones[:] = [bone for bone in arm.data.bones if bone.name != PMX[12]]
            with self.assertRaisesRegex(ValueError, 'Character bones changed after import: 首'):
                geometry.register(folder, 'pmx', rig, arm, names, rest)
            self.assertFalse((folder / 'pmx-contact-geometry.npz').exists())
            fixture, arm, _, _ = two_spine_bake_fixture(folder)
            arm.data.bones[:] = [bone for bone in arm.data.bones if bone.name != PMX[12]]
            with self.assertRaisesRegex(ValueError, 'Character bones changed after import: 首'):
                fixture[0].bake('pmx', 'motion', 'stance')
            self.assertEqual(fixture[1], {})

    def test_virtual_names_require_persisted_profile_in_both_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            geometry, _, arm, names, rest = geometry_fixture(True)
            with self.assertRaisesRegex(ValueError, 'explicit body profile'):
                geometry.register(folder, 'pmx', {}, arm, names, rest)
            fixture, _, _, _ = two_spine_bake_fixture(folder)
            (folder / 'pmx-rig.json').write_text(json.dumps({'armature': 'Armature'}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'explicit body profile'):
                fixture[0].bake('pmx', 'motion', 'stance')
            self.assertEqual(fixture[1], {})


if __name__ == '__main__':
    unittest.main()
