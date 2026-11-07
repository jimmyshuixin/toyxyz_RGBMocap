"""CPU checks for upstream PMX contact-geometry registration.

Uses synthetic meshes and real NPZ files. Blender API stand-ins exercise the
production weight registration and parent/rest-space bake arithmetic.
"""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'tests'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_longvideo_retarget import Bones, Matrix, module
from mocap.retarget.mapping import A, PARENTS, PMX


class Vertices(list):
    def foreach_get(self, prop, target):
        assert prop == 'co'
        target[:] = np.array([v.co for v in self]).ravel()


def geometry_fixture():
    names = list(PMX)
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
    bones, by_slot = Bones(), {}
    for slot, name in enumerate(names):
        parent = int(PARENTS[slot])
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
    rig = {}
    return geometry, rig, arm, names, rest


class PmxUpstreamGeometryTests(unittest.TestCase):
    def test_contact_weights_keep_upstream_d_and_ex_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            geometry, rig, arm, names, rest = geometry_fixture()
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


if __name__ == '__main__':
    unittest.main()
