"""CPU regressions for upstream whole-take retarget surface evaluation.

The reference evaluates the complete take with the original LBS equations.
Fixtures use synthetic skeletons and surfaces, without user/model assets.
"""
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap.retarget import solver


def fixture(count, vertices=67, dtype=np.float64):
    rng = np.random.default_rng(20261107)
    rest = rng.normal(size=(22, 3)).astype(dtype)
    local = Rotation.from_rotvec(rng.normal(0, .12, size=(count * 22, 3))).as_matrix().reshape(count, 22, 3, 3).astype(dtype)
    root = rng.normal(size=(count, 3)).astype(dtype)
    points, world = solver.fk(local, rest, root)
    data = dict(rest=rest, local=local, root=root, points=points, world=world, ratio=1., names=np.array([f'joint_{j}' for j in range(22)]))
    geo = {}
    for side in (0, 1):
        for name, size in (('foot', vertices + side), ('palm', 37 + side)):
            v = rng.normal(size=(size, 3)).astype(dtype)
            w = rng.uniform(size=(size, 22)).astype(dtype)
            w /= w.sum(1, keepdims=True)
            geo[f'{name}_v{side}'], geo[f'{name}_w{side}'] = v, w
        geo[f'heel{side}'] = np.arange(0, vertices // 2)
        geo[f'toe{side}'] = np.arange(vertices // 2, vertices + side)
    palms = [np.array([2, 4, 5, 9]), np.array([1, 3, 6, 10, 13])]
    return data, geo, palms


def dense_skin(data, vertices, weights):
    weighted = (vertices[None] - data['rest'][:, None]) * weights.T[:, :, None]
    return (np.einsum('tjik,jvk->tvi', data['world'], weighted, optimize=True)
            + np.einsum('vj,tji->tvi', weights, data['points'], optimize=True))


def dense_samples(data, geo, palms):
    probes, feet, hands = [], [], []
    for side in (0, 1):
        points = dense_skin(data, geo[f'foot_v{side}'], geo[f'foot_w{side}'])
        feet.append(points)
        for ids in (geo[f'heel{side}'], geo[f'toe{side}']):
            part = points[:, ids]
            sample = part.mean(1)
            sample[:, 1] = part[:, :, 1].min(1)
            probes.append(sample)
        palm = dense_skin(data, geo[f'palm_v{side}'][palms[side]], geo[f'palm_w{side}'][palms[side]])
        hands.append(palm.mean(1))
    return np.stack(probes, 1), feet, np.stack(hands, 1)


def dense_pins(data, geo, plan):
    probes, feet, hands = dense_samples(data, geo, plan['palms'])
    pins = solver.trajectory.support_samples(probes, plan['weights'])
    pins[:, :, 1] = np.stack([foot[:, :, 1].min(1) for foot in feet], axis=1)
    result = np.concatenate((pins, hands), axis=1)
    if 'skin_bias' in plan:
        result += plan['skin_bias']
    return result, feet


def contact_fixture(count):
    """A coherent human-sized rig with sparse ankle/toe and wrist skinning."""
    data, geo, palms = fixture(count, vertices=31)
    rest = np.array([
        [0, 1., 0], [.10, .95, 0], [-.10, .95, 0], [0, 1.08, 0],
        [.10, .55, .02], [-.10, .55, .02], [0, 1.22, .005],
        [.10, .12, 0], [-.10, .12, 0], [0, 1.38, .01],
        [.10, .04, .18], [-.10, .04, .18], [0, 1.56, 0],
        [.08, 1.49, 0], [-.08, 1.49, 0], [0, 1.69, 0],
        [.21, 1.48, 0], [-.21, 1.48, 0], [.48, 1.28, .01],
        [-.48, 1.28, .01], [.69, 1.08, 0], [-.69, 1.08, 0]])
    phase = np.linspace(0, 2 * np.pi, count)
    local = np.broadcast_to(np.eye(3), (count, 22, 3, 3)).copy()
    for joint, sign in ((4, 1), (5, 1), (18, 1), (19, -1)):
        axis = np.zeros((count, 3))
        axis[:, 0 if joint < 6 else 2] = sign * (.15 + .02 * np.sin(phase))
        local[:, joint] = Rotation.from_rotvec(axis).as_matrix()
    root = np.broadcast_to(rest[0], (count, 3)).copy()
    root[:, 0] += .02 * np.sin(phase)
    points, world = solver.fk(local, rest, root)
    data.update(rest=rest, local=local, root=root, points=points, world=world)
    for side in (0, 1):
        count_v = len(geo[f'foot_v{side}'])
        vertices = np.zeros((count_v, 3))
        vertices[:, 0] = rest[7 + side, 0] + .03 * np.cos(np.arange(count_v))
        vertices[:, 1] = .01 + .002 * np.sin(np.arange(count_v))
        vertices[:, 2] = np.linspace(-.03, .22, count_v)
        weights = np.zeros((count_v, 22))
        weights[:, 7 + side], weights[:, 10 + side] = .8, .2
        geo[f'foot_v{side}'], geo[f'foot_w{side}'] = vertices, weights
        geo[f'palm_v{side}'] = rest[20 + side] + geo[f'palm_v{side}'] * .02
        geo[f'palm_w{side}'][:] = 0
        geo[f'palm_w{side}'][:, 20 + side] = 1
    return data, geo, palms


class UpstreamRetargetGeometryTests(unittest.TestCase):
    def assert_tree_close(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_tree_close(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.assert_tree_close(a, e)
        elif isinstance(expected, np.ndarray):
            if expected.dtype.kind in 'biufc':
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
            else:
                np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, (float, np.floating)):
            self.assertAlmostEqual(actual, expected, places=10)
        else:
            self.assertEqual(actual, expected)

    def test_all_vertices_and_patch_reductions_match_complete_take(self):
        for count in (1, 4, 127, 128, 129, 257):
            for dtype in (np.float32, np.float64):
                with self.subTest(count=count, dtype=dtype):
                    data, geo, palms = fixture(count, dtype=dtype)
                    before = deepcopy(data)
                    expected = dense_samples(data, geo, palms)
                    actual = solver.geometry_samples(data, geo, palms)
                    self.assert_tree_close(actual, expected)
                    for key in before:
                        np.testing.assert_array_equal(data[key], before[key])

    def test_palm_selection_uses_first_contact_frame(self):
        data, geo, palms = fixture(257)
        active = np.zeros((257, 4), dtype=bool)
        active[149:181, 2] = True
        active[:70, 0] = True
        expected = []
        for side, first in ((0, 149), (1, 0)):
            full = dense_skin(data, geo[f'palm_v{side}'], geo[f'palm_w{side}'])
            expected.append(np.argsort(full[first, :, 1])[:max(4, min(12, len(full[first]) // 4))])
        plan = solver.setup_targets(data, deepcopy(data), active, geo)
        for actual, selected in zip(plan['palms'], expected):
            np.testing.assert_array_equal(actual, selected)

    def test_pin_bias_and_all_score_diagnostics_match_dense_reference(self):
        data, geo, palms = fixture(257)
        rng = np.random.default_rng(4401)
        active = rng.uniform(size=(257, 4)) > .35
        active[:, 3] = False
        plan = dict(palms=palms, weights=rng.uniform(size=(257, 2)),
                    goals=rng.normal(size=(257, 4, 3)), skin_bias=rng.normal(0, .001, size=(257, 4, 3)))
        actual = solver.eval_pins(data, geo, plan)
        self.assert_tree_close(actual, dense_pins(data, geo, plan))
        with patch.object(solver, 'eval_pins', side_effect=dense_pins):
            expected_score = solver.score(data, data, active, geo, plan)
        score = solver.score(data, data, active, geo, plan)
        self.assertEqual(score, expected_score)

    def test_full_refine_and_score_match_dense_geometry(self):
        data, geo, _ = contact_fixture(129)
        active = np.zeros((129, 4), dtype=bool)
        active[7:56, 0] = True
        active[42:99, 1] = True

        # Exercise the complete solver, including FK release, iterative free
        # clearance, contact planning and early-stop decisions in both runs.
        with patch.dict(os.environ, {'STOP_STAGNATION': '1'}):
            with patch.object(solver, 'geometry_samples', side_effect=dense_samples):
                reference = solver.refine(deepcopy(data), deepcopy(data), active, geo)
                score = solver.score(reference[0], data, active, geo, reference[1])
            actual = solver.refine(deepcopy(data), deepcopy(data), active, geo)
            actual_score = solver.score(actual[0], data, active, geo, actual[1])
        self.assert_tree_close(actual, reference)
        self.assert_tree_close(actual_score, score)


if __name__ == '__main__':
    unittest.main()
