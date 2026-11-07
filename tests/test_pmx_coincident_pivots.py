"""Regression for native co-located PMX torso pivots with float roundoff."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mocap.retarget import solver


class CoincidentPivotTests(unittest.TestCase):
    def test_zero_direction_has_no_rotation_constraint(self):
        for tiny in (0., np.finfo(float).eps * 2, 1e-13):
            for reverse in (False, True):
                a, b = np.array([0., tiny, 0.]), np.array([.1, .2, .3])
                if reverse:
                    a, b = b, a
                np.testing.assert_array_equal(solver.align(a, b), np.eye(3))

    def test_ordinary_and_antiparallel_vectors_keep_rigid_alignment(self):
        a = np.array([.2, -.4, .7])
        for b in (np.array([.5, .3, -.1]), -a, a):
            rotation = solver.align(a, b)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(rotation), 1., places=14)
            np.testing.assert_allclose(rotation @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-14)

    def test_repeated_fk_preserves_lengths_for_a_coincident_hip_spine(self):
        rng = np.random.default_rng(20261107)
        source_rest = rng.normal(size=(22, 3)) * .1
        target_rest = source_rest.copy()
        target_rest[3] = target_rest[0] + [0., np.finfo(float).eps * 2, 0.]
        source_world = Rotation.from_rotvec(rng.normal(0., .2, size=(5 * 22, 3))).as_matrix().reshape(5, 22, 3, 3)
        local = solver.baseline(source_world, source_rest, target_rest)
        root = np.zeros((5, 3))
        for _ in range(12):
            points, world = solver.fk(local, target_rest, root)
            local = solver.local_from_world(world)
        expected = np.linalg.norm(target_rest[1:] - target_rest[solver.PARENTS[1:]], axis=1)
        lengths = np.linalg.norm(points[:, 1:] - points[:, solver.PARENTS[1:]], axis=2)
        np.testing.assert_allclose(lengths, np.broadcast_to(expected, lengths.shape), atol=1e-8, rtol=0)
        np.testing.assert_allclose(world.swapaxes(-1, -2) @ world, np.broadcast_to(np.eye(3), world.shape), atol=1e-8, rtol=0)


if __name__ == '__main__':
    unittest.main()
