"""Output heading checks; no inference, source-cache writes or Blender required."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mocap import motion_heading as heading
from mocap.retarget import worker


def placement():
    """A tilted frame with translation; it does not commute with a yaw turn."""
    angle = .43
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.eye(4)
    matrix[:3, :3] = [[1., 0., 0.], [0., c, -s], [0., s, c]]
    matrix[:3, 3] = [2.3, .7, -4.1]
    return matrix


def expected_points(points):
    result = np.array(points, dtype=float, copy=True)
    result[..., [0, 2]] *= -1
    return result


class MotionHeadingTests(unittest.TestCase):
    def test_capture_forward_becomes_blender_negative_y_without_mirroring(self):
        forward = heading.output_points([0., 0., -1.])
        # Independently spell out app Y-up -> Blender Z-up, rather than use
        # the same application mapping matrix as the implementation.
        blender_forward = [forward[0], -forward[2], forward[1]]
        np.testing.assert_array_equal(blender_forward, [0., -1., 0.])
        np.testing.assert_array_equal(heading.output_points([0., 1., 0.]), [0., 1., 0.])
        np.testing.assert_allclose(heading.ROTATION.T @ heading.ROTATION, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(heading.ROTATION), 1.)
        np.testing.assert_array_equal(heading.TRANSFORM[:3, :3], heading.ROTATION)
        np.testing.assert_array_equal(heading.TRANSFORM[3], [0., 0., 0., 1.])

    def test_heading_follows_shared_placement_and_preserves_inputs(self):
        native = placement()
        saved = native.copy()
        points = np.array([[.2, .8, -.3], [-.7, .1, 2.]])
        saved_points = points.copy()
        output = heading.output_transform(native)
        actual = points @ output[:3, :3].T + output[:3, 3]
        placed = points @ native[:3, :3].T + native[:3, 3]
        np.testing.assert_allclose(actual, expected_points(placed), rtol=0, atol=1e-14)
        self.assertFalse(np.allclose(output, native @ heading.TRANSFORM))
        np.testing.assert_array_equal(native, saved)
        np.testing.assert_array_equal(points, saved_points)
        np.testing.assert_array_equal(heading.output_transform(), heading.TRANSFORM)

    def test_full_scene_turn_preserves_relative_motion_and_floor_heights(self):
        # Two actors and three times share one world, including a stance foot.
        points = np.array([[[1., 1., -2.], [3., 1.2, -4.], [.8, .02, -2.]],
                           [[1.2, 1.05, -2.2], [2.8, 1.3, -4.4], [.8, .02, -2.]],
                           [[1.5, 1.1, -2.3], [2.6, 1.1, -4.5], [.8, .02, -2.]]])
        original = points.copy()
        turned = heading.output_points(points)
        np.testing.assert_allclose(np.linalg.norm(turned[:, 0] - turned[:, 1], axis=1),
                                   np.linalg.norm(points[:, 0] - points[:, 1], axis=1))
        np.testing.assert_array_equal(turned[..., 1], points[..., 1])
        np.testing.assert_array_equal(np.diff(turned[:, 2], axis=0), 0.)
        np.testing.assert_allclose(np.diff(turned, axis=0), expected_points(np.diff(points, axis=0)))
        np.testing.assert_array_equal(points, original)

    def test_parent_child_local_rotation_and_translation_are_preserved(self):
        parent = placement()
        local = np.eye(4)
        local[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        local[:3, 3] = [.4, -.25, .1]
        child = parent @ local
        turned_parent = heading.output_transform(parent)
        turned_child = heading.output_transform(child)
        np.testing.assert_allclose(np.linalg.inv(turned_parent) @ turned_child, local, atol=1e-14)

    def test_render_camera_matches_rotated_geometry_and_projection(self):
        points = np.array([[.3, .8, 2.], [-1., .2, -.7], [2., 1.2, 1.4]])
        eye = np.array([4., 2.3, -3.])
        basis = placement()[:3, :3]
        eye_before, basis_before = eye.copy(), basis.copy()
        source_eye, source_basis = heading.source_camera(eye, basis)
        source_view = (points - source_eye) @ source_basis.T
        output_view = (heading.output_points(points) - eye) @ basis.T
        np.testing.assert_allclose(source_view, output_view, atol=1e-14)
        # The same basis change applied to a world-to-camera matrix leaves
        # camera coordinates (and consequently video projection) unchanged.
        extrinsic = placement()
        output_extrinsic = extrinsic @ np.linalg.inv(heading.TRANSFORM)
        homogeneous = np.column_stack((points, np.ones(len(points))))
        turned = np.column_stack((heading.output_points(points), np.ones(len(points))))
        np.testing.assert_allclose(turned @ output_extrinsic.T,
                                   homogeneous @ extrinsic.T, atol=1e-14)
        np.testing.assert_array_equal(eye, eye_before)
        np.testing.assert_array_equal(basis, basis_before)

    def test_character_heading_applies_only_to_captured_body_takes(self):
        native = placement().tolist()
        captured = dict(source='source-project', transform=native)
        saved = copy.deepcopy(captured)
        np.testing.assert_allclose(heading.character_transform(captured), heading.output_transform(native))
        np.testing.assert_array_equal(heading.character_transform(dict(source='source-project')), heading.TRANSFORM)
        np.testing.assert_array_equal(heading.character_transform(dict(transform=native)), native)
        self.assertIsNone(heading.character_transform({}))
        self.assertEqual(captured, saved)


class CharacterHeadingBoundaryTests(unittest.TestCase):
    def test_character_preview_rotates_capture_once_and_keeps_static_pmx(self):
        try:
            from mocap.character_ui import CharacterWorkspace
        except ImportError:
            from mocap.runtime_registry import activate
            activate(ROOT, 'gui')
            from mocap.character_ui import CharacterWorkspace
        base = Path('heading-test-virtual-assets')
        native = placement().tolist()
        common = dict(folder='take-0', start=0., frames=10, fps=30., transform=native)
        characters = [dict(id='rest', name='Rest', asset='rest/asset'),
                      dict(id='face', name='Face only', asset='face/asset', animation='face/animation',
                           takes=[dict(common)]),
                      dict(id='capture', name='Capture', asset='capture/asset', animation='capture/animation',
                           takes=[dict(common, source='native-source-project')])]
        matrices = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
        paths = [base / 'rest/asset', base / 'face/animation/take-0', base / 'capture/animation/take-0']
        requested = []
        workspace = SimpleNamespace(characters=characters, motion_view=SimpleNamespace(scene_time=.1),
                                    character_path=lambda relative: base / relative,
                                    character_assets={str(path): {'asset': {}} for path in paths},
                                    character_loader=SimpleNamespace(frame=lambda path, frame: matrices,
                                                                     request=lambda *args: requested.append(args)))
        draws = list(CharacterWorkspace.character_draws(workspace))
        self.assertEqual(len(draws), 3)
        self.assertEqual(requested, [])
        self.assertIsNone(draws[0][-1])
        np.testing.assert_allclose(draws[1][-1], native)
        np.testing.assert_allclose(draws[2][-1], heading.output_transform(native))
        self.assertEqual(characters[2]['takes'][0]['transform'], native)
        for draw in draws:
            np.testing.assert_array_equal(draw[2], matrices)

    def test_character_export_matches_preview_without_rewriting_takes(self):
        for fmt in ('fbx', 'vmd'):
            with self.subTest(format=fmt), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                native = placement().tolist()
                takes = [dict(folder='capture', source='native-source-project', transform=native),
                         dict(folder='face', transform=native)]
                job = dict(mode='export', output=str(root / 'job-output'), animation=str(root / 'animation'),
                           takes=takes, files=[str(root / f'capture.{fmt}'), str(root / f'face.{fmt}')],
                           source_name='Character', format=fmt)
                request = root / 'request.json'
                original = json.dumps(job)
                request.write_text(original, encoding='utf-8')
                with patch.object(worker, 'invoke') as invoked, patch.object(worker, 'report'):
                    worker.main(['--request', str(request)])
                self.assertEqual(invoked.call_count, 2)
                capture, face = [call.args for call in invoked.call_args_list]
                self.assertEqual(capture[0], 'export')
                np.testing.assert_allclose(json.loads(capture[4]), heading.output_transform(native))
                np.testing.assert_allclose(json.loads(face[4]), native)
                self.assertEqual(json.loads(capture[6])['format'], fmt)
                self.assertEqual(request.read_text(encoding='utf-8'), original)
                self.assertTrue((root / 'job-output/complete.json').is_file())


if __name__ == '__main__':
    unittest.main()
