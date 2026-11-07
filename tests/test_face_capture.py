"""Legacy Reze geometry and native MediaPipe atomic capture regression tests.

Synthetic landmark/video fixtures test contracts, not face-estimation accuracy.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mocap.face.solver import FaceSolver, FaceThresholds, MORPH_NAMES, OneEuroFilter
from mocap.face.blendshapes import BLENDSHAPE_NAMES, scores_from_categories, side_permutation


def native(**values):
    return {'weights': [values.get(name, 0.) for name in BLENDSHAPE_NAMES]}


def face(left_open=1.0, right_open=1.0, mouth=0.0, smile=0.0, gaze=(0.0, 0.0)):
    """Anatomical left/right; EAR .1 closed/.3 open, known target levels."""
    p = [[0.0, 0.0, 0.0] for _ in range(478)]
    for a, b, top, bottom, iris, x, opened in (
        (33, 133, 159, 145, 468, .3, right_open),
        (362, 263, 386, 374, 473, .7, left_open),
    ):
        half_height = (.1 + .2 * opened) * .1 / 2
        p[a], p[b] = [x - .05, .3, 0], [x + .05, .3, 0]
        p[top], p[bottom] = [x, .3 - half_height, 0], [x, .3 + half_height, 0]
        p[iris] = [x + gaze[0] * .05, .3 + gaze[1] * .025, 0]
    h = (.18 + .2 * mouth) * .2
    p[13], p[14] = [.5, .7 - h / 2, 0], [.5, .7 + h / 2, 0]
    corner_y = .7 - (.008 + smile / 120) if smile else .7
    p[61], p[291] = [.4, corner_y, 0], [.6, corner_y, 0]
    return p


class FaceSolverTests(unittest.TestCase):
    def test_analytic_levels_and_anatomical_wink(self):
        solver = FaceSolver(smoothing=False)
        result = solver.solve(face(left_open=0, right_open=1, mouth=.6, smile=.4))
        self.assertTrue(result.detected)
        expected = (.5, 1, 0, .6, .4)
        for n, v in zip(MORPH_NAMES, expected):
            self.assertAlmostEqual(result.morph_weights[n], v, places=12)
        switched = FaceSolver(smoothing=False, mirror=False).solve(face(left_open=0))
        self.assertAlmostEqual(switched.morph_weights["ウィンク右"], 1)

    def test_neutral_and_clamp(self):
        solver = FaceSolver(smoothing=False)
        result = solver.solve(face())
        self.assertTrue(result.detected)
        for v in result.morph_weights.values():
            self.assertAlmostEqual(v, 0, places=12)
        result = solver.solve(face(left_open=-1, right_open=-1, mouth=10, smile=10))
        for n, expected in zip(MORPH_NAMES, (1, 0, 0, 1, 1)):
            self.assertAlmostEqual(result.morph_weights[n], expected)

    def test_filter_reference_and_discontinuity(self):
        f = OneEuroFilter(min_cutoff=2, beta=0)
        self.assertEqual(f.filter(0, 0), 0)
        expected_alpha = 1 / (1 + 30 / (4 * math.pi))
        self.assertAlmostEqual(f.filter(1, 1000 / 30), expected_alpha)
        self.assertEqual(f.filter(.4, 0), .4)
        self.assertEqual(f.filter(.8, 1001), .8)

    def test_missing_nonfinite_and_collapsed_faces_reset_to_rest(self):
        solver = FaceSolver()
        for invalid in (None, [], [[0, 0, 0]] * 478):
            solver.solve(face(left_open=0), 0)
            result = solver.solve(invalid, 33)
            self.assertFalse(result.detected)
            self.assertEqual(sum(result.morph_weights.values()), 0)
            self.assertAlmostEqual(solver.solve(face(), 66).morph_weights["まばたき"], 0)
        points = face()
        points[159][0] = float("nan")
        self.assertFalse(solver.solve(points).detected)
        with self.assertRaises(ValueError):
            FaceThresholds(eye_open=.1, eye_closed=.2)

    def test_gaze_quaternion_against_axis_rotation_composition(self):
        q = FaceSolver(smoothing=False).solve(face(gaze=(.6, .4))).eye_rotation_xyzw
        self.assertAlmostEqual(sum(x * x for x in q), 1)
        # Independently compose yaw-about-Y after pitch-about-X, then apply to Z.
        pitch, yaw = .4 * math.pi / 12, -.6 * math.pi / 6
        expected = (math.sin(yaw) * math.cos(pitch), -math.sin(pitch), math.cos(yaw) * math.cos(pitch))
        x, y, z, w = q
        actual = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=12)

    def test_model_morph_aliases(self):
        s = FaceSolver(smoothing=False)
        s.configure(["瞬き", "ウィンク２", "ウインク右", "あ２", "にっこり"])
        self.assertEqual(set(s.solve(face()).morph_weights), {"瞬き", "ウィンク２", "ウインク右", "あ２", "にっこり"})


class CaptureTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.video = self.root / "fixture.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for i in range(10):
            writer.write(np.full((48, 64, 3), 20 * i, dtype=np.uint8))
        writer.release()
        self.model = self.root / "test.task"
        self.model.write_bytes(b"fake model for injected detector")
        self.job = {"video": str(self.video), "model_path": str(self.model),
                    "output_dir": str(self.root / "new-result"), "fps": 10,
                    "start_seconds": .2, "end_seconds": .6, "timeline_offset_seconds": 15}

    def tearDown(self):
        self.tmp.cleanup()

    def test_interval_alignment_gap_estimation_and_atomic_metadata(self):
        import numpy as np
        from mocap.face.capture import run
        seen = []
        @contextmanager
        def detector(*args):
            def detect(rgb, timestamp):
                seen.append((float(rgb.mean()), timestamp))
                return None if timestamp == 100 else native(eyeBlinkLeft=1., jawOpen=.7)
            yield detect
        with patch("mocap.face.capture._detector", detector):
            result = run(self.job)
        self.assertEqual(result["frame_count"], 4)
        self.assertEqual(result["detected_frames"], 3)
        self.assertEqual(result["start_seconds"], 15.2)
        self.assertEqual([t for _, t in seen], [0, 100, 200, 300])
        for (actual, _), expected in zip(seen, (40, 60, 80, 100)):
            self.assertLess(abs(actual - expected), 3)
        with np.load(result["npz"], allow_pickle=False) as data:
            self.assertEqual(data["weights"].shape, (4, 52))
            self.assertEqual(data["weights"].dtype, np.float32)
            self.assertEqual(data["morph_names"].tolist(), list(BLENDSHAPE_NAMES))
            self.assertEqual(data["detected"].tolist(), [True, False, True, True])
            self.assertEqual(data["raw_weights"][1].sum(), 0)
            self.assertGreater(data["weights"][1, BLENDSHAPE_NAMES.index('jawOpen')], .5)
            self.assertTrue(data['interpolated'][1])
            self.assertFalse(data['predicted'][1])
            self.assertAlmostEqual(float(data["start_seconds"]), 15.2)
        metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
        self.assertEqual(metadata["video_start_seconds"], .2)
        self.assertTrue(metadata["complete"])
        self.assertEqual(metadata['provider'], 'mediapipe-blendshapes')
        self.assertEqual(metadata['schema_version'], 2)
        self.assertEqual(metadata['interpolated_frames'], 1)
        self.assertFalse(list(self.root.glob("*.partial")))
        with self.assertRaises(FileExistsError):
            run(self.job)

    def test_cancel_and_no_detections_publish_nothing(self):
        from mocap.face.capture import run
        @contextmanager
        def detector(*args):
            yield lambda *args: None
        with patch("mocap.face.capture._detector", detector):
            with self.assertRaises(InterruptedError):
                run(self.job, cancelled=lambda: True)
            self.assertFalse(Path(self.job["output_dir"]).exists())
            with self.assertRaisesRegex(RuntimeError, "No face"):
                run(self.job)
        self.assertFalse(Path(self.job["output_dir"]).exists())

    def test_crop_does_not_rescale_native_expression_coefficients(self):
        import numpy as np
        from mocap.face.capture import run
        observation = native(eyeBlinkLeft=.3, eyeBlinkRight=.8, jawOpen=.6, browInnerUp=.4)
        crop = [.25, .125, .75, .875]  # 32 x 36 pixels in a 64 x 48 source
        seen = []
        @contextmanager
        def detector(*args):
            def detect(rgb, timestamp):
                seen.append(rgb.shape)
                return observation
            yield detect
        with patch("mocap.face.capture._detector", detector):
            result = run(dict(self.job, crop=crop, smoothing=False))
        self.assertEqual(seen, [(36, 32, 3)] * 4)
        with np.load(result["npz"], allow_pickle=False) as data:
            np.testing.assert_allclose(data["weights"][0], observation['weights'], atol=1e-7)
            self.assertAlmostEqual(float(data["start_seconds"]), 15.2)
        metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
        self.assertEqual(metadata["crop"], crop)

    def test_prediction_off_keeps_missing_neutral(self):
        import numpy as np
        from mocap.face.capture import run
        @contextmanager
        def detector(*args):
            yield lambda rgb, timestamp: None if timestamp == 100 else native(jawOpen=.7)
        with patch('mocap.face.capture._detector', detector):
            result = run(dict(self.job, prediction=False))
        with np.load(result['npz']) as data:
            self.assertEqual(data['weights'][1].sum(), 0.)
            self.assertFalse(data['predicted'].any() or data['interpolated'].any())

    def test_actor_absence_and_multiple_faces_block_propagation(self):
        import numpy as np
        from mocap.face.capture import run
        @contextmanager
        def detector(*args):
            yield lambda rgb, timestamp: {'blocked': True} if timestamp == 200 else native(jawOpen=.7)
        def crop(index, shape):
            return None if index == 3 else [0, 0, 1, 1]
        with patch('mocap.face.capture._detector', detector):
            result = run(self.job, crop_for_frame=crop)
        with np.load(result['npz']) as data:
            self.assertEqual(data['detected'].tolist(), [True, False, False, True])
            self.assertEqual(data['blocked'].tolist(), [False, True, True, False])
            self.assertEqual(data['weights'][1:3].sum(), 0.)
            self.assertFalse(data['predicted'].any() or data['interpolated'].any())

    def test_native_sides_and_explicit_mirror_swap(self):
        import numpy as np
        from mocap.face.capture import run
        sample = native(eyeBlinkLeft=.8, eyeBlinkRight=.1, jawLeft=.6, mouthLeft=.4)
        @contextmanager
        def detector(*args):
            yield lambda *args: sample
        with patch('mocap.face.capture._detector', detector):
            result = run(dict(self.job, mirror=False, smoothing=False))
        with np.load(result['npz']) as data:
            self.assertAlmostEqual(float(data['weights'][0, BLENDSHAPE_NAMES.index('eyeBlinkRight')]), .8)
            self.assertAlmostEqual(float(data['weights'][0, BLENDSHAPE_NAMES.index('jawRight')]), .6)
            self.assertAlmostEqual(float(data['weights'][0, BLENDSHAPE_NAMES.index('mouthRight')]), .4)

    def test_native_category_names_are_validated_and_order_independent(self):
        from types import SimpleNamespace
        categories = [SimpleNamespace(category_name=n, score=i/100) for i,n in enumerate(BLENDSHAPE_NAMES)]
        self.assertEqual(scores_from_categories(list(reversed(categories))), [i/100 for i in range(52)])
        for invalid in (categories[:-1], categories + categories[:1],
                        [SimpleNamespace(category_name='_neutral', score=float('nan'))] + categories[1:]):
            with self.assertRaises(ValueError): scores_from_categories(invalid)
        self.assertEqual(side_permutation(), list(range(52)))

    def test_invalid_crop_rejected_before_decoding(self):
        from mocap.face.capture import run
        for crop in ([0, 0, 0, 1], [-.1, 0, 1, 1], [0, 0, 1.1, 1],
                     [0, .6, 1, .5], [0, 0, float("nan"), 1], [0, 1]):
            with self.subTest(crop=crop), patch("cv2.VideoCapture") as decoder:
                with self.assertRaisesRegex(ValueError, "crop"):
                    run(dict(self.job, crop=crop))
                decoder.assert_not_called()

    def test_capture_limit_matches_loader_before_array_allocation(self):
        import cv2
        from mocap.face.capture import run
        from mocap.retarget.face_track import MAX_FRAMES
        job = dict(self.job, start_seconds=0, end_seconds=(MAX_FRAMES + 1) / 30, fps=30)
        with patch('cv2.VideoCapture') as factory, patch('numpy.zeros') as allocate:
            cap = factory.return_value
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda prop: 30 if prop == cv2.CAP_PROP_FPS else MAX_FRAMES + 100
            with self.assertRaisesRegex(ValueError, '54,000'):
                run(job)
            allocate.assert_not_called()
            cap.release.assert_called_once()
        self.assertFalse(Path(job['output_dir']).exists())


if __name__ == "__main__":
    unittest.main()
