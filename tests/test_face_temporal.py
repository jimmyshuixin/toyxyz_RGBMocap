"""Behavioural checks for bounded missing-face expression estimates."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mocap.face.temporal import process_track


NAMES = ('jawOpen', 'eyeBlinkLeft', 'mouthSmileLeft')


class FaceTemporalTests(unittest.TestCase):
    def test_short_gap_uses_both_observations_without_marking_a_detection(self):
        raw = np.zeros((12, 3), np.float32)
        raw[:4] = .2; raw[8:] = .8
        detected = np.ones(12, bool); detected[4:8] = False
        original = detected.copy()
        result = process_track(raw, detected, 30, NAMES, smoothing=False)
        middle = result['weights'][4:8, 0]
        self.assertTrue(np.all((middle > .2) & (middle < .8)))
        self.assertTrue(np.all(np.diff(middle) > 0))
        np.testing.assert_array_equal(result['interpolated'], ~detected)
        self.assertFalse(result['predicted'].any())
        np.testing.assert_array_equal(detected, original)
        self.assertTrue(np.all(result['reliability'][~detected] < 1))

    def test_tail_prediction_fades_to_exact_neutral(self):
        raw = np.zeros((75, 3)); raw[:15] = .7
        detected = np.zeros(75, bool); detected[:15] = True
        result = process_track(raw, detected, 30, NAMES)
        self.assertGreater(result['weights'][15, 0], .5)
        self.assertTrue(result['predicted'][15])
        self.assertLess(result['weights'][30, 0], result['weights'][18, 0])
        np.testing.assert_array_equal(result['weights'][38:], 0)
        self.assertFalse(result['predicted'][38:].any())
        self.assertEqual(float(result['reliability'][-1]), 0)
        # A missed face must not freeze a closed eye for the full jaw fade.
        self.assertLess(float(result['weights'][24, 1]), .02)

    def test_long_gap_does_not_borrow_a_future_expression(self):
        raw = np.zeros((60, 3)); raw[:5] = .25; raw[50:] = .9
        detected = np.zeros(60, bool); detected[:5] = True; detected[50:] = True
        result = process_track(raw, detected, 30, NAMES, smoothing=False)
        self.assertFalse(result['interpolated'].any())
        np.testing.assert_array_equal(result['weights'][29:50], 0)
        self.assertLessEqual(float(result['weights'][5:29].max()), .25)

    def test_prediction_has_a_bounded_positive_recent_trend(self):
        raw = np.zeros((40, 3)); raw[:10, 0] = np.linspace(.1, .7, 10)
        detected = np.zeros(40, bool); detected[:10] = True
        result = process_track(raw, detected, 30, NAMES, smoothing=False)
        self.assertGreater(float(result['weights'][10, 0]), .7)
        self.assertLessEqual(float(result['weights'][10:16, 0].max()), .821)
        self.assertTrue(np.isfinite(result['weights']).all())
        self.assertTrue(np.all((result['weights'] >= 0) & (result['weights'] <= 1)))

    def test_frame_rate_changes_do_not_change_prediction_time_constants(self):
        samples = []
        for fps in (30, 60, 120):
            raw = np.zeros((fps * 2 + 1, 3)); raw[:fps + 1] = .6
            detected = np.zeros(len(raw), bool); detected[:fps + 1] = True
            result = process_track(raw, detected, fps, NAMES, smoothing=False)
            samples.append(result['weights'][[fps + round(t * fps) for t in (.1, .2, .4, .7, .8)]])
        np.testing.assert_allclose(samples[0], samples[1], atol=1e-6)
        np.testing.assert_allclose(samples[0], samples[2], atol=1e-6)

    def test_invisible_eyes_do_not_develop_a_predicted_blink(self):
        raw = np.zeros((45, 3)); raw[:10, 1] = np.linspace(0, .3, 10)
        detected = np.zeros(45, bool); detected[:10] = True
        result = process_track(raw, detected, 30, NAMES, smoothing=False)
        self.assertLessEqual(float(result['weights'][10:, 1].max()), .301)
        self.assertTrue(np.all(np.diff(result['weights'][10:, 1]) <= 0))
        np.testing.assert_array_equal(result['weights'][17:, 1], 0)

    def test_reacquisition_reduces_the_initial_jump(self):
        raw = np.zeros((65, 3)); raw[:5] = .4; raw[45:] = 1
        detected = np.zeros(65, bool); detected[:5] = True; detected[45:] = True
        eased = process_track(raw, detected, 30, NAMES)
        abrupt = process_track(raw, detected, 30, NAMES, smoothing=False)
        self.assertLess(float(eased['weights'][45, 0]), float(abrupt['weights'][45, 0]) * .6)
        self.assertGreater(float(eased['weights'][52, 0]), .95)
        self.assertTrue(np.all(np.diff(eased['weights'][45:55, 0]) >= 0))

    def test_smoothing_reduces_stationary_noise_without_erasing_blinks(self):
        raw = np.full((90, 3), .4)
        raw[:, 2] += np.where(np.arange(90) % 2, .05, -.05)
        raw[:, 1] = 0; raw[30:32, 1] = 1
        result = process_track(raw, np.ones(90, bool), 30, NAMES)
        self.assertLess(float(result['weights'][10:, 2].std()), float(raw[10:, 2].std()) * .6)
        self.assertGreater(float(result['weights'][30:32, 1].max()), .95)
        self.assertLess(float(result['weights'][34, 1]), .02)

    def test_leading_or_entirely_missing_frames_do_not_invent_expressions(self):
        raw = np.ones((30, 3))
        detected = np.zeros(30, bool)
        empty = process_track(raw, detected, 30, NAMES)
        np.testing.assert_array_equal(empty['weights'], 0)
        self.assertFalse(empty['predicted'].any())
        detected[20:] = True
        late = process_track(raw, detected, 30, NAMES)
        np.testing.assert_array_equal(late['weights'][:20], 0)
        self.assertFalse(late['interpolated'].any())

    def test_barrier_blocks_identity_leak_and_cross_gap_interpolation(self):
        raw = np.zeros((15, 3)); raw[:3] = .8; raw[10:] = .2
        detected = np.zeros(15, bool); detected[:3] = True; detected[10:] = True
        blocked = np.zeros(15, bool); blocked[5] = True
        result = process_track(raw, detected, 30, NAMES, blocked=blocked)
        self.assertGreater(float(result['weights'][4, 0]), 0)
        np.testing.assert_array_equal(result['weights'][5:10], 0)
        self.assertFalse(result['interpolated'].any())
        self.assertFalse(result['predicted'][5:10].any())
        self.assertLess(float(result['weights'][10, 0]), .2)

    def test_disabling_prediction_leaves_missing_frames_neutral(self):
        raw = np.full((20, 3), .7); detected = np.ones(20, bool); detected[8:10] = False
        result = process_track(raw, detected, 30, NAMES, prediction=False)
        np.testing.assert_array_equal(result['weights'][8:10], 0)
        self.assertFalse(result['predicted'].any())
        self.assertFalse(result['interpolated'].any())

    def test_inputs_are_not_modified_and_masks_are_disjoint(self):
        raw = np.full((45, 3), .4, np.float32)
        detected = np.zeros(45, bool); detected[:5] = True; detected[7:10] = True
        blocked = np.zeros(45, bool); blocked[-1] = True
        original = [x.copy() for x in (raw, detected, blocked)]
        result = process_track(raw, detected, 30, NAMES, blocked=blocked)
        for got, expected in zip((raw, detected, blocked), original):
            np.testing.assert_array_equal(got, expected)
        self.assertFalse(np.any(result['predicted'] & result['interpolated']))
        self.assertFalse(np.any((result['predicted'] | result['interpolated']) & detected))
        self.assertEqual(result['weights'].dtype, np.float32)
        self.assertEqual(result['predicted'].dtype, np.bool_)

    def test_invalid_data_is_rejected(self):
        raw = np.zeros((5, 3)); detected = np.ones(5, bool)
        for fps in (0, float('nan'), 300):
            with self.assertRaises(ValueError): process_track(raw, detected, fps, NAMES)
        with self.assertRaises(ValueError): process_track(raw, detected.astype(int), 30, NAMES)
        with self.assertRaises(ValueError): process_track(raw, detected, 30, ('x', 'x', 'z'))
        with self.assertRaises(ValueError): process_track(raw, detected, 30, NAMES, blocked=detected)
        raw[0, 0] = float('nan')
        with self.assertRaises(ValueError): process_track(raw, detected, 30, NAMES)


if __name__ == '__main__':
    unittest.main()
