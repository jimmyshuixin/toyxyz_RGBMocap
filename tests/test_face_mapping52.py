"""Native expression retargeting, provenance and independent VMD parsing."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mocap.face.blendshapes import BLENDSHAPE_NAMES
from mocap.face.vmd import export_vmd
from mocap.retarget.face_track import (load_track, sample_track, map_shape_channels,
                                      mapped_source_channels, MMD_NATIVE_CHANNELS)


def native(**values):
    result = np.zeros((1, len(BLENDSHAPE_NAMES)), np.float32)
    for name, value in values.items():
        result[0, BLENDSHAPE_NAMES.index(name)] = value
    return result


def read_vmd(path):
    """Independent fixed-width parser; do not reuse the writer's code."""
    data = Path(path).read_bytes()
    assert data[:30].rstrip(b'\0') == b'Vocaloid Motion Data 0002'
    bones, count = struct.unpack_from('<II', data, 50)
    assert bones == 0
    records = []
    for index in range(count):
        offset = 58+index*23
        name = data[offset:offset+15].split(b'\0')[0].decode('cp932')
        frame, value = struct.unpack_from('<If', data, offset+15)
        records.append((name, frame, value))
    assert len(data) == 58+count*23+16
    assert data[-16:] == bytes(16)
    return records


class NativeMappingTests(unittest.TestCase):
    def test_native_coefficients_preserved_without_neutral_or_alias_duplication(self):
        weights = np.linspace(0, 1, 104, dtype=np.float32).reshape(2, 52)
        available = [*BLENDSHAPE_NAMES, 'まばたき', 'ウィンク', 'ウィンク右',
                     'eyeBlink_L', 'eyeBlink_R', 'にやり', 'あ', '困る']
        mapped = map_shape_channels(BLENDSHAPE_NAMES, weights, available)
        self.assertEqual(set(mapped), set(BLENDSHAPE_NAMES)-{'_neutral'})
        for index, name in enumerate(BLENDSHAPE_NAMES):
            if name != '_neutral':
                np.testing.assert_array_equal(mapped[name], weights[:, index])

    def test_mmd_shared_and_residual_eyelids_reproduce_asymmetric_closure(self):
        weights = np.concatenate([native(eyeBlinkLeft=.8, eyeBlinkRight=.25),
                                  native(eyeBlinkLeft=.1, eyeBlinkRight=.9),
                                  native(eyeBlinkLeft=1, eyeBlinkRight=1)])
        mapped = map_shape_channels(BLENDSHAPE_NAMES, weights, MMD_NATIVE_CHANNELS)
        np.testing.assert_allclose(mapped['まばたき'], [.25, .1, 1])
        np.testing.assert_allclose(mapped['まばたき']+mapped['ウィンク'], [.8, .1, 1])
        np.testing.assert_allclose(mapped['まばたき']+mapped['ウィンク右'], [.25, .9, 1])
        self.assertTrue(np.all(mapped['ウィンク']*mapped['ウィンク右'] == 0))

    def test_split_vrm_eyes_and_explicit_side_aliases(self):
        weights = native(eyeBlinkLeft=.8, eyeBlinkRight=.2, browDownLeft=.3, jawOpen=.4)
        mapped = map_shape_channels(BLENDSHAPE_NAMES, weights,
                                    ['Fcl_EYE_Close', 'Fcl_EYE_Close_L', 'Fcl_EYE_Close_R',
                                     'browDown_L', 'Fcl_MTH_A'])
        self.assertNotIn('Fcl_EYE_Close', mapped)
        self.assertAlmostEqual(mapped['Fcl_EYE_Close_L'][0], .8)
        self.assertAlmostEqual(mapped['Fcl_EYE_Close_R'][0], .2)
        self.assertAlmostEqual(mapped['browDown_L'][0], .3)
        self.assertAlmostEqual(mapped['Fcl_MTH_A'][0], .4)

    def test_native_brows_mouth_and_cheek_survive_where_keys_exist(self):
        weights = native(browInnerUp=.7, cheekPuff=.2, noseSneerLeft=.6,
                         mouthPressLeft=.3, mouthRollUpper=.8)
        available = ['browInnerUp', 'cheekPuff', 'noseSneer_L', 'mouthPress_L', 'mouthRollUpper']
        mapped = map_shape_channels(BLENDSHAPE_NAMES, weights, available)
        np.testing.assert_allclose([mapped[n][0] for n in available], [.7, .2, .6, .3, .8])

    def test_mmd_vowels_are_bounded_and_brows_are_independent(self):
        weights = native(jawOpen=1, mouthPucker=.8, mouthFunnel=.9,
                         mouthStretchLeft=.6, mouthStretchRight=.4,
                         browInnerUp=.7, browOuterUpLeft=.6, browOuterUpRight=.2,
                         browDownLeft=.3, browDownRight=.5,
                         mouthSmileLeft=.8, mouthSmileRight=.4)
        mapped = map_shape_channels(BLENDSHAPE_NAMES, weights, MMD_NATIVE_CHANNELS)
        self.assertAlmostEqual(sum(mapped[n][0] for n in ('あ', 'い', 'う', 'お')), 1., places=6)
        np.testing.assert_allclose([mapped[n][0] for n in ('上', '下', '困る', 'にやり')], [.4, .4, .7, .6])
        self.assertTrue(all(np.all((values >= 0) & (values <= 1)) for values in mapped.values()))

    def test_closed_lip_smile_does_not_open_wa(self):
        weights = native(jawOpen=0, mouthSmileLeft=1, mouthSmileRight=1)
        mapped = map_shape_channels(BLENDSHAPE_NAMES, weights, ['ワ'])
        self.assertEqual(mapped['ワ'][0], 0)
        weights[0, BLENDSHAPE_NAMES.index('jawOpen')] = .5
        self.assertAlmostEqual(map_shape_channels(BLENDSHAPE_NAMES, weights, ['ワ'])['ワ'][0], .5)

    def test_nonlinear_coverage_and_neutral_exclusion(self):
        names = ['_neutral', 'eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen', 'mouthClose']
        coverage = mapped_source_channels(names, ['まばたき', 'あ'])
        np.testing.assert_array_equal(coverage, [False, True, True, True, True])
        self.assertEqual(map_shape_channels(['_neutral'], np.ones((2,1), np.float32), ['_neutral']), {})


class NativeTrackTests(unittest.TestCase):
    def base(self):
        return dict(morph_names=np.asarray(BLENDSHAPE_NAMES),
                    weights=np.repeat(native(jawOpen=.6), 3, axis=0),
                    fps=30., start_seconds=2., detected=np.array([True, False, False]),
                    raw_weights=np.concatenate([native(jawOpen=.6), native(), native()]),
                    predicted=np.array([False, True, False]),
                    interpolated=np.array([False, False, False]),
                    blocked=np.array([False, False, True]),
                    reliability=np.array([1., .8, 0.], np.float32))

    def test_optional_provenance_loaded_and_prediction_samples_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'face.npz'
            data = self.base(); data['weights'][2] = 0
            np.savez(path, **data)
            track = load_track(path)
            self.assertFalse(track['detected'][1])
            self.assertTrue(track['predicted'][1])
            np.testing.assert_array_equal(track['raw_weights'], data['raw_weights'])
            np.testing.assert_allclose(sample_track(track, 2., 3, 30.), data['weights'])

    def test_invalid_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'face.npz'
            replacements = {
                'predicted': np.array([True, True, False]),
                'interpolated': np.array([False, True, False]),
                'blocked': np.array([False, True, True]),
                'raw_weights': np.zeros((3, 51), np.float32),
                'reliability': np.array([1., np.nan, 0.]),
            }
            for name, value in replacements.items():
                with self.subTest(name=name):
                    np.savez(path, **dict(self.base(), **{name:value}))
                    with self.assertRaises(ValueError):
                        load_track(path)

    def test_native_vmd_has_short_mmd_names_and_retains_predicted_expression(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/'face.npz', Path(tmp)/'face.vmd'
            weights = np.concatenate([native(eyeBlinkLeft=.8, eyeBlinkRight=.2, jawOpen=.6),
                                      native(eyeBlinkLeft=.7, eyeBlinkRight=.2, jawOpen=.5)])
            np.savez(source, morph_names=BLENDSHAPE_NAMES, weights=weights, fps=30.,
                     start_seconds=12., detected=np.array([True, False]))
            export_vmd(source, target)
            records = read_vmd(target)
            self.assertEqual(len(records), 2*len(MMD_NATIVE_CHANNELS))
            self.assertEqual(set(name for name, _, _ in records), set(MMD_NATIVE_CHANNELS))
            values = {(name, frame): value for name, frame, value in records}
            self.assertAlmostEqual(values['まばたき', 1]+values['ウィンク', 1], .7)
            self.assertAlmostEqual(values['あ', 1], .5)
            self.assertEqual(set(frame for _, frame, _ in records), {0,1})

    def test_legacy_vmd_keeps_names_values_and_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/'face.npz', Path(tmp)/'face.vmd'
            names = ['まばたき', 'ウィンク', 'ウィンク右', 'あ', 'ワ']
            weights = np.array([[.1, .2, .3, .4, .5]], np.float32)
            np.savez(source, morph_names=names, weights=weights, fps=30.,
                     start_seconds=2., detected=np.array([True]))
            export_vmd(source, target)
            self.assertEqual(target.read_bytes()[30:50], b'reze-mipo face'.ljust(20,b'\0'))
            records = read_vmd(target)
            self.assertEqual([r[0] for r in records], names)
            np.testing.assert_array_equal([r[2] for r in records], weights[0])

    def test_native_vmd_rate_conversion_precedes_nonlinear_rig_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/'face.npz', Path(tmp)/'face.vmd'
            weights = np.concatenate([
                native(eyeBlinkLeft=.8, eyeBlinkRight=.2, jawOpen=1., mouthFunnel=0.),
                native(eyeBlinkLeft=.2, eyeBlinkRight=.8, jawOpen=0., mouthFunnel=1.),
            ])
            np.savez(source, morph_names=BLENDSHAPE_NAMES, weights=weights, fps=15.,
                     start_seconds=7.5, detected=np.array([True, True]))
            export_vmd(source, target)
            records = read_vmd(target)
            values = {(name, frame): value for name, frame, value in records}
            self.assertEqual(set(frame for _, frame, _ in records), {0,1,2})
            # At t=1/30, native L/R both equal .5: shared blink is .5 and
            # both residual winks are zero. Mapping first would yield .2/.3/.3.
            self.assertAlmostEqual(values['まばたき', 1], .5)
            self.assertAlmostEqual(values['ウィンク', 1], 0.)
            self.assertAlmostEqual(values['ウィンク右', 1], 0.)
            # Native jaw=.5 and funnel=.5 give A=.5*(1-.5)=.25, O=.5.
            # Interpolating already-mapped endpoints would incorrectly give A=.5.
            self.assertAlmostEqual(values['あ', 1], .25)
            self.assertAlmostEqual(values['お', 1], .5)


if __name__ == '__main__':
    unittest.main()
