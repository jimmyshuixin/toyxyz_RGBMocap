"""Face timeline boundaries and named channel mapping; no inference or assets."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mocap.retarget.face_track import load_track, sample_track, map_shape_channels


class FaceTrackTest(unittest.TestCase):
    def test_absolute_timing_and_rate_conversion(self):
        track=dict(weights=np.array([[0.], [1.], [0.]], np.float32),fps=2.,start_seconds=10.)
        np.testing.assert_allclose(sample_track(track,9.75,7,4.)[:,0],[0,0,.5,1,.5,0,0])

    def test_mmd_and_split_eye_aliases(self):
        names=['まばたき','ウィンク','ウィンク右','あ','ワ']
        w=np.array([[.2,.3,.4,.5,.6]],np.float32)
        exact=map_shape_channels(names,w,names)
        np.testing.assert_allclose(np.column_stack([exact[n] for n in names]),w)
        aliases=map_shape_channels(names,w,['eyeBlinkLeft','eyeBlinkRight','jawOpen','mouthSmileLeft','mouthSmileRight'])
        np.testing.assert_allclose([aliases[n][0] for n in aliases],[.5,.6,.5,.6,.6])

    def test_invalid_track_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'face.npz'
            data=dict(morph_names=np.array(['あ']),weights=np.array([[.25]],np.float32),fps=30.,start_seconds=1.,detected=np.array([True]))
            np.savez(p,**data)
            self.assertEqual(load_track(p)['morph_names'],['あ'])
            data['weights'][0,0]=np.nan
            np.savez(p,**data)
            with self.assertRaises(ValueError):load_track(p)


if __name__=='__main__':unittest.main()
