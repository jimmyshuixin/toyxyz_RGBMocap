"""Changed attention code must invalidate an existing ExMR motion receipt."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mocap import exmr_backend, models


class ExMRReceiptTests(unittest.TestCase):
    def test_both_attention_sources_are_in_motion_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            (project / "project.json").write_text(
                json.dumps({"frames": 10800, "fps": 30, "pipeline": {"model": "exmr"}}),
                encoding="utf-8",
            )
            for name in ("input.mp4", "tracking.npz", "exmr_keypoints.pt",
                         "exmr_body_features.pt", "exmr_hand_features.pt", "weight"):
                (project / name).write_bytes(b"identity fixture only; never decoded")
            vendor = root / "upstream" / "ExMR"
            files = ["RGB_MOCAP_PROVENANCE.json", "exmr/pipeline.py",
                     "exmr/model/exmr.py", "exmr/model/global_traj.py", "exmr/model/ik.py",
                     "exmr/geometry/camera.py", "exmr/model/layers.py",
                     "exmr/model/query_batched_attention.py"]
            for name in files:
                destination = vendor / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / "third_party" / "ExMR" / name, destination)
            with patch.object(exmr_backend, "UPSTREAM", root / "upstream"), \
                    patch.object(models, "BY_KEY", {"fixture": SimpleNamespace(path=project / "weight")}), \
                    patch.object(models, "STAGE_REQUIREMENTS", {"exmr": ("fixture",)}):
                initial = exmr_backend.identity(project, "exmr")
                self.assertEqual(initial, exmr_backend.identity(project, "exmr"))
                for name in ("model/layers.py", "model/query_batched_attention.py"):
                    with self.subTest(source=name):
                        path = vendor / "exmr" / name
                        original = path.read_bytes()
                        path.write_bytes(original + b"\n# changed solver implementation\n")
                        changed = exmr_backend.identity(project, "exmr")
                        self.assertNotEqual(initial["motion_solver"][name],
                                            changed["motion_solver"][name])
                        self.assertNotEqual(initial, changed)
                        path.write_bytes(original)
                        self.assertEqual(initial, exmr_backend.identity(project, "exmr"))


if __name__ == "__main__":
    unittest.main()
