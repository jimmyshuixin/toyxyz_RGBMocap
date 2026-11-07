"""Original PMX mapping limits and an independent directional retarget check."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mocap.retarget import mapping, solver
from mocap.retarget.pmx_validation import validate_asset


class UpstreamMappingTests(unittest.TestCase):
    def test_standard_pmx_keeps_original_order_and_real_spine_bones(self):
        bones = list(reversed(mapping.PMX)) + ['全ての親', 'センター', '操作中心']
        actual = mapping.body_mapping('pmx', bones, {})
        self.assertEqual(actual, mapping.PMX)
        self.assertEqual([actual[i] for i in (3, 6, 9)], ['上半身', '上半身1', '上半身2'])

    def test_two_spine_numbered_and_renamed_models_are_not_substituted(self):
        for substitute in (None, '上半身3', '__rgb_mocap_virtual_spine1__', 'renamed_spine'):
            with self.subTest(substitute=substitute):
                bones = [name for name in mapping.PMX if name != '上半身1']
                if substitute is not None:
                    bones.append(substitute)
                with self.assertRaisesRegex(ValueError, 'Spine1'):
                    mapping.body_mapping('pmx', bones, {})

    def test_other_humanoid_mappings_keep_their_own_names(self):
        names = ['Rig:' + name for name in mapping.NAMES]
        self.assertEqual(mapping.body_mapping('mixamo', names, {}), names)
        semantic = {key: 'vrm_' + str(i) for i, key in enumerate(mapping.VRM)}
        self.assertEqual(mapping.body_mapping('vrm', semantic.values(), semantic), list(semantic.values()))

    def test_baseline_aligns_each_real_segment_to_source_direction(self):
        rng = np.random.default_rng(74213)
        source = rng.normal(size=(22, 3))
        target = rng.normal(size=(22, 3))
        source_world = Rotation.random(4 * 22, random_state=rng).as_matrix().reshape(4, 22, 3, 3)
        local = solver.baseline(source_world, source, target)
        # Independently compose local rotations; do not use the solver FK helper.
        world = np.empty_like(local)
        for joint, parent in enumerate(mapping.PARENTS):
            world[:, joint] = local[:, joint] if parent < 0 else world[:, parent] @ local[:, joint]
        children = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11,
                    9: 12, 12: 15, 13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21}
        for joint, child in children.items():
            incoming = target[child] - target[joint]
            reference = source[child] - source[joint]
            actual = world[:, joint] @ (incoming / np.linalg.norm(incoming))
            expected = source_world[:, joint] @ (reference / np.linalg.norm(reference))
            np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)

    def test_non_degenerate_align_is_rigid_including_opposite_directions(self):
        a = np.array([.2, -.4, .7])
        for b in (np.array([.5, .3, -.1]), -a, a):
            rotation = solver.align(a, b)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(rotation), 1., places=14)
            np.testing.assert_allclose(rotation @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-14)


class CachedUpstreamMappingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.arrays = dict(names=np.asarray(mapping.PMX), parents=mapping.PARENTS.copy(), rest=np.zeros((22, 3)))
        self.rig = dict(bones=[dict(name=name) for name in mapping.PMX])

    def write_asset(self):
        np.savez(self.folder / 'pmx-canonical.npz', **self.arrays)
        (self.folder / 'pmx-rig.json').write_text(json.dumps(self.rig), encoding='utf-8')

    def profile(self):
        return dict(version=1, id='pmx_three_spine_v1', slot_bones=list(mapping.PMX), virtual_slots=[])

    def test_original_and_compatible_fork_caches_are_read_only(self):
        for use_profile in (False, True):
            with self.subTest(fork_metadata=use_profile):
                if use_profile:
                    self.rig['body_profile'] = self.profile()
                    self.arrays['virtual_slots'] = np.array([], dtype=np.int64)
                self.write_asset()
                before = {path.name: path.read_bytes() for path in self.folder.iterdir()}
                validate_asset(self.folder)
                self.assertEqual(before, {path.name: path.read_bytes() for path in self.folder.iterdir()})

    def test_old_virtual_or_numbered_names_are_rejected(self):
        for name in ('__rgb_mocap_virtual_spine1__', '上半身2', 'renamed_spine'):
            with self.subTest(name=name):
                names = list(mapping.PMX)
                names[6] = name
                self.arrays['names'] = np.asarray(names)
                self.write_asset()
                with self.assertRaisesRegex(ValueError, 'Import the character again'):
                    validate_asset(self.folder)

    def test_virtual_metadata_and_changed_parent_table_are_rejected(self):
        self.arrays['virtual_slots'] = np.array([6], dtype=np.int64)
        self.write_asset()
        with self.assertRaisesRegex(ValueError, 'non-upstream'):
            validate_asset(self.folder)
        self.arrays.pop('virtual_slots')
        self.arrays['parents'][9] = 3
        self.write_asset()
        with self.assertRaisesRegex(ValueError, 'non-upstream'):
            validate_asset(self.folder)

    def test_missing_real_bone_and_mismatched_profile_are_rejected(self):
        self.rig['bones'].pop(6)
        self.write_asset()
        with self.assertRaisesRegex(ValueError, 'non-upstream'):
            validate_asset(self.folder)
        self.rig['bones'] = [dict(name=name) for name in mapping.PMX]
        for key, value in (('id', 'pmx_two_spine_v1'), ('version', 2),
                           ('virtual_slots', [6]), ('slot_bones', list(reversed(mapping.PMX)))):
            with self.subTest(field=key):
                self.rig['body_profile'] = self.profile()
                self.rig['body_profile'][key] = value
                self.write_asset()
                with self.assertRaisesRegex(ValueError, 'non-upstream'):
                    validate_asset(self.folder)


if __name__ == '__main__':
    unittest.main()
