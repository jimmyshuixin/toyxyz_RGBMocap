"""Synthetic, CPU-only regressions for a PMX target with two torso bones.

Run: python tests/test_pmx_two_spine.py -v

No PMX, model weights, Blender, or private character assets are needed. The FK
and skinning reference is an independently composed 21-bone homogeneous rig.
It proves consistency of the solver proxy, not real PMX/MMD evaluation.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mocap.retarget import mapping, pipeline, solver  # noqa: E402


PMX_NAMES = [
    '下半身', '足.L', '足.R', '上半身', 'ひざ.L', 'ひざ.R', '上半身1',
    '足首.L', '足首.R', '上半身2', 'つま先.L', 'つま先.R', '首',
    '肩.L', '肩.R', '頭', '腕.L', '腕.R', 'ひじ.L', 'ひじ.R', '手首.L', '手首.R',
]
REAL_SLOTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
REAL_PARENTS = {
    0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 7: 4, 8: 5, 9: 3,
    10: 7, 11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14,
    18: 16, 19: 17, 20: 18, 21: 19,
}
ABSENT = object()
METRICS = {}


def synthetic_rest():
    return np.array([
        [0, 1.00, 0], [.10, .95, 0], [-.10, .95, 0], [0, 1.08, 0],
        [.10, .55, .02], [-.10, .55, .02], [0, 1.22, .005],
        [.10, .12, 0], [-.10, .12, 0], [0, 1.38, .01],
        [.10, .04, .18], [-.10, .04, .18], [0, 1.56, 0],
        [.08, 1.49, 0], [-.08, 1.49, 0], [0, 1.69, 0],
        [.21, 1.48, 0], [-.21, 1.48, 0], [.48, 1.28, .01],
        [-.48, 1.28, .01], [.69, 1.08, 0], [-.69, 1.08, 0],
    ], dtype=np.float64)


def pmx_fixture(two_spine=True, renamed=False):
    """A native PMX-style tree, including pelvis/spine siblings and controls."""
    names = [f'native_{i}' if renamed else name for i, name in enumerate(PMX_NAMES)]
    if two_spine:
        names[6] = mapping.VIRTUAL_SPINE1
    slots = REAL_SLOTS if two_spine else list(range(22))
    parent_slots = dict(REAL_PARENTS)
    if not two_spine:
        parent_slots[6] = 3
        parent_slots[9] = 6
    parents = {'全ての親': None, '操作中心': None, 'センター': '全ての親',
               'グルーブ': 'センター', '腰': 'グルーブ'}
    for slot in slots:
        parent = parent_slots[slot]
        parents[names[slot]] = '腰' if slot in (0, 3) else names[parent]
    for slot, side in [(13, 'L'), (14, 'R')]:
        helper = f'shoulder_helper_{side}'
        parents[helper] = names[9]
        parents[names[slot]] = helper
    original = {}
    for slot in slots:
        name = PMX_NAMES[slot]
        if name.endswith('.L'):
            name = '左' + name[:-2]
        elif name.endswith('.R'):
            name = '右' + name[:-2]
        original[name] = [names[slot]]
    return names, parents, original


def frozen_baseline(source_world, srest, trest):
    """Baseline from ef05127, before virtual-slot support; unchanged helpers."""
    child = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11,
             9: 12, 12: 15, 13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21}
    offsets = np.tile(np.eye(3), (22, 1, 1))
    for j, k in child.items():
        offsets[j] = solver.align(trest[k] - trest[j], srest[k] - srest[j])
    for j in [10, 11, 15, 20, 21]:
        offsets[j] = offsets[solver.PARENTS[j]]
    return solver.local_from_world(source_world @ offsets)


def physical_rig_matrices(world, rest, root):
    """Ordinary 21-bone FK via 4x4 matrix composition; no virtual node."""
    matrices = {}
    for slot in REAL_SLOTS:
        local = np.broadcast_to(np.eye(4), (len(root), 4, 4)).copy()
        parent = REAL_PARENTS[slot]
        if parent == -1:
            local[:, :3, :3] = world[:, slot]
            local[:, :3, 3] = root
            matrices[slot] = local
        else:
            local[:, :3, :3] = world[:, parent].transpose(0, 2, 1) @ world[:, slot]
            local[:, :3, 3] = rest[slot] - rest[parent]
            matrices[slot] = matrices[parent] @ local
    return np.stack([matrices[slot] for slot in REAL_SLOTS], axis=1)


class SolverProxyTests(unittest.TestCase):
    def test_real_fk_and_zero_virtual_weight_skin_match_21_bone_rig(self):
        rng = np.random.default_rng(2801)
        rest = synthetic_rest()
        rest[6] = rest[9]
        world = Rotation.random(17 * 22, random_state=rng).as_matrix().reshape(17, 22, 3, 3)
        root = rng.normal(size=(17, 3))
        local = solver.local_from_world(world)
        points, calculated_world = solver.fk(local, rest, root)
        physical = physical_rig_matrices(world, rest, root)
        np.testing.assert_allclose(points[:, REAL_SLOTS], physical[:, :, :3, 3], atol=2e-14, rtol=0)
        np.testing.assert_allclose(calculated_world[:, REAL_SLOTS], physical[:, :, :3, :3], atol=2e-14, rtol=0)

        vertices = rng.normal(size=(211, 3))
        weights = rng.uniform(size=(211, 22))
        weights[:, 6] = 0
        weights /= weights.sum(axis=1, keepdims=True)
        actual = solver.skin(dict(rest=rest, world=calculated_world, points=points), vertices, weights)
        inverse_bind = np.broadcast_to(np.eye(4), (21, 4, 4)).copy()
        inverse_bind[:, :3, 3] = -rest[REAL_SLOTS]
        deformed = physical @ inverse_bind
        vertex_h = np.column_stack((vertices, np.ones(len(vertices))))
        expected = np.einsum('vj,tjki,vi->tvk', weights[:, REAL_SLOTS], deformed, vertex_h)[..., :3]
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
        METRICS['real_fk_max_error_m'] = float(np.abs(points[:, REAL_SLOTS] - physical[:, :, :3, 3]).max())
        METRICS['proxy_lbs_max_error_m'] = float(np.abs(actual - expected).max())

        # Alter only the virtual rotation. No actual pivot or skin may move.
        changed = world.copy()
        changed[:, 6] = Rotation.random(17, random_state=rng).as_matrix()
        points2, world2 = solver.fk(solver.local_from_world(changed), rest, root)
        skin2 = solver.skin(dict(rest=rest, world=world2, points=points2), vertices, weights)
        np.testing.assert_allclose(points2[:, REAL_SLOTS], points[:, REAL_SLOTS], atol=2e-14, rtol=0)
        np.testing.assert_allclose(skin2, actual, atol=2e-14, rtol=0)

    def test_midpoint_virtual_pivot_can_match_orientation_but_not_position(self):
        rest = synthetic_rest()
        rest[6] = (rest[3] + rest[9]) / 2
        world = np.broadcast_to(np.eye(3), (1, 22, 3, 3)).copy()
        world[0, 6] = Rotation.from_euler('z', 60, degrees=True).as_matrix()
        world[0, 9] = Rotation.from_euler('xyz', [20, -15, 10], degrees=True).as_matrix()
        root = rest[0][None]
        points, actual_world = solver.fk(solver.local_from_world(world), rest, root)
        physical = physical_rig_matrices(world, rest, root)
        chest = REAL_SLOTS.index(9)
        np.testing.assert_allclose(actual_world[:, 9], physical[:, chest, :3, :3], atol=2e-14, rtol=0)
        self.assertGreater(np.linalg.norm(points[:, 9] - physical[:, chest, :3, 3]), .10)
        with self.assertRaisesRegex(ValueError, 'coincide'):
            solver.baseline(world, synthetic_rest(), rest, virtual_slots=[6])

    def test_three_spine_baseline_matches_frozen_original_bit_for_bit(self):
        rng = np.random.default_rng(407)
        source = synthetic_rest()
        target = source * [1.07, .91, 1.15]
        world = Rotation.random(9 * 22, random_state=rng).as_matrix().reshape(9, 22, 3, 3)
        for shared_pelvis_pivot in (False, True):
            with self.subTest(shared_pelvis_pivot=shared_pelvis_pivot):
                rest = target.copy()
                if shared_pelvis_pivot:
                    rest[3] = rest[0]
                expected = frozen_baseline(world, source, rest)
                np.testing.assert_array_equal(solver.baseline(world, source, rest), expected)
                np.testing.assert_array_equal(solver.baseline(world, source, rest, virtual_slots=[]), expected)
        METRICS['legacy_baseline_max_error'] = 0.0

    def test_declared_virtual_edge_is_not_direction_aligned(self):
        source = synthetic_rest()
        target = source.copy()
        target[6] = target[9]
        world = np.broadcast_to(np.eye(3), (2, 22, 3, 3)).copy()
        original_align = solver.align

        def real_direction_align(target_vector, source_vector):
            self.assertGreater(np.linalg.norm(target_vector), 0)
            return original_align(target_vector, source_vector)

        with mock.patch.object(solver, 'align', side_effect=real_direction_align):
            result = solver.baseline(world, source, target, virtual_slots=np.array([6], np.int64))
        self.assertTrue(np.isfinite(result).all())

    def test_invalid_virtual_configuration_is_rejected(self):
        source = synthetic_rest()
        target = source.copy()
        target[6] = target[9]
        world = np.broadcast_to(np.eye(3), (1, 22, 3, 3))
        for slots in ([3], [9], [6, 6], [6, 9], [6.0], [True], ['6'], [[6]], 6, None):
            with self.subTest(slots=slots), self.assertRaises(ValueError):
                solver.baseline(world, source, target, virtual_slots=slots)
        for change in ('offset', 'nan', 'inf', 'shape'):
            bad = target.copy()
            if change == 'offset':
                bad[6, 0] += 1e-10
            elif change in ('nan', 'inf'):
                bad[[6, 9], 0] = float(change)
            else:
                bad = bad[:21]
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'coincide'):
                solver.baseline(world, source, bad, virtual_slots=[6])


class MappingProfileTests(unittest.TestCase):
    def test_two_spine_native_tree_has_only_21_real_bindings(self):
        names, parents, _ = pmx_fixture()
        actual = mapping.body_mapping('pmx', parents.keys(), {}, parents=parents)
        self.assertEqual(actual, names)
        profile = mapping.mapping_profile('pmx', actual)
        self.assertEqual(profile['id'], 'pmx_two_spine_v1')
        self.assertEqual(profile['virtual_slots'], [6])
        self.assertIsNone(profile['slot_bones'][6])
        real = mapping.real_bone_mapping(actual, profile)
        self.assertEqual(len(real), 21)
        self.assertNotIn(mapping.VIRTUAL_SPINE1, real)
        self.assertEqual(real['上半身'], 3)
        self.assertEqual(real['上半身2'], 9)
        self.assertEqual(mapping.body_trajectory_root(parents, real), '全ての親')

    def test_original_pmx_names_resolve_renamed_blender_bones(self):
        expected, parents, original = pmx_fixture(renamed=True)
        actual = mapping.body_mapping('pmx', parents.keys(), {}, parents=parents, pmx_names=original)
        self.assertEqual(actual, expected)
        original['上半身'] = [expected[3], expected[9]]
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            mapping.body_mapping('pmx', parents.keys(), {}, parents=parents, pmx_names=original)

    def test_bad_two_spine_hierarchy_or_additional_missing_bone_is_rejected(self):
        names, parents, _ = pmx_fixture()
        variants = []
        bad = parents.copy(); bad[names[9]] = names[0]; variants.append(bad)
        bad = parents.copy(); bad[names[12]] = names[3]; variants.append(bad)
        bad = parents.copy(); bad['shoulder_helper_L'] = 'shoulder_helper_L'; variants.append(bad)
        bad = parents.copy(); bad.pop(names[20]); variants.append(bad)
        bad = parents.copy(); bad[mapping.VIRTUAL_SPINE1] = '腰'; variants.append(bad)
        for index, bad in enumerate(variants):
            with self.subTest(variant=index), self.assertRaises(ValueError):
                mapping.body_mapping('pmx', bad.keys(), {}, parents=bad)
        with self.assertRaises(ValueError):
            mapping.body_mapping('pmx', parents.keys(), {})

    def test_supported_three_spine_mixamo_and_vrm_mappings_are_unchanged(self):
        names, parents, _ = pmx_fixture(two_spine=False)
        self.assertEqual(mapping.body_mapping('pmx', parents.keys(), {}), PMX_NAMES)
        for kind, target_names, semantic in [
            ('pmx', names, {}),
            ('mixamo', ['Rig:' + n for n in mapping.NAMES], {}),
            ('vrm', ['vrm_' + str(i) for i in range(22)],
             {label: 'vrm_' + str(i) for i, label in enumerate(mapping.VRM)}),
        ]:
            with self.subTest(kind=kind):
                actual = mapping.body_mapping(kind, target_names, semantic)
                self.assertEqual(actual, target_names)
                profile = mapping.mapping_profile(kind, actual)
                self.assertEqual(profile['virtual_slots'], [])
                self.assertEqual(mapping.real_bone_mapping(actual), mapping.real_bone_mapping(actual, profile))

    def test_profile_cannot_hide_missing_metadata_duplicate_or_changed_bindings(self):
        names, _, _ = pmx_fixture()
        profile = mapping.mapping_profile('pmx', names)
        with self.assertRaises(ValueError):
            mapping.real_bone_mapping(names)
        with self.assertRaises(ValueError):
            mapping.mapping_profile('vrm', names)
        bad_profiles = []
        for field, value in [('version', 2), ('id', 'humanoid_22_v1'),
                             ('virtual_slots', [6.0]), ('virtual_slots', [True]),
                             ('virtual_slots', [6, 6])]:
            bad = deepcopy(profile); bad[field] = value; bad_profiles.append(bad)
        bad = deepcopy(profile); bad['slot_bones'][9] = 'changed_chest'; bad_profiles.append(bad)
        bad = deepcopy(profile); bad['slot_bones'][6] = '上半身1'; bad_profiles.append(bad)
        for bad in bad_profiles:
            with self.subTest(profile=bad), self.assertRaises(ValueError):
                mapping.real_bone_mapping(names, bad)
        duplicate = names.copy(); duplicate[9] = duplicate[3]
        with self.assertRaises(ValueError):
            mapping.real_bone_mapping(duplicate, profile)


class PersistedMetadataTests(unittest.TestCase):
    def check_pair(self, names, slots=ABSENT, profile=ABSENT):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            arrays = dict(names=np.asarray(names, dtype='U'), rest=synthetic_rest())
            if slots is not ABSENT:
                arrays['virtual_slots'] = np.asarray(slots)
            np.savez(folder / 'pmx-canonical.npz', **arrays)
            if profile is not ABSENT:
                rig = {} if profile is None else {'body_profile': profile}
                (folder / 'pmx-rig.json').write_text(json.dumps(rig), encoding='utf-8')
            with np.load(folder / 'pmx-canonical.npz', allow_pickle=False) as target:
                self.assertEqual(target['names'].dtype.kind, 'U')
                return pipeline._target_virtual_slots(folder, 'pmx', target).tolist()

    def test_legacy_and_current_three_spine_files_need_no_virtual_metadata(self):
        profile = mapping.mapping_profile('pmx', PMX_NAMES)
        self.assertEqual(self.check_pair(PMX_NAMES), [])
        self.assertEqual(self.check_pair(PMX_NAMES, profile=None), [])
        self.assertEqual(self.check_pair(PMX_NAMES, np.array([], np.int64), profile), [])

    def test_two_spine_unicode_npz_and_json_profile_agree(self):
        names, _, _ = pmx_fixture()
        profile = mapping.mapping_profile('pmx', names)
        self.assertEqual(self.check_pair(names, np.array([6], np.int64), profile), [6])

    def test_missing_malformed_or_mixed_generation_virtual_metadata_is_rejected(self):
        names, _, _ = pmx_fixture()
        profile = mapping.mapping_profile('pmx', names)
        for slots, stored_profile in [
            (ABSENT, profile), ([], profile), ([3], profile), ([6, 6], profile),
            ([6.0], profile), ([True], profile), ([[6]], profile),
            ([6], ABSENT), ([6], None),
        ]:
            with self.subTest(slots=slots, profile=stored_profile), self.assertRaises(ValueError):
                self.check_pair(names, slots, stored_profile)
        with self.assertRaises(ValueError):
            self.check_pair(PMX_NAMES, [6], mapping.mapping_profile('pmx', PMX_NAMES))
        altered = names.copy(); altered[9] = 'a_different_chest'
        with self.assertRaises(ValueError):
            self.check_pair(altered, [6], profile)


def tearDownModule():
    print('\nSynthetic PMX two-spine validation: ' + json.dumps(METRICS, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
