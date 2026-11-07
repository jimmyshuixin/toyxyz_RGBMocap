"""PMX torso naming/hierarchy regressions with synthetic native identities."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mocap.retarget.mapping import (PMX, PMX_ORIGINAL, PARENTS, VIRTUAL_SPINE1,
                                    body_mapping, mapping_profile, real_bone_mapping)


def rig(renamed=False):
    native = list(PMX_ORIGINAL)
    native[6], native[9] = '上半身2', '上半身3'
    names = [f'physical_{i}' for i in range(22)] if renamed else list(PMX)
    if not renamed:
        names[6], names[9] = '上半身2', '上半身3'
    parents = {name: names[parent] if parent >= 0 else None
               for name, parent in zip(names, PARENTS)}
    identities = {japanese: [name] for japanese, name in zip(native, names)}
    return names, parents, identities


class NumberedSpineTests(unittest.TestCase):
    def test_native_numbered_chain_uses_all_three_physical_pivots(self):
        for renamed in (False, True):
            with self.subTest(renamed=renamed):
                names, parents, identities = rig(renamed)
                mapped = body_mapping('pmx', names, {}, parents=parents, pmx_names=identities)
                self.assertEqual(mapped, names)
                profile = mapping_profile('pmx', mapped)
                self.assertEqual(profile['virtual_slots'], [])
                self.assertEqual(profile['id'], 'pmx_three_spine_v1')
                self.assertEqual(len(real_bone_mapping(mapped, profile)), 22)

    def test_intermediate_parent_controls_preserve_native_bindings(self):
        names, parents, identities = rig()
        for slot in (3, 6, 12, 13, 14):
            old_parent = parents[names[slot]]
            for suffix in range(5):
                helper = f'control_{slot}_{suffix}'
                parents[helper] = old_parent
                old_parent = helper
            parents[names[slot]] = old_parent
        mapped = body_mapping('pmx', parents, {}, parents=parents, pmx_names=identities)
        self.assertEqual(mapped, names)

    def test_rejects_broken_or_ambiguous_numbered_chain(self):
        for slot in (6, 9, 12, 13, 14):
            names, parents, identities = rig()
            parents[names[slot]] = names[0]
            with self.subTest(slot=slot), self.assertRaisesRegex(ValueError, 'UpperBody3 hierarchy'):
                body_mapping('pmx', names, {}, parents=parents, pmx_names=identities)
        names, parents, identities = rig()
        identities['上半身3'].append('duplicate')
        with self.assertRaisesRegex(ValueError, 'ambiguous PMX bone identity'):
            body_mapping('pmx', names + ['duplicate'], {}, parents=parents, pmx_names=identities)

    def test_rejects_cyclic_helper_chain_and_missing_hierarchy(self):
        names, parents, identities = rig()
        parents[names[6]] = 'helper'
        parents['helper'] = 'helper'
        with self.assertRaisesRegex(ValueError, 'Invalid PMX upper-body hierarchy'):
            body_mapping('pmx', parents, {}, parents=parents, pmx_names=identities)
        with self.assertRaisesRegex(ValueError, 'native bone hierarchy'):
            body_mapping('pmx', names, {}, pmx_names=identities)

    def test_legacy_convention_and_true_two_spine_remain_unchanged(self):
        names = list(PMX)
        parents = {name: names[parent] if parent >= 0 else None
                   for name, parent in zip(names, PARENTS)}
        self.assertEqual(body_mapping('pmx', names, {}, parents=parents), names)
        parents[names[9]] = names[3]
        parents.pop(names[6])
        available = [name for name in names if name != names[6]]
        expected = names.copy()
        expected[6] = VIRTUAL_SPINE1
        self.assertEqual(body_mapping('pmx', available, {}, parents=parents), expected)

    def test_props_have_actionable_diagnostic(self):
        with self.assertRaisesRegex(ValueError, 'prop or accessory'):
            body_mapping('pmx', ['全ての親', 'weapon'], {}, parents={'全ての親': None, 'weapon': '全ての親'})


if __name__ == '__main__':
    unittest.main()
