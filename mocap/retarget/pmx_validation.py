"""Validate native PMX input and compatibility with the upstream body mapping."""


def validate_asset(folder):
    """Reject cached extended mappings before any source preparation or solve.

    Standard three-spine assets imported by the fork are still usable. Virtual
    or renamed slots must never be silently fed into the original solver.
    This check only reads the existing asset; it does not migrate user data.
    """
    import json
    from pathlib import Path
    import numpy as np
    from .mapping import PMX, PARENTS

    folder = Path(folder)
    message = ('This cached PMX uses a non-upstream bone mapping. Import the character again. '
               'Original retargeting requires real 上半身, 上半身1 and 上半身2 bones; '
               'two-spine and alternate numbered mappings are not supported.')
    rig = json.loads((folder / 'pmx-rig.json').read_text(encoding='utf-8'))
    with np.load(folder / 'pmx-canonical.npz', allow_pickle=False) as canonical:
        if (canonical['names'].tolist() != PMX
                or not np.array_equal(canonical['parents'], PARENTS)
                or ('virtual_slots' in canonical and canonical['virtual_slots'].size)):
            raise ValueError(message)
    if not set(PMX).issubset(bone['name'] for bone in rig['bones']):
        raise ValueError(message)
    profile = rig.get('body_profile')
    if profile is not None and (profile.get('id') != 'pmx_three_spine_v1'
                                or profile.get('version') != 1
                                or profile.get('virtual_slots') != []
                                or profile.get('slot_bones') != PMX):
        raise ValueError(message)


def self_grants(bones):
    """Return active self-referential append transforms, including their indices."""
    return [(i, bone) for i, bone in enumerate(bones)
            if (bone.hasAdditionalRotate or bone.hasAdditionalLocation)
            and bone.additionalTransform is not None
            and bone.additionalTransform[0] == i
            and bone.additionalTransform[1] != 0]


def validate_bones(bones):
    invalid = self_grants(bones)
    if invalid:
        names = ', '.join(f'{bone.name} (#{index})' for index, bone in invalid)
        raise ValueError('Invalid PMX self-referencing append transforms: ' + names
                         + '. They create unstable deformation in Blender. Repair a separate PMX copy'
                         + ' in a model editor, then import that copy. VMD must use the same repaired model.')
