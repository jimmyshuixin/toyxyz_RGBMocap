"""Reject undefined native PMX dependencies before Blender builds its rig."""


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
