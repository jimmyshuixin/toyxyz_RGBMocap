import numpy as np
A = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
NAMES = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg', 'RightLeg', 'Spine1', 'LeftFoot', 'RightFoot', 'Spine2', 'LeftToeBase', 'RightToeBase', 'Neck', 'LeftShoulder', 'RightShoulder', 'Head', 'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand']
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
VRM = ['hips', 'leftUpperLeg', 'rightUpperLeg', 'spine', 'leftLowerLeg', 'rightLowerLeg', 'chest', 'leftFoot', 'rightFoot', 'upperChest', 'leftToes', 'rightToes', 'neck', 'leftShoulder', 'rightShoulder', 'head', 'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm', 'leftHand', 'rightHand']
PMX = ['下半身', '足.L', '足.R', '上半身', 'ひざ.L', 'ひざ.R', '上半身1', '足首.L', '足首.R', '上半身2', 'つま先.L', 'つま先.R', '首', '肩.L', '肩.R', '頭', '腕.L', '腕.R', 'ひじ.L', 'ひじ.R', '手首.L', '手首.R']
PMX_ORIGINAL = ['下半身', '左足', '右足', '上半身', '左ひざ', '右ひざ', '上半身1', '左足首', '右足首', '上半身2', '左つま先', '右つま先', '首', '左肩', '右肩', '頭', '左腕', '右腕', '左ひじ', '右ひじ', '左手首', '右手首']
# An internal slot label, never a Blender/PMX bone or an exported VMD channel.
VIRTUAL_SPINE1 = '__rgb_mocap_virtual_spine1__'


def mapping_profile(kind, mapping):
    """Describe actual bindings separately from the fixed 22 solver slots."""
    names = list(mapping)
    virtual = [6] if len(names) == 22 and names[6] == VIRTUAL_SPINE1 else []
    if virtual and kind != 'pmx':
        raise ValueError('Virtual Spine1 is supported only for the PMX two-spine profile.')
    profile_id = ('pmx_two_spine_v1' if virtual else 'pmx_three_spine_v1') if kind == 'pmx' else 'humanoid_22_v1'
    profile = dict(version=1, id=profile_id,
                   slot_bones=[None if i in virtual else name for i, name in enumerate(names)],
                   virtual_slots=virtual)
    real_bone_mapping(names, profile)
    return profile


def real_bone_mapping(names, profile=None):
    """Validate persisted bindings and return only actual bone -> slot pairs.

    Older imported characters have no profile and retain their 22 real bones.
    A virtual label without matching metadata is rejected, never guessed.
    """
    names = list(names)
    if len(names) != 22 or any(not isinstance(n, str) or not n for n in names):
        raise ValueError('Expected 22 named humanoid slots. Import the character again.')
    if len(set(names)) != 22:
        raise ValueError('Humanoid bones must have distinct mappings')
    if profile is None:
        if VIRTUAL_SPINE1 in names:
            raise ValueError('Virtual Spine1 requires an explicit body profile. Import the character again.')
        return {name: i for i, name in enumerate(names)}
    if not isinstance(profile, dict) or type(profile.get('version')) is not int or profile['version'] != 1:
        raise ValueError('Unsupported character body profile. Import the character again.')
    virtual = profile.get('virtual_slots')
    if not isinstance(virtual, list) or virtual not in ([], [6]) or any(type(i) is not int for i in virtual):
        raise ValueError('Unsupported virtual humanoid slots. Import the character again.')
    profile_id = profile.get('id')
    allowed = ('pmx_two_spine_v1',) if virtual else ('pmx_three_spine_v1', 'humanoid_22_v1')
    if profile_id not in allowed:
        raise ValueError('Character body profile and virtual slots disagree. Import the character again.')
    expected = [None if i in virtual else name for i, name in enumerate(names)]
    if profile.get('slot_bones') != expected:
        raise ValueError('Character bone bindings changed. Import the character again.')
    if (virtual and names[6] != VIRTUAL_SPINE1) or any(n == VIRTUAL_SPINE1 for i, n in enumerate(names) if i not in virtual):
        raise ValueError('Invalid virtual bone label. Import the character again.')
    return {name: i for i, name in enumerate(names) if i not in virtual}


def _pmx_mapping(names, pmx_names):
    if VIRTUAL_SPINE1 in names:
        raise ValueError('The PMX contains a reserved internal bone name: ' + VIRTUAL_SPINE1)
    if pmx_names is None:
        return [name if name in names else None for name in PMX]
    mapping = []
    for original, converted in zip(PMX_ORIGINAL, PMX):
        candidates = set()
        for alias in {original, converted}:
            values = pmx_names.get(alias, ())
            if isinstance(values, str):
                values = [values]
            candidates.update(values)
        if converted in names:
            candidates.add(converted)
        if not candidates.issubset(names) or len(candidates) > 1:
            raise ValueError('Missing or ambiguous PMX bone identity: ' + original)
        mapping.append(next(iter(candidates)) if candidates else None)
    return mapping


def _validate_two_spine(mapping, parents):
    if parents is None or parents.get(mapping[9]) != mapping[3]:
        raise ValueError('PMX two-spine support requires UpperBody2 to be a direct child of UpperBody.')
    actual = {name for name in mapping if name is not None}
    # Shoulder helper controls may lie between Chest and the mapped shoulders.
    # They remain native bones; they do not become additional solver pivots.
    for slot in (12, 13, 14):
        current = parents.get(mapping[slot])
        seen = set()
        while current is not None and current not in actual:
            if current in seen or current not in parents:
                raise ValueError('Invalid PMX upper-body hierarchy.')
            seen.add(current)
            current = parents[current]
        if current != mapping[9]:
            raise ValueError('PMX two-spine support requires the neck and shoulders to descend from UpperBody2.')

def body_trajectory_root(parents, mapped):
    """Use the body's existing root; independent control trees may coexist."""
    roots = set()
    for name in mapped:
        seen = set()
        while True:
            if name not in parents or name in seen:
                raise ValueError('Invalid humanoid bone hierarchy.')
            seen.add(name)
            parent = parents[name]
            if parent is None:
                roots.add(name)
                break
            name = parent
    if len(roots) != 1:
        raise ValueError('The mapped humanoid body must share one hierarchy root.')
    return next(iter(roots))

def body_mapping(kind,bones,semantic,*,parents=None,pmx_names=None):
    names=set(bones)
    if kind=='mixamo':
        suffix={}
        for name in names:
            suffix.setdefault(name.split(':')[-1],[]).append(name)
        mapping=[suffix.get(n,[]) for n in NAMES]
        missing=[n for n,v in zip(NAMES,mapping) if len(v)!=1]
        if missing:raise ValueError('Expected a Mixamo humanoid rig. Missing or ambiguous bones: '+', '.join(missing))
        return [v[0] for v in mapping]
    if kind == 'pmx':
        mapping = _pmx_mapping(names, pmx_names)
        if [i for i, name in enumerate(mapping) if name is None] == [6]:
            _validate_two_spine(mapping, parents)
            mapping[6] = VIRTUAL_SPINE1
        missing = [NAMES[i] for i, name in enumerate(mapping) if name is None]
        if missing:
            raise ValueError('Unsupported humanoid mapping (manual mapping is not available yet): ' + ', '.join(missing))
        mapping_profile(kind, mapping)
        return mapping
    mapping=[semantic.get(n) for n in VRM] if kind=='vrm' else PMX
    missing=[NAMES[i] for i,n in enumerate(mapping) if n not in names]
    if missing:raise ValueError('Unsupported humanoid mapping (manual mapping is not available yet): '+', '.join(missing))
    if len(set(mapping))!=22:raise ValueError('Humanoid bones must have distinct mappings')
    return mapping
