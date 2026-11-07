"""MediaPipe FaceLandmarker expression names, independent of its native runtime.

The pinned model emits 52 scores including _neutral (not a deforming shape).
Names and indices follow mediapipe.tasks.python.vision.face_landmarker.Blendshapes.
"""
import math

BLENDSHAPE_NAMES = (
    '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp',
    'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft',
    'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
    'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
    'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft',
    'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft',
    'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight',
    'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft',
    'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight',
    'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper',
    'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight',
    'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
    'noseSneerLeft', 'noseSneerRight',
)


def scores_from_categories(categories):
    """Read by category name, never rely on incidental result ordering."""
    scores = {}
    for category in categories:
        name, value = category.category_name, float(category.score)
        if name in scores or name not in BLENDSHAPE_NAMES:
            raise ValueError('Unexpected or duplicate face blendshape: ' + str(name))
        if not math.isfinite(value) or not -1e-6 <= value <= 1 + 1e-6:
            raise ValueError('Invalid face blendshape score: ' + name)
        scores[name] = min(1., max(0., value))
    if len(scores) != len(BLENDSHAPE_NAMES):
        raise ValueError('FaceLandmarker did not return all 52 expression scores.')
    return [scores[name] for name in BLENDSHAPE_NAMES]


def side_permutation(swap=False):
    """Preserve anatomical sides by default; explicitly swap mirrored recordings."""
    names = BLENDSHAPE_NAMES
    def opposite(name):
        if name.endswith('Left'): return name[:-4] + 'Right'
        if name.endswith('Right'): return name[:-5] + 'Left'
        return name
    return [names.index(opposite(name) if swap else name) for name in names]
