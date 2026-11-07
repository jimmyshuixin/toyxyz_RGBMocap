"""Validated, timeline-anchored facial coefficients shared by preview and export."""
from pathlib import Path
import zipfile
import numpy as np

MAX_FRAMES = 54000
MAX_CHANNELS = 64


def load_track(path):
    path = Path(path)
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError('Face capture exceeds the 64 MB limit.')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 128 or sum(item.file_size for item in entries) > 64 * 1024**2:
            raise ValueError('Expanded face capture exceeds the 64 MB limit.')
    with np.load(path, allow_pickle=False) as data:
        names = data['morph_names']
        weights = np.asarray(data['weights'], dtype=np.float32)
        fps = float(data['fps'])
        start = float(data['start_seconds'])
        detected = np.asarray(data['detected'])
        extra = {name: np.asarray(data[name]) for name in
                 ('raw_weights', 'predicted', 'interpolated', 'reliability', 'blocked') if name in data}
    if names.ndim != 1 or names.dtype.kind not in 'US':
        raise ValueError('Invalid face morph names.')
    names = [str(name) for name in names.astype('U')]
    if (not 1 <= len(names) <= MAX_CHANNELS or len(set(names)) != len(names)
            or any(not name or len(name) > 128 for name in names)):
        raise ValueError('Invalid or duplicate face morph names.')
    if weights.ndim != 2 or weights.shape[1] != len(names) or not 1 <= len(weights) <= MAX_FRAMES:
        raise ValueError('Invalid face coefficient dimensions.')
    if (not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1)
            or not np.isfinite(fps) or not 1 <= fps <= 240
            or not np.isfinite(start) or start < 0):
        raise ValueError('Invalid face coefficients or timing.')
    if detected.shape != (len(weights),) or detected.dtype.kind != 'b':
        raise ValueError('Invalid face detection mask.')
    for name in ('predicted', 'interpolated', 'blocked'):
        value = extra.get(name)
        if value is not None and (value.shape != detected.shape or value.dtype.kind != 'b'):
            raise ValueError('Invalid face '+name+' mask.')
    for name in ('predicted', 'interpolated'):
        if name in extra and np.any(extra[name] & detected):
            raise ValueError('Estimated face frames cannot be marked detected.')
    if 'predicted' in extra and 'interpolated' in extra and np.any(extra['predicted'] & extra['interpolated']):
        raise ValueError('Face prediction and interpolation masks must be distinct.')
    if 'blocked' in extra and np.any(extra['blocked'] & detected):
        raise ValueError('Blocked face frames cannot be marked detected.')
    if 'blocked' in extra:
        for name in ('predicted', 'interpolated'):
            if name in extra and np.any(extra['blocked'] & extra[name]):
                raise ValueError('Blocked face frames cannot contain estimated expressions.')
    for name, shape in (('raw_weights', weights.shape), ('reliability', detected.shape)):
        value = extra.get(name)
        if value is not None:
            if (value.shape != shape or value.dtype.kind not in 'fiu'
                    or not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1)):
                raise ValueError('Invalid face '+name+'.')
            extra[name] = value.astype(np.float32)
    return dict(morph_names=names, weights=weights, fps=fps,
                start_seconds=start, detected=detected, **extra)


def sample_track(track, start_seconds, frames, fps):
    """Linear samples on absolute video time; outside the captured interval is neutral."""
    frames = int(frames)
    if not 1 <= frames <= MAX_FRAMES or not np.isfinite(fps) or not 1 <= fps <= 240:
        raise ValueError('Invalid target face timeline.')
    if not np.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError('Invalid target face start time.')
    position = (start_seconds - track['start_seconds'] + np.arange(frames) / fps) * track['fps']
    last = len(track['weights']) - 1
    inside = (position >= -1e-6) & (position <= last + 1e-6)
    position = np.clip(position, 0, last)
    left = np.floor(position).astype(np.int64)
    right = np.minimum(left + 1, last)
    fraction = (position - left).astype(np.float32)[:, None]
    result = track['weights'][left] * (1 - fraction) + track['weights'][right] * fraction
    result[~inside] = 0
    return result.astype(np.float32)


MMD_NATIVE_CHANNELS = ('まばたき', 'ウィンク', 'ウィンク右', 'あ', 'い', 'う', 'お',
                       'にやり', 'への字', '上', '下', '困る')


def is_native_track(names):
    from mocap.face.blendshapes import BLENDSHAPE_NAMES
    return any(name in BLENDSHAPE_NAMES for name in names)


def map_shape_channels(names, weights, available):
    """Prefer native coefficients, with explicit lower-dimensional rig fallbacks.

    MMD vowel shapes are geometric approximations, not recognized speech sounds.
    A rig needs the corresponding actual shape keys; no arbitrary mesh binding
    or emotional blendshape inference is performed here. Legacy five-channel
    captures retain their original mapping.
    """
    if is_native_track(names):
        return _native_shape_channels(names, weights, available)
    return _legacy_shape_channels(names, weights, available)


def _native_shape_channels(names, weights, available):
    from mocap.face.blendshapes import BLENDSHAPE_NAMES
    available = set(available)
    source = {name: weights[:, i] for i, name in enumerate(names) if name != '_neutral'}
    result = {name: source[name] for name in source if name in available}
    used = set(result)
    zero = np.zeros(len(weights), np.float32)

    def get(name):
        return source.get(name, zero)

    def mean(left, right):
        return (get(left)+get(right))*.5

    def choose(*candidates):
        return next((name for name in candidates if name in available), None)

    # Explicit ARKit-side suffix aliases. Never apply an alias as well as its
    # canonical key: models sometimes expose both on the same vertices.
    for name in BLENDSHAPE_NAMES:
        if name in source and name not in used and name.endswith(('Left', 'Right')):
            suffix = '_L' if name.endswith('Left') else '_R'
            alias = name.removesuffix('Left').removesuffix('Right')+suffix
            if alias in available:
                result[alias] = source[name]
                used.add(name)

    # Independent ARKit/VRM eye keys take full left/right weights. A common MMD
    # blink plus winks is decomposed into min(L,R) and nonnegative residuals,
    # so the shared eyelid displacement is never added a second time.
    left = choose('eyeBlinkLeft', 'eyeBlink_L', 'blinkLeft', 'Fcl_EYE_Close_L', 'ウィンク')
    right = choose('eyeBlinkRight', 'eyeBlink_R', 'blinkRight', 'Fcl_EYE_Close_R', 'ウィンク右')
    common = choose('まばたき', 'blink', 'Fcl_EYE_Close')
    if 'eyeBlinkLeft' in source or 'eyeBlinkRight' in source:
        lvalue, rvalue = get('eyeBlinkLeft'), get('eyeBlinkRight')
        split = left and right and left != 'ウィンク' and right != 'ウィンク右'
        shared = np.minimum(lvalue, rvalue) if common and not split else zero
        if common and not split:
            result[common] = shared
        if left:
            result[left] = np.maximum(0, lvalue-shared)
        if right:
            result[right] = np.maximum(0, rvalue-shared)

    articulation = ('jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthStretchLeft', 'mouthStretchRight')
    if not used.intersection(articulation) and any(name in source for name in articulation):
        opened = get('jawOpen')*(1-get('mouthClose'))
        pucker, funnel = get('mouthPucker'), get('mouthFunnel')
        candidates = (
            (('あ', 'aa', 'A', 'Fcl_MTH_A'), opened*(1-np.maximum(pucker, funnel))),
            (('い', 'ih', 'I', 'Fcl_MTH_I'), mean('mouthStretchLeft', 'mouthStretchRight')),
            (('う', 'ou', 'U', 'Fcl_MTH_U'), pucker),
            (('お', 'oh', 'O', 'Fcl_MTH_O'), funnel),
        )
        vowels = {target: value for aliases, value in candidates if (target := choose(*aliases))}
        # Simultaneously additive phoneme shapes are usually not authored to
        # exceed a total of one; normalize only the shapes the rig can express.
        total = np.maximum(1., sum(vowels.values(), zero.copy()))
        result.update({name: value/total for name, value in vowels.items()})

    def fallback(sources, aliases, value):
        if not used.intersection(sources) and any(name in source for name in sources):
            target = choose(*aliases)
            if target:
                result[target] = value

    smile_sources = ('mouthSmileLeft', 'mouthSmileRight')
    smile = mean(*smile_sources)
    fallback(smile_sources, ('にやり', 'Fcl_MTH_Fun'), smile)
    # ワ is an open smiling mouth, not a substitute for a closed-lip smile.
    if not used.intersection(smile_sources) and not choose('にやり', 'Fcl_MTH_Fun'):
        fallback(smile_sources, ('ワ',), smile*get('jawOpen')*(1-get('mouthClose')))
    fallback(('mouthFrownLeft', 'mouthFrownRight'), ('への字', 'Fcl_MTH_Sorrow'),
             mean('mouthFrownLeft', 'mouthFrownRight'))
    fallback(('browOuterUpLeft', 'browOuterUpRight'), ('上', 'Fcl_BRW_Up'),
             mean('browOuterUpLeft', 'browOuterUpRight'))
    fallback(('browDownLeft', 'browDownRight'), ('下', 'Fcl_BRW_Angry'),
             mean('browDownLeft', 'browDownRight'))
    fallback(('browInnerUp',), ('困る', 'Fcl_BRW_Sorrow'), get('browInnerUp'))
    return {name: np.clip(value, 0, 1).astype(np.float32) for name, value in result.items()}


def mapped_source_channels(names, available):
    """Diagnose source coverage even for nonlinear common-blink/vowel mapping."""
    n = len(names)
    eye = np.eye(n, dtype=np.float32)
    probes = np.vstack((np.zeros((1,n), np.float32), eye,
                        np.full((1,n), .5, np.float32), .5*(1-eye)))
    mapped = map_shape_channels(names, probes, available)
    if not mapped:
        return np.zeros(n, bool)
    values = np.column_stack(list(mapped.values()))
    return (np.any(np.abs(values[1:n+1]-values[0]) > 1e-6, axis=1)
            | np.any(np.abs(values[n+2:]-values[n+1]) > 1e-6, axis=1))


def _legacy_shape_channels(names, weights, available):
    """The original reze-mipo five-channel mapping, retained byte-for-byte."""
    available = set(available)
    source = {name: weights[:, i] for i, name in enumerate(names)}
    result = {name: source[name] for name in names if name in available}
    zero = np.zeros(len(weights), np.float32)
    blink = source.get('まばたき', zero)
    left = source.get('ウィンク', zero)
    right = source.get('ウィンク右', zero)
    # Prefer the native combined blink if available; otherwise distribute it
    # to the two independent eye channels. These are explicit semantic aliases.
    for combined in ('blink', 'Fcl_EYE_Close'):
        if combined in available and 'まばたき' not in available:
            result[combined] = blink
            break
    combined_exists = any(n in result for n in ('まばたき', 'blink', 'Fcl_EYE_Close'))
    for alternatives, wink in [(('eyeBlinkLeft', 'blinkLeft', 'Fcl_EYE_Close_L'), left),
                                (('eyeBlinkRight', 'blinkRight', 'Fcl_EYE_Close_R'), right)]:
        for name in alternatives:
            if name in available:
                result.setdefault(name, np.clip(wink + (zero if combined_exists else blink), 0, 1))
                break
    if 'あ' not in available and 'あ' in source:
        for name in ('jawOpen', 'aa', 'A', 'Fcl_MTH_A'):
            if name in available:
                result.setdefault(name, source['あ'])
                break
    if 'ワ' not in available and 'ワ' in source:
        smile = [name for name in ('mouthSmileLeft', 'mouthSmileRight') if name in available]
        for name in smile:
            result.setdefault(name, source['ワ'])
        if not smile:
            for name in ('happy', 'Joy', 'Fcl_MTH_Fun'):
                if name in available:
                    result.setdefault(name, source['ワ'])
                    break
    return result
