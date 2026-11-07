"""Optional per-actor facial pass over the unchanged CFR body working video.

The independent face receipt never participates in body inference identities.
Completed generations are retained; only the small current pointer is replaced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

from mocap.paths import ROOT
from mocap.storage import atomic_json, sha256

SUPPORTED_MODELS = {'exmr', 'flowhmr', 'gvhmr'}
MAX_FRAMES = 54_000
SCHEMA = 2


class FaceSkipped(ValueError):
    """Body motion remains usable when its face cannot be identified."""


def options(value=None):
    value = value or {}
    if not isinstance(value, dict):
        raise ValueError('Face options must be an object.')
    for key in ('enabled', 'smoothing'):
        if key in value and not isinstance(value[key], bool):
            raise ValueError('Face option ' + key + ' must be a boolean.')
    return dict(enabled=value.get('enabled', False), smoothing=value.get('smoothing', True))


def _read_json(path):
    if path.stat().st_size > 1024**2:
        raise ValueError('Face/project metadata exceeds the size limit.')
    return json.loads(path.read_text(encoding='utf8'))


def _stat(path):
    value = path.stat()
    return [value.st_size, value.st_mtime_ns]


def _state(project):
    return _read_json(Path(project) / 'project.json')


def _signature(project, state, config):
    """Bind only relevant source timing/identity and independent face code."""
    project = Path(project)
    import math
    n = int(state['frames'])
    fps, start, end = (float(state[key]) for key in ('fps', 'start', 'end'))
    if (not 1 <= n <= MAX_FRAMES or not 1 <= fps <= 120 or start < 0 or end <= start
            or not all(math.isfinite(x) for x in (fps, start, end))
            or abs((end - start) * fps - n) > 1.01):
        raise ValueError('Body working video has invalid face-capture timing.')
    model = state.get('pipeline', {}).get('model', 'gvhmr')
    return dict(schema=SCHEMA, options=config, source=_stat(project/'input.mp4'),
                tracking=_stat(project/'tracking.npz'),
                actor={key:state.get(key) for key in ('person_id', 'person_frames', 'person_fingerprint')},
                timeline=dict(model=model, frames=n, fps=fps, start=start, end=end),
                code={name:sha256(Path(__file__).with_name(name)) for name in ('pipeline.py', 'capture.py', 'blendshapes.py', 'temporal.py')},
                runtime=sha256(ROOT/'installer/face-runtime-lock.json'))


def _child(project, relative):
    if not isinstance(relative, str):
        raise ValueError('Invalid face generation path.')
    base = (Path(project) / 'face').resolve()
    path = (Path(project) / relative).resolve()
    if not path.is_relative_to(base) or path == base or path.name != 'face.npz':
        raise ValueError('Face track lies outside its project generation.')
    return path


def load_current(project):
    """Read a valid enabled automatic track, or None; never run inference here."""
    project = Path(project)
    try:
        state = _state(project)
        config = options(state.get('face_options'))
        if not config['enabled'] or state.get('pipeline', {}).get('model', 'gvhmr') not in SUPPORTED_MODELS:
            return None
        receipt = _read_json(project/'face_capture.json')
        if receipt.get('schema') != SCHEMA or receipt.get('status') != 'ready':
            return None
        if receipt.get('signature') != _signature(project, state, config):
            return None
        path = _child(project, receipt['track'])
        if _stat(path) != receipt['track_stat']:
            return None
        from mocap.retarget.face_track import load_track
        track = load_track(path)
        if (len(track['weights']) != int(state['frames'])
                or abs(track['fps'] - float(state['fps'])) > 1e-6
                or abs(track['start_seconds'] - float(state['start'])) > 1e-6):
            return None
        return path
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        return None


def actor_crops(project, state):
    """Dynamic upper-body crops belong to one tracked actor; gaps are skipped.

    Crops are a detection aid, not a new tracking model. Rejecting multiple
    detected faces is conservative near overlaps; it cannot fix an identity
    switch already present in the body tracker.
    """
    import numpy as np
    path = Path(project)/'tracking.npz'
    if not path.is_file():
        raise FaceSkipped('No actor tracking is available for automatic face capture.')
    if path.stat().st_size > 64*1024**2:
        raise FaceSkipped('Actor tracking exceeds the face capture size limit.')
    with np.load(path, allow_pickle=False) as data:
        boxes = np.asarray(data['boxes'], dtype=np.float64)
        observed = np.asarray(data['observed']) if 'observed' in data else np.ones(len(boxes), bool)
    n = int(state['frames'])
    if boxes.shape != (n,4) or observed.shape != (n,):
        raise FaceSkipped('Actor tracking does not match the working video; automatic face capture skipped.')
    if observed.dtype.kind != 'b':
        raise FaceSkipped('Invalid actor detection flags.')
    def crop(index, image_shape):
        if not 0 <= index < n or not observed[index]:
            return None
        height, width = image_shape[:2]
        left, top, right, bottom = boxes[index]
        if not np.isfinite(boxes[index]).all() or right <= left or bottom <= top:
            return None
        # Restrict to the tracked actor rather than applying a global face to
        # every person. The upper 55% accommodates shoulders and raised arms.
        bottom = top + .55*(bottom-top)
        left, right = np.clip([left, right], 0, width)
        top, bottom = np.clip([top, bottom], 0, height)
        if right-left < 8 or bottom-top < 8:
            return None
        return [float(left/width), float(top/height), float(right/width), float(bottom/height)]
    return crop


def run_project(project, face_options=None, emit=lambda *_a, **_k:None, *, force=False):
    project = Path(project).resolve()
    state = _state(project)
    config = options(state.get('face_options') if face_options is None else face_options)
    pointer = project/'face_capture.json'
    def skip(message, status='skipped'):
        # Stop automatic consumption of any previous track, retaining its files.
        atomic_json(pointer, dict(schema=SCHEMA, enabled=config['enabled'], status=status,
                                  message=message, options=config))
        emit('info', stage='face_capture', message=message)
        emit('face_ready', project=str(project), status=status, message=message)
        return dict(status=status, message=message)
    if not config['enabled']:
        return skip('Automatic face capture disabled; previous face generations were kept.', 'disabled')
    if state.get('pipeline', {}).get('model', 'gvhmr') not in SUPPORTED_MODELS:
        return skip('Automatic face capture currently requires ExMR, FlowHMR or GVHMR actor tracking.')
    try:
        crops = actor_crops(project, state)
    except (FaceSkipped, OSError, ValueError, KeyError) as exc:
        return skip(str(exc))
    signature = _signature(project, state, config)
    existing = load_current(project) if not force else None
    if existing is not None:
        receipt = _read_json(pointer)
        if receipt.get('signature') == signature:
            emit('info', stage='face_capture', message='Using current facial capture for this actor.')
            emit('face_ready', project=str(project), status='ready', track=str(existing), cached=True,
                 detected_frames=receipt['detected_frames'], frames=int(state['frames']))
            return receipt
    from . import capture, runtime
    output = project/'face'/('generation-'+uuid.uuid4().hex)
    job = dict(video=str(project/'input.mp4'), output_dir=str(output), model_path=str(runtime.model_path()),
               start_seconds=0., end_seconds=int(state['frames'])/float(state['fps']),
               fps=float(state['fps']), timeline_offset_seconds=float(state['start']),
               smoothing=config['smoothing'], mirror=True, reject_multiple_faces=True)
    def progress(event):
        emit('progress', stage='face_capture', current=event.get('frame',0),
             total=event.get('total',state['frames']), message=event.get('message','Capturing actor expressions'))
    try:
        result = capture.run(job, progress, crop_for_frame=crops)
    except RuntimeError as exc:
        if str(exc).startswith('No face was detected'):
            return skip('No unambiguous face was detected for this actor. Body motion is ready; try a clearer face video or manual Face Capture.')
        raise
    if signature != _signature(project, _state(project), config):
        raise RuntimeError('Body video or actor tracking changed during face capture; new face result was not activated.')
    path = Path(result['npz'])
    receipt = dict(schema=SCHEMA, enabled=True, status='ready', signature=signature,
                   track=path.relative_to(project).as_posix(), track_stat=_stat(path),
                   detected_frames=result['detected_frames'], frames=result['frame_count'],
                   predicted_frames=result.get('predicted_frames',0),
                   interpolated_frames=result.get('interpolated_frames',0))
    atomic_json(pointer, receipt)
    message=(f"Facial capture ready: {result['detected_frames']}/{result['frame_count']} detected frames; "
             f"{receipt['interpolated_frames']} interpolated, {receipt['predicted_frames']} predicted/fading.")
    emit('info', stage='face_capture', message=message)
    emit('face_ready', project=str(project), status='ready', track=str(path),
         detected_frames=result['detected_frames'], frames=result['frame_count'], message=message)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description='Optional actor-specific facial capture')
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--face-options')
    parser.add_argument('--force', action='store_true')
    args=parser.parse_args(argv)
    def emit(kind, **value):
        print('@@MOCAP@@'+json.dumps(dict(type=kind, **value), ensure_ascii=True, allow_nan=False), flush=True)
    try:
        run_project(args.project, json.loads(args.face_options) if args.face_options else None, emit, force=args.force)
        return 0
    except KeyboardInterrupt:
        emit('info', stage='face_capture', message='Face capture cancelled; body motion was kept.')
        return 130
    except Exception as exc:
        emit('failed', stage='face_capture', message='Face capture failed: '+str(exc))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
