"""Install and verify optional CPU face capture without changing body runtimes.

Each attempt prepares a new venv generation. The previous working registration
is replaced only after package checks and a real FaceLandmarker inference pass.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mocap.storage import atomic_json
from mocap.face.runtime import identity, _configuration
from tools.pinned_download import download, valid


@contextmanager
def setup_lock():
    # Independent from the app lock: this setup touches only its own generation.
    import msvcrt
    folder = ROOT / '.setup'
    folder.mkdir(exist_ok=True)
    with (folder / 'face-setup.lock').open('a+b') as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError('Another face setup is already running.') from None
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def verify(python: Path, model: Path, spec: dict) -> None:
    if not valid(model, spec['model']):
        raise ValueError('Face model does not match its pinned size and SHA-256.')
    expected = {row['name']: row['version'] for row in spec['wheels']}
    code = (
        'import importlib.metadata as m;'
        'expected=' + repr(expected) + ';'
        'assert all(m.version(k)==v for k,v in expected.items()), "Package version mismatch";'
        'import mediapipe as mp,numpy as np,cv2;from pathlib import Path;'
        'from mediapipe.tasks.python import vision;'
        'options=vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions('
        'model_asset_buffer=Path(' + repr(str(model)) + ').read_bytes(),delegate=mp.tasks.BaseOptions.Delegate.CPU),'
        'running_mode=vision.RunningMode.VIDEO,num_faces=1,output_face_blendshapes=True);'
        'detector=vision.FaceLandmarker.create_from_options(options);'
        'result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,'
        'data=np.zeros((128,128,3),dtype=np.uint8)),0);'
        'detector.close();'
        'assert result.face_landmarks==[], "Unexpected blank-image detection";'
        'print("FaceLandmarker CPU inference verified; versions and model hash match.",flush=True)'
    )
    subprocess.run([str(python), '-I', '-X', 'utf8', '-B', '-c', code],
                   check=True, timeout=120)
    subprocess.run([str(python), '-I', '-X', 'utf8', '-m', 'pip', 'check'],
                   check=True, timeout=60)


def prepare() -> Path:
    if os.name != 'nt' or sys.version_info[:2] != (3, 10):
        raise RuntimeError('Face setup requires the installed Windows Python 3.10 runtime.')
    spec = json.loads((ROOT / 'installer/face-runtime-lock.json').read_text(encoding='utf8'))
    pin = identity()
    try:
        current = _configuration()
    except (OSError, ValueError, KeyError, TypeError):
        current = None
    if current is not None:
        print('Checking the existing face runtime...', flush=True)
        try:
            verify(current['python'], current['model'], spec)
        except (OSError, ValueError, subprocess.SubprocessError):
            print('Existing face runtime needs repair; preparing a new generation.', flush=True)
        else:
            print('Face capture is ready. Existing generation retained.', flush=True)
            return current['folder']

    base_python = ROOT / 'runtime/python.exe'
    if not base_python.is_file():
        raise RuntimeError('Run install.cmd first.')
    cache = ROOT / '.setup/face-downloads'
    cache.mkdir(parents=True, exist_ok=True)
    folder = ROOT / 'runtime_face' / ('face-' + pin[:12] + '-' + uuid.uuid4().hex[:8])
    print('Creating isolated face capture runtime: ' + str(folder), flush=True)
    subprocess.run([str(base_python), '-I', '-X', 'utf8', '-m', 'venv', '--copies', str(folder)],
                   check=True, timeout=180)
    python = folder / 'Scripts/python.exe'
    requirements = folder / 'pinned-requirements.txt'
    requirements.write_text('\n'.join(row['url'] + ' --hash=sha256:' + row['sha256']
                                     for row in spec['wheels']) + '\n', encoding='utf8')
    print('Installing hash-pinned face packages into the isolated runtime...', flush=True)
    subprocess.run([str(python), '-I', '-X', 'utf8', '-m', 'pip', 'install',
                    '--disable-pip-version-check', '--no-index', '--no-deps',
                    '--require-hashes', '--only-binary=:all:', '--retries', '3',
                    '--timeout', '30', '--cache-dir', str(cache / 'pip'),
                    '-r', str(requirements)], check=True, timeout=1800)
    print('Preparing the pinned MediaPipe FaceLandmarker model...', flush=True)
    archive = download(cache, 'face-landmarker-v1', spec['model'])
    model = folder / 'face_landmarker.task'
    shutil.copyfile(archive, model)
    verify(python, model, spec)
    if identity() != pin:
        raise RuntimeError('Face dependency lock changed during setup. Run setup again.')
    # Watch native binaries and package entry points without a full tree scan in
    # the GUI. Full package versions and model hash are checked by verify above.
    observed = [python, folder / 'pyvenv.cfg', model]
    packages = folder / 'Lib/site-packages'
    observed.extend(packages / name / '__init__.py'
                    for name in ('mediapipe', 'numpy', 'cv2'))
    observed.extend((packages / 'mediapipe').rglob('*.dll'))
    observed.extend((packages / 'mediapipe').rglob('*.pyd'))
    files = {p.relative_to(folder).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
             for p in observed}
    atomic_json(folder / 'PROVENANCE.json', spec)
    atomic_json(ROOT / '.setup/face-runtime.json', dict(
        schema=1, identity=pin, folder=folder.relative_to(ROOT).as_posix(),
        verified_files=files, verification='FaceLandmarker CPU detect_for_video on blank image; pip check'))
    print('Face capture is ready. Body inference runtimes were preserved.', flush=True)
    return folder


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf8', errors='backslashreplace')
    try:
        with setup_lock():
            prepare()
    except Exception as exc:
        print('Face setup failed: ' + str(exc), file=sys.stderr, flush=True)
        sys.exit(1)
