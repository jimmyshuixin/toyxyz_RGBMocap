"""Optional face runtime, isolated from every body-inference environment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mocap.paths import ROOT


def identity() -> str:
    return hashlib.sha256((ROOT / 'installer/face-runtime-lock.json').read_bytes()).hexdigest()


def _configuration() -> dict:
    receipt = json.loads((ROOT / '.setup/face-runtime.json').read_text(encoding='utf8'))
    if receipt.get('schema') != 1 or receipt.get('identity') != identity():
        raise ValueError('Face runtime dependencies changed; run Setup_Face.cmd again.')
    folder = (ROOT / receipt['folder']).resolve()
    base = (ROOT / 'runtime_face').resolve()
    if not folder.is_relative_to(base) or folder == base:
        raise ValueError('Invalid face runtime location.')
    for relative, expected in receipt['verified_files'].items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder):
            raise ValueError('Invalid face runtime receipt path.')
        info = path.stat()
        if [info.st_size, info.st_mtime_ns] != expected:
            raise ValueError('Face runtime files changed; run Setup_Face.cmd again.')
    python = folder / 'Scripts/python.exe'
    model = folder / 'face_landmarker.task'
    if not python.is_file() or not model.is_file():
        raise ValueError('Face runtime is incomplete; run Setup_Face.cmd again.')
    return dict(receipt, folder=folder, python=python, model=model)


def status() -> dict:
    """Cheap GUI readiness check; native model verification happens during setup."""
    try:
        config = _configuration()
        return dict(ready=True, message='Face capture ready (MediaPipe 0.10.32, CPU).',
                    folder=str(config['folder']))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        detail = str(exc) if isinstance(exc, ValueError) else 'Face capture is not installed.'
        return dict(ready=False, message=detail + ' Run Setup_Face.cmd or use Setup Face.')


def model_path() -> Path:
    return _configuration()['model']


def command(request_path: str | Path) -> list[str]:
    config = _configuration()
    code = ('import runpy,sys;sys.path.insert(0,' + repr(str(ROOT)) + ');'
            "runpy.run_module('mocap.face.worker',run_name='__main__')")
    return [str(config['python']), '-I', '-X', 'utf8', '-B', '-c', code,
            '--request', str(Path(request_path).resolve())]


def pipeline_command(args) -> list[str]:
    config = _configuration()
    code = ('import runpy,sys;sys.path.insert(0,' + repr(str(ROOT)) + ');'
            "runpy.run_module('mocap.face.pipeline',run_name='__main__')")
    return [str(config['python']), '-I', '-X', 'utf8', '-B', '-c', code, *map(str,args)]


def setup_command() -> list[str]:
    python = ROOT / 'runtime/python.exe'
    if not python.is_file():
        raise RuntimeError('Run install.cmd to prepare the base Python runtime first.')
    return [str(python), '-I', '-X', 'utf8', '-B', str(ROOT / 'tools/setup_face.py')]
