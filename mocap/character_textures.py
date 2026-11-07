"""Load budgeted texture caches on the character I/O thread, never during painting."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .retarget.file_paths import io_path

CACHE = 'preview-textures-v2'


def load(source, asset, cancelled):
    source = Path(source).resolve()
    folder = Path(io_path(source / CACHE))
    if not (folder / 'manifest.json').is_file():
        from .retarget.runtime import blender
        entry = Path(__file__).parent / 'retarget' / 'blender_entry.py'
        command = [str(blender()), '-b', '--factory-startup', '--disable-autoexec', '--threads', '3',
                   '--python-exit-code', '1', '--python', str(entry), '--', 'textures', str(source)]
        env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
        with open(io_path(source / 'texture-preview.log'), 'w', encoding='utf8') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                deadline = time.monotonic() + 180
                while process.poll() is None:
                    if cancelled():
                        raise InterruptedError('Texture preparation cancelled')
                    if time.monotonic() > deadline:
                        raise RuntimeError('Texture preparation timed out. See texture-preview.log in the character asset folder.')
                    time.sleep(.1)
                if process.returncode:
                    raise RuntimeError('Cannot prepare character preview textures. See texture-preview.log in the character asset folder.')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
    manifest = folder / 'manifest.json'
    if manifest.stat().st_size > 4 * 1024**2:
        raise ValueError('Character texture metadata exceeds limits')
    meta = json.loads(manifest.read_text(encoding='utf8'))
    validate_geometry(meta, asset)
    if (folder / 'corners.npy').stat().st_size > asset['faces'].size*12 + 4096:
        raise ValueError('Character UV cache exceeds geometry limits')
    corners = np.load(folder / 'corners.npy', allow_pickle=False)
    if corners.shape != (asset['faces'].size, 3) or corners.dtype != np.float32 or not np.isfinite(corners).all():
        raise ValueError('Invalid character texture UV cache')
    ids = corners[:, 0]
    if np.any(ids < 0) or np.any(ids >= len(asset['vertices'])) or np.any(ids != np.floor(ids)):
        raise ValueError('Invalid character texture vertex indices')
    images, total = {}, corners.nbytes
    for key, size in meta['images'].items():
        if not key.startswith('image') or not key[5:].isdigit() or len(size) != 2:
            raise ValueError('Invalid character texture image entry')
        w, h = size
        if not isinstance(w, int) or not isinstance(h, int) or min(w, h) < 1 or w*h*4 > 256*1024**2:
            raise ValueError('Character texture size exceeds limits')
        total += w*h*4
        if total > 576*1024**2:
            raise ValueError('Character textures exceed memory limits')
        filename = folder / (key + '.npy')
        if filename.stat().st_size > w*h*4 + 4096:
            raise ValueError('Invalid character texture file size')
        pixels = np.load(filename, allow_pickle=False)
        if pixels.shape != (h, w, 4) or pixels.dtype != np.uint8:
            raise ValueError('Invalid character texture pixel data')
        images[key] = pixels
    end = 0
    for batch in meta['batches']:
        if (batch['start'] != end or not isinstance(batch['count'], int) or batch['count'] <= 0
                or batch['count'] % 3 or batch['image'] is not None and batch['image'] not in images
                or len(batch['color']) != 4 or not np.isfinite(batch['color']).all()):
            raise ValueError('Invalid character texture material batch')
        end += batch['count']
    if end != len(corners):
        raise ValueError('Incomplete character texture triangles')
    return dict(key=str(source), corners=corners, images=images, batches=meta['batches'], metadata=meta, bytes=total)


def validate_geometry(meta, asset):
    if (meta.get('version') != 2 or meta['vertices'] != len(asset['vertices'])
            or meta['faces_sha256'] != hashlib.sha256(asset['faces'].tobytes()).hexdigest()):
        raise ValueError('Character texture and animation geometry do not match. Reimport the character.')
