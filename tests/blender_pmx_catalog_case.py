"""Actual Blender import and bounded diagnostics for one local PMX fixture."""
import json
from pathlib import Path
import sys
import time
import traceback
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    source, destination = map(Path, sys.argv[sys.argv.index('--') + 1:])
    destination.mkdir(parents=True, exist_ok=False)
    # Freeze production retarget code per catalog even while fixes are prepared.
    snapshot = destination.parents[1] / 'source-snapshot'
    if snapshot.is_dir():
        class FrozenLoader(importlib.machinery.SourceFileLoader):
            def get_code(self, fullname):
                return compile((snapshot / Path(self.path).name).read_bytes(), self.path, 'exec')
        class FrozenFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith('mocap.retarget.'):
                    filename = fullname.rsplit('.', 1)[1] + '.py'
                    if (snapshot / filename).is_file():
                        original = str(ROOT / 'mocap/retarget' / filename)
                        return importlib.util.spec_from_loader(fullname, FrozenLoader(fullname, original))
                    raise ImportError('Frozen retarget snapshot is missing: ' + filename)
        sys.meta_path.insert(0, FrozenFinder())
    from mocap.retarget import blender_io
    result = dict(source=str(source), blender_version=bpy.app.version_string,
                  status='failed', timings_seconds={})
    started = time.perf_counter()
    # Measure the production functions without changing their inputs or outputs.
    from mocap.retarget import blender_geometry, blender_textures
    for module, name in ((blender_geometry, 'register'), (blender_io, 'preview'),
                         (blender_textures, 'extract')):
        original = getattr(module, name)
        def timed(*args, _original=original, _name=name, **kwargs):
            start = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                result['timings_seconds'][_name] = time.perf_counter() - start
        setattr(module, name, timed)
    try:
        blender_io.import_character(source, destination)
        with np.load(destination / 'preview.npz', allow_pickle=False) as data:
            vertices, weights, rows = data['vertices'], data['weights'], data['rows']
            assert np.isfinite(vertices).all() and np.isfinite(weights).all()
            assert len(rows) == len(vertices)
            sums = np.add.reduceat(weights[:, 1], rows[:, 0])
            assert np.allclose(sums, 1, atol=1e-5)
            result['preview'] = dict(vertices=len(vertices), triangles=len(data['faces']),
                                     minimum=vertices.min(0).tolist(), maximum=vertices.max(0).tolist(),
                                     max_weight_sum_error=float(np.abs(sums-1).max()))
        result['import'] = json.loads((destination / 'import.json').read_text(encoding='utf-8'))
        result['status'] = 'passed'
    except Exception as error:
        result['exception'] = f'{type(error).__name__}: {error}'
        result['traceback'] = traceback.format_exc()
        traceback.print_exc()
    result['images'] = [dict(name=image.name, filepath=image.filepath, size=list(image.size),
                             packed=bool(image.packed_file)) for image in bpy.data.images
                        if image.source == 'FILE']
    result['materials'] = [dict(name=mat.name, diffuse=list(mat.diffuse_color),
                               backface_culling=mat.use_backface_culling)
                           for mat in bpy.data.materials]
    result['elapsed_seconds'] = time.perf_counter() - started
    result['loaded_source_sha256'] = {name: hashlib.sha256((snapshot / Path(module.__file__).name
                                                         if (snapshot / Path(module.__file__).name).is_file()
                                                         else Path(module.__file__)).read_bytes()).hexdigest()
                                     for name, module in tuple(sys.modules.items())
                                     if name.startswith('mocap.retarget.') and getattr(module, '__file__', None)}
    (destination / 'catalog-case.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('CATALOG_RESULT ' + json.dumps({k: result.get(k) for k in ('source', 'status', 'exception', 'elapsed_seconds')}, ensure_ascii=False), flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
