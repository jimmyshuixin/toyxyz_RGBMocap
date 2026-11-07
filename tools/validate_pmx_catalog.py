"""Local-only, repeatable real PMX import audit; never changes source models.

Run with the installed Python and explicit --model-root (repeatable), --output
and --blender paths. Each run requires a new output directory. Model-derived
artifacts and logs stay there; do not publish them with application source.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blender', type=Path, default=ROOT / 'runtime_character/blender.exe')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--source-snapshot', type=Path, help='Explicit frozen retarget source directory for baseline comparisons')
    args = parser.parse_args()
    source_dir = (args.source_snapshot or ROOT / 'mocap/retarget').resolve(strict=True)
    required = ('__init__.py', 'blender_io.py', 'mapping.py', 'file_paths.py')
    if not source_dir.is_dir() or any(not (source_dir / name).is_file() for name in required):
        parser.error('The retarget source snapshot is missing required Python modules.')
    roots = [root.resolve(strict=True) for root in args.model_root]
    sources = sorted({p.resolve() for root in roots for p in root.rglob('*') if p.suffix.lower() == '.pmx'}, key=lambda p: str(p).casefold())
    if not sources:
        parser.error('No PMX files were found in the supplied roots.')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'imports').mkdir()
    cases = [dict(id=f'case-{i:02d}', source=str(source), bytes=source.stat().st_size,
                  source_sha256=digest(source), output=str(out / 'imports' / f'case-{i:02d}'))
             for i, source in enumerate(sources, 1)]
    write_json(out / 'catalog.json', cases)
    snapshots = out / 'source-snapshot'
    snapshots.mkdir()
    for path in source_dir.glob('*.py'):
        (snapshots / path.name).write_bytes(path.read_bytes())
    write_json(out / 'run.json', dict(roots=list(map(str, roots)), blender=str(args.blender),
                                    source_hashes={p.name: digest(p) for p in snapshots.glob('*.py')}))
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    for case in cases:
        log = out / (case['id'] + '.log')
        command = [str(args.blender.resolve()), '-b', '--factory-startup', '--disable-autoexec',
                   '--threads', '3', '--python-exit-code', '1', '--python',
                   str(ROOT / 'tests/blender_pmx_catalog_case.py'), '--', case['source'], case['output']]
        start = time.perf_counter()
        print(f"IMPORT {case['id']} {case['source']}", flush=True)
        with log.open('w', encoding='utf-8') as stream:
            try:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                         env=environment, timeout=args.timeout,
                                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                case['exit_code'] = process.returncode
            except subprocess.TimeoutExpired:
                case['exit_code'] = None
                case['exception'] = f'Import exceeded {args.timeout} seconds'
        case['wall_seconds'] = time.perf_counter() - start
        diagnostic = Path(case['output']) / 'catalog-case.json'
        if diagnostic.exists():
            case['diagnostic'] = json.loads(diagnostic.read_text(encoding='utf-8'))
        case['source_unchanged'] = digest(Path(case['source'])) == case['source_sha256']
        write_json(out / 'catalog.json', cases)
        print(json.dumps({k: case.get(k) for k in ('id', 'exit_code', 'wall_seconds', 'source_unchanged')}), flush=True)
    print(f'CATALOG {out / "catalog.json"}', flush=True)


if __name__ == '__main__':
    main()
