"""Verify bounded texture extraction from a separately supplied real rest asset."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mocap.retarget.blender_textures import CACHE, extract
from mocap.retarget.blender_materials import base_surface

parser = argparse.ArgumentParser()
parser.add_argument('--asset', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
args.output.mkdir(parents=True, exist_ok=False)
shutil.copyfile(args.asset/'preview.npz', args.output/'preview.npz')
bpy.ops.wm.open_mainfile(filepath=str(args.asset/'pmx-rest.blend'), load_ui=False)
before = {image.name: list(image.size) for image in bpy.data.images}
start = time.perf_counter()
metadata = extract(args.output)
elapsed = time.perf_counter()-start
assert before == {image.name: list(image.size) for image in bpy.data.images}
assert sum(w*h*4 for w, h in metadata['images'].values()) <= 512*1024**2
assert all(w*h*4 <= 256*1024**2 for w, h in metadata['images'].values())
assert extract(args.output) == metadata, 'Existing-cache return must retain warnings'
maximum = 0
for key, change in metadata['downsampled'].items():
    image = bpy.data.images[change['name']]
    width, height = change['original']
    factor = change['factor']
    cached = np.load(args.output/CACHE/(key+'.npy'), mmap_mode='r')
    for y, x in ((0, 0), (cached.shape[0]//2, cached.shape[1]//2), (cached.shape[0]-1, cached.shape[1]-1)):
        # Independent direct block mean, covering top/bottom, center and alpha.
        source = np.array([image.pixels[(sy*width+x*factor)*4:(sy*width+min(width, (x+1)*factor))*4]
                           for sy in range(y*factor, min(height, (y+1)*factor))], np.float64).reshape(-1, 4)
        expected = (np.clip(source.mean(0), 0, 1)*255).astype(np.uint8)
        difference = np.abs(cached[y, x].astype(int)-expected.astype(int))
        maximum = max(maximum, int(difference.max()))
        assert difference.max() <= 1, (key, x, y, cached[y, x], expected)
    cached._mmap.close()
summary = dict(status='passed', asset=str(args.asset), blender=bpy.app.version_string,
               seconds=elapsed, original_rgba_bytes=sum(w*h*4 for w, h in metadata['original_sizes'].values()),
               preview_rgba_bytes=sum(w*h*4 for w, h in metadata['images'].values()),
               unchanged_blender_image_sizes=True, sample_max_uint8_error=maximum,
               downsampled=metadata['downsampled'], warnings=metadata['warnings'])
(args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False), flush=True)
