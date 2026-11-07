"""Budgeted base-color images and per-corner UVs, independent of motion."""
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import bpy
import numpy as np

from .file_paths import io_path
from .blender_materials import base_surface
from .texture_budget import allocate_preview_sizes, area_reduce

CACHE = 'preview-textures-v2'


def extract(out):
    out = Path(io_path(out))
    if (out / CACHE / 'manifest.json').is_file():
        return json.loads((out / CACHE / 'manifest.json').read_text(encoding='utf8'))
    temp = out / ('.textures-' + uuid.uuid4().hex)
    temp.mkdir()
    try:
        images, originals, image_objects, batches, corners, faces = {}, {}, {}, [], [], []
        offset = 0
        for obj in bpy.context.scene.objects:
            if obj.type != 'MESH':
                continue
            mesh = obj.data
            mesh.calc_loop_triangles()
            uv = mesh.uv_layers.active
            triangles = np.empty((len(mesh.loop_triangles), 3), np.int32)
            triangle_loops = np.empty_like(triangles)
            material_ids = np.empty(len(triangles), np.int32)
            mesh.loop_triangles.foreach_get('vertices', triangles.ravel())
            mesh.loop_triangles.foreach_get('loops', triangle_loops.ravel())
            mesh.loop_triangles.foreach_get('material_index', material_ids)
            faces.append((triangles + offset).astype(np.uint32))
            loop_uv = np.zeros((len(mesh.loops), 2), np.float32)
            if uv:
                uv.data.foreach_get('uv', loop_uv.ravel())
            for mi in np.unique(material_ids):
                mat = mesh.materials[mi] if mi < len(mesh.materials) else None
                color, tex = base_surface(mat)
                key = None
                if tex and uv and tex.size[0] and tex.size[1]:
                    key = images.get(tex.name)
                    if key is None:
                        w, h = tex.size
                        key = 'image' + str(len(images))
                        images[tex.name] = key
                        originals[key] = [w, h]
                        image_objects[key] = tex
                selected = material_ids == mi
                ids = (triangles[selected] + offset).ravel()
                block = np.empty((len(ids), 3), np.float32)
                block[:, 0] = ids
                block[:, 1:] = loop_uv[triangle_loops[selected].ravel()]
                start = batches[-1]['start'] + batches[-1]['count'] if batches else 0
                corners.append(block)
                batches.append(dict(start=start, count=len(block), image=key, color=color))
            offset += len(mesh.vertices)
        faces = np.concatenate(faces) if faces else np.empty((0, 3), np.uint32)
        with np.load(out / 'preview.npz', allow_pickle=False) as asset:
            if len(asset['vertices']) != offset or not np.array_equal(asset['faces'], faces):
                raise ValueError('Stored character geometry differs from the texture source. Reimport the character.')
        np.save(temp / 'corners.npy', np.concatenate(corners) if corners else np.empty((0, 3), np.float32))
        sizes, factors = allocate_preview_sizes(originals)
        downsampled, warnings = {}, []
        for key, tex in image_objects.items():
            write_preview_image(tex, temp / (key + '.npy'), sizes[key], factors[key])
            if factors[key] > 1:
                downsampled[key] = dict(name=tex.name, original=originals[key], preview=sizes[key], factor=factors[key])
                ow, oh = originals[key]
                pw, ph = sizes[key]
                warning = f'Preview texture {tex.name}: {ow}x{oh} -> {pw}x{ph} to fit the preview memory budget; original retained in the Blender asset.'
                warnings.append(warning)
                print('CHARACTER: ' + warning, flush=True)
        metadata = dict(version=2, vertices=offset, faces_sha256=hashlib.sha256(faces.tobytes()).hexdigest(),
                        batches=batches, images=sizes, original_sizes=originals,
                        downsampled=downsampled, warnings=warnings)
        (temp / 'manifest.json').write_text(json.dumps(metadata), encoding='utf8')
        temp.rename(out / CACHE)
        return metadata
    finally:
        # Only this invocation's temporary directory, never a user asset.
        if temp.exists():
            shutil.rmtree(temp)


def write_preview_image(image, target, size, factor):
    """Write RGBA8 in blocks; large float buffers use a temporary mapped file.

    Read only the pixel array. Never scale or copy Blender's image datablock,
    which remains the original packed texture used by character exports.
    """
    width, height = image.size
    out_width, out_height = size
    output = np.lib.format.open_memmap(target, mode='w+', dtype=np.uint8,
                                      shape=(out_height, out_width, 4))
    source = None
    scratch = target.with_suffix('.float32.tmp')
    try:
        if width * height * 16 <= 64 * 1024**2:
            source = np.empty((height, width, 4), np.float32)
        else:
            # Blender RNA slices may convert the entire ImBuf for each slice.
            # One foreach_get avoids that repeated work. File-backed storage
            # avoids a full private/commit allocation and is reclaimed by the
            # OS; it is not a hard bound on the process's resident working set.
            source = np.memmap(scratch, mode='w+', dtype=np.float32,
                               shape=(height, width, 4))
        image.pixels.foreach_get(source.reshape(-1))
        rows = max(1, (4 * 1024**2)//(width * 16 * factor))
        for y in range(0, out_height, rows):
            pixels = source[y*factor:min(height, (y+rows)*factor)]
            reduced = area_reduce(pixels, factor)
            np.clip(reduced, 0, 1, out=reduced)
            reduced *= 255
            output[y:y+rows] = reduced.astype(np.uint8)
        output.flush()
    finally:
        output._mmap.close()
        if isinstance(source, np.memmap):
            source._mmap.close()
        scratch.unlink(missing_ok=True)
