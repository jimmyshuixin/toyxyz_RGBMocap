"""Compare old/current texture extraction on an explicitly supplied PMX.

Run inside Blender with --pmx, --baseline (old blender_textures.py), --output.
No model is modified or rendered; the generated caches remain local.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import addon_utils
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'third_party/character_import'))
from mocap.retarget import blender_textures
from mocap.retarget.blender_materials import base_surface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pmx', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--materials-only', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable('mmd_tools', default_set=False, persistent=False)
    bpy.ops.mmd_tools.import_model(filepath=str(args.pmx), types={'MESH', 'ARMATURE'}, scale=.08, log_level='ERROR')
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    faces, count = [], 0
    material_rows = []
    for obj in meshes:
        obj.data.calc_loop_triangles()
        faces.extend([[count+i for i in tri.vertices] for tri in obj.data.loop_triangles])
        count += len(obj.data.vertices)
        for mat in obj.data.materials:
            color, tex = base_surface(mat)
            nodes = mat.node_tree.nodes
            old_node = next((n for n in nodes if n.type == 'TEX_IMAGE' and n.image and 'shade' not in n.name.lower()), None)
            row = dict(name=mat.name, alpha=float(mat.mmd_material.alpha), color=color,
                       old_image=old_node.image.name if old_node else None,
                       base_image=tex.name if tex else None)
            material_rows.append(row)
            assert abs(color[3] - mat.mmd_material.alpha) < 1e-7
            base_node = nodes.get('mmd_base_tex')
            assert tex is (base_node.image if base_node else None)
    if args.materials_only:
        result = dict(pmx=str(args.pmx), blender=bpy.app.version_string, materials=material_rows,
                      corrected_image_selections=sum(r['old_image'] != r['base_image'] for r in material_rows),
                      hidden_textured_materials=sum(r['alpha'] < .1 and r['base_image'] is not None for r in material_rows))
        (args.output/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k != 'materials'}, ensure_ascii=False), flush=True)
        return
    faces = np.asarray(faces, np.uint32)
    spec = importlib.util.spec_from_file_location('mocap.retarget._texture_baseline', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    timings, outputs = {}, {}
    for label, module in [('baseline', baseline), ('candidate', blender_textures),
                          ('candidate_repeat', blender_textures), ('baseline_repeat', baseline)]:
        folder = args.output / label
        folder.mkdir()
        np.savez(folder/'preview.npz', faces=faces, vertices=np.zeros((count, 3), np.float32))
        start = time.perf_counter()
        module.extract(folder)
        timings[label] = time.perf_counter()-start
        outputs[label] = folder / module.CACHE
    old_corners = np.load(outputs['baseline']/'corners.npy')
    new_corners = np.load(outputs['candidate']/'corners.npy')
    np.testing.assert_array_equal(old_corners, new_corners)
    old_meta = json.loads((outputs['baseline']/'manifest.json').read_text())
    new_meta = json.loads((outputs['candidate']/'manifest.json').read_text())
    assert old_meta['faces_sha256'] == new_meta['faces_sha256']
    assert [(r['start'], r['count']) for r in old_meta['batches']] == [(r['start'], r['count']) for r in new_meta['batches']]
    result = dict(pmx=str(args.pmx), blender=bpy.app.version_string, vertices=count,
                  triangles=len(faces), timings_seconds=timings, corners_exact=True,
                  materials=material_rows,
                  corrected_image_selections=sum(r['old_image'] != r['base_image'] for r in material_rows),
                  hidden_textured_materials=sum(r['alpha'] < .1 and r['base_image'] is not None for r in material_rows))
    (args.output/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'materials'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
