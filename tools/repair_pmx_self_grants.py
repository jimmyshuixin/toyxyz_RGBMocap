"""Create an explicit local PMX copy with undefined self-grants removed.

Only flags and the corresponding append-transform payload bytes change. This
does not reserialize the model: unknown extensions, additional UVs, comments,
weights, morphs and all remaining bytes are preserved. The original is read-only.
VMD for this copy must be played on this copy, not the invalid original.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mocap.retarget.pmx_validation import self_grants, validate_bones


def reader():
    spec = importlib.util.spec_from_file_location('local_pmx_reader', ROOT / 'third_party/character_import/mmd_tools/core/pmx/__init__.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repair(source, output):
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory: ' + str(output))
    raw = source.read_bytes()
    original_sha = hashlib.sha256(raw).hexdigest()
    pmx = reader()
    offsets = []
    original_load = pmx.Bone.load
    def load_bone(bone, stream):
        offsets.append(stream._FileReadStream__fin.tell())
        return original_load(bone, stream)
    with patch.object(pmx.Bone, 'load', load_bone):
        model = pmx.load(str(source))
    invalid = self_grants(model.bones)
    if not invalid:
        raise ValueError('No active self-referencing append transforms were found; no copy was written.')
    external = [texture.path for texture in model.textures
                if not Path(texture.path).resolve().is_relative_to(source.parent)]
    if external:
        raise ValueError('This byte-preserving repair requires textures inside the model folder. '
                         'External references would change meaning after relocation; no copy was written: '
                         + ', '.join(external))
    changes, edits = [], []
    index_bytes = model.header.bone_index_size
    for index, bone in invalid:
        position = offsets[index]
        # Two length-prefixed names, position, parent index, transform order.
        for _ in range(2):
            length = struct.unpack_from('<i', raw, position)[0]
            if length < 0:
                raise ValueError('Invalid PMX name length')
            position += 4 + length
        flags_at = position + 12 + index_bytes + 4
        flags = struct.unpack_from('<H', raw, flags_at)[0]
        grant_at = flags_at + 2 + (index_bytes if flags & 1 else 12)
        target = int.from_bytes(raw[grant_at:grant_at+index_bytes], 'little', signed=True)
        rate = struct.unpack_from('<f', raw, grant_at+index_bytes)[0]
        if not flags & 0x300 or target != index or rate != bone.additionalTransform[1]:
            raise ValueError('Pinned PMX parser layout does not match the source bytes')
        edits.extend([(flags_at, flags_at+2, struct.pack('<H', flags & ~0x300)),
                      (grant_at, grant_at+index_bytes+4, b'')])
        changes.append(dict(bone=bone.name, index=index, old_flags=flags,
                            new_flags=flags & ~0x300, flags_offset=flags_at,
                            removed_payload_offset=grant_at, removed_payload_bytes=index_bytes+4))
    repaired = bytearray(raw)
    for begin, end, value in sorted(edits, reverse=True):
        repaired[begin:end] = value
    output.mkdir(parents=True, exist_ok=False)
    target = output / (source.stem + '_self_grants_fixed.pmx')
    target.write_bytes(repaired)
    fixed = pmx.load(str(target))
    validate_bones(fixed.bones)
    # Independent parse of the repaired stream must preserve every bone property
    # except the three explicitly changed append-transform fields.
    assert len(fixed.bones) == len(model.bones)
    affected = {index for index, _ in invalid}
    for index, (before, after) in enumerate(zip(model.bones, fixed.bones)):
        fields = ('name', 'name_e', 'location', 'parent', 'transform_order', 'displayConnection',
                  'isRotatable', 'isMovable', 'visible', 'isControllable', 'isIK', 'axis',
                  'transAfterPhis', 'externalTransKey', 'target', 'loopCount', 'rotationConstraint')
        assert all(getattr(before, field) == getattr(after, field) for field in fields)
        if index not in affected:
            assert before.additionalTransform == after.additionalTransform
        else:
            assert after.additionalTransform is None
    copied, missing, external = [], [], []
    for texture in model.textures:
        path = Path(texture.path).resolve()
        if not path.is_file():
            missing.append(str(path))
        elif not path.is_relative_to(source.parent):
            external.append(str(path))
        else:
            destination = output / path.relative_to(source.parent)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                copied.append(str(destination.relative_to(output)))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_sha
    receipt = dict(source=str(source), source_sha256=original_sha, repaired=str(target),
                   repaired_sha256=hashlib.sha256(repaired).hexdigest(), changes=changes,
                   copied_textures=copied, missing_textures=missing, external_texture_references=external,
                   original_unchanged=True, all_other_bytes_preserved=True,
                   usage='Import this repaired PMX and play its exported VMD on this same repaired copy.')
    (output / 'repair-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(repair(args.input, args.output_directory), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
