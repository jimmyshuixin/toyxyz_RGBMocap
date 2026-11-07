"""Pack available character images without rejecting missing optional MMD maps."""
from pathlib import Path

import bpy

from .file_paths import io_path


def pack_images():
    """Return explicit diagnostics; leave missing references and source files intact.

    MMD shared toon files are external to many distributed models. Blender's
    pack_all raises after packing the valid images if any such reference is
    missing. Pack each usable image instead, without inventing replacement art.
    """
    warnings = []
    for image in bpy.data.images:
        if image.source != 'FILE' or image.packed_file:
            continue
        path = bpy.path.abspath(image.filepath, library=image.library)
        if not Path(io_path(path)).is_file():
            warnings.append(dict(code='missing_texture', image=image.name, path=path))
            continue
        if not image.size[0] or not image.size[1]:
            warnings.append(dict(code='unreadable_texture', image=image.name, path=path))
            continue
        try:
            image.pack()
        except RuntimeError as error:
            warnings.append(dict(code='texture_pack_failed', image=image.name,
                                 path=path, message=str(error)))
    return warnings
