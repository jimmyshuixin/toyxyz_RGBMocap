"""Atomic facial-only MMD VMD output; no Blender or MediaPipe import required."""
import math
from pathlib import Path
import struct
import uuid

from mocap.retarget.face_track import (load_track, sample_track, map_shape_channels,
                                      is_native_track, MMD_NATIVE_CHANNELS)


def export_vmd(track_path, target):
    track = load_track(track_path)
    native = is_native_track(track['morph_names'])
    names = track['morph_names']
    if native:
        # Channel availability depends on names, not coefficient values. Keep
        # source coefficients native until after sampling: min(L,R), products
        # and mouth normalization do not commute with rate conversion.
        mapped = map_shape_channels(names, track['weights'][:1], MMD_NATIVE_CHANNELS)
        if not mapped:
            raise ValueError('Face track has no expressions supported by MMD VMD.')
        import numpy as np
        names = list(mapped)
    count = max(1, math.ceil((len(track['weights'])-1)/track['fps']*30-1e-8)+1)
    encoded = []
    for name in names:
        raw = name.encode('cp932')
        if len(raw)>15: raise ValueError('VMD morph name exceeds 15 bytes: '+name)
        encoded.append(raw.ljust(15,b'\0'))
    target = Path(target)
    temporary = target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.partial')
    try:
        with temporary.open('xb') as output:
            output.write(b'Vocaloid Motion Data 0002'.ljust(30,b'\0'))
            output.write((b'MediaPipe face' if native else b'reze-mipo face').ljust(20,b'\0'))
            output.write(struct.pack('<II',0,count*len(encoded)))
            for start in range(0,count,256):
                weights=sample_track(track,track['start_seconds']+start/30,min(256,count-start),30)
                if native:
                    mapped = map_shape_channels(track['morph_names'], weights, MMD_NATIVE_CHANNELS)
                    weights = np.column_stack([mapped[name] for name in names])
                for i,row in enumerate(weights):
                    for name,value in zip(encoded,row):
                        output.write(name); output.write(struct.pack('<If',start+i,float(value)))
            output.write(struct.pack('<4I',0,0,0,0))  # camera, light, shadow, IK
        temporary.replace(target)
    finally: temporary.unlink(missing_ok=True)
    return target
