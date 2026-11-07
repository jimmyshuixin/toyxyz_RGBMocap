"""Standalone OpenGL readback check; run with --asset <character take folder>."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runtime_gui'))
from PySide6.QtGui import QGuiApplication,QOpenGLContext,QOffscreenSurface,QSurfaceFormat
from mocap.character_gl import CharacterRenderer

parser=argparse.ArgumentParser();parser.add_argument('--asset',required=True)
args=parser.parse_args();path=Path(args.asset)
app=QGuiApplication([])
fmt=QSurfaceFormat();fmt.setVersion(4,3);fmt.setProfile(QSurfaceFormat.CoreProfile)
surface=QOffscreenSurface();surface.setFormat(fmt);surface.create()
context=QOpenGLContext();context.setFormat(fmt)
assert context.create() and context.makeCurrent(surface),'OpenGL 4.3 context unavailable'
with np.load(path/'preview.npz') as archive:asset={k:archive[k] for k in archive.files}
asset['key']=str(path)
frames=np.load(path/'preview-frames.npy',mmap_mode='r')
renderer=CharacterRenderer();renderer.load(asset,frames.shape[1])
maximum=0.
for frame in sorted({0,len(frames)//2,len(frames)-1}):
    renderer.deform(frames[frame],frame)
    actual=renderer.read_vertices()[:,:3]
    posed=asset['vertices'].copy()
    flat=frames[frame].ravel();offset=int(asset['morph_weight_offset'])
    for v,(first,count) in enumerate(asset['morph_rows']):
        entries=asset['morph_entries'][first:first+count]
        if len(entries):posed[v]+=(entries[:,1:]*flat[offset+entries[:,0].astype(int),None]).sum(0)
    expected=np.zeros_like(posed)
    for v,(first,count) in enumerate(asset['rows']):
        for bone,weight in asset['weights'][first:first+count]:
            expected[v]+=weight*(frames[frame,int(bone)]@np.r_[posed[v],1])[:3]
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=2e-6)
    maximum=max(maximum,float(abs(actual-expected).max()))
renderer.close();frames._mmap.close();context.doneCurrent()
print(f'FACE_GPU_PASS maximum absolute vertex difference {maximum:g}')
