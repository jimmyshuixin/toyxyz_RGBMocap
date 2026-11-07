"""Render a caller-selected licensed character through the actual viewport."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runtime_gui'))
from PySide6.QtGui import QGuiApplication, QOpenGLContext, QOffscreenSurface, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat
from mocap import character_gl, character_textures

parser = argparse.ArgumentParser()
parser.add_argument('--asset', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--legacy-shader', action='store_true')
args = parser.parse_args()
if args.legacy_shader:
    character_gl.TEXTURE_FRAGMENT = character_gl.TEXTURE_FRAGMENT.replace('texture(baseTexture,uv)*baseColor', 'texture(baseTexture,uv)')
app = QGuiApplication([])
fmt = QSurfaceFormat()
fmt.setVersion(4, 3)
fmt.setProfile(QSurfaceFormat.CoreProfile)
surface = QOffscreenSurface()
surface.setFormat(fmt)
surface.create()
context = QOpenGLContext()
context.setFormat(fmt)
assert context.create() and context.makeCurrent(surface)
with np.load(args.asset/'preview.npz') as archive:
    asset = {key: archive[key] for key in archive.files}
asset['key'] = str(args.asset)
asset['textures'] = character_textures.load(args.asset, asset, lambda:False)
frames = np.load(args.asset/'preview-frames.npy', mmap_mode='r')
frame_format = QOpenGLFramebufferObjectFormat()
frame_format.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
framebuffer = QOpenGLFramebufferObject(768, 1024, frame_format)
assert framebuffer.isValid() and framebuffer.bind()
renderer = character_gl.CharacterRenderer()
renderer.load(asset, frames.shape[1])
renderer.deform(frames[0], 0)
points = renderer.read_vertices()[:, :3]
lo, hi = points.min(0), points.max(0)
center = (lo+hi)/2
distance = max((hi[1]-lo[1])*1.6, (hi[0]-lo[0])*2.1)
eye = center + np.array([0., 0., distance])
renderer.gl.glViewport(0, 0, 768, 1024)
renderer.gl.glClearColor(.86, .88, .91, 1.)
renderer.gl.glClear(0x4000 | 0x0100)
renderer.draw(eye, np.diag([-1., 1., -1.]), 768, 1024)
assert framebuffer.toImage().save(str(args.output))
renderer.close()
frames._mmap.close()
framebuffer.release()
del framebuffer
context.doneCurrent()
print(args.output)
