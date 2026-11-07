"""Read back real GL pixels for diffuse tint and alpha-masked materials."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runtime_gui'))
from PySide6.QtGui import QGuiApplication, QOpenGLContext, QOffscreenSurface, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLFramebufferObject
from mocap import character_gl
from mocap.mesh_data import adjacency

legacy = '--legacy-shader' in sys.argv
if legacy:
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
assert context.create() and context.makeCurrent(surface), 'OpenGL 4.3 unavailable'
framebuffer = QOpenGLFramebufferObject(64, 64)
assert framebuffer.isValid() and framebuffer.bind()
vertices = np.array([[-.8, -.8, 0], [.8, -.8, 0], [0, .8, 0]], np.float32)
faces = np.array([[0, 1, 2]], np.uint32)
offsets, neighbors = adjacency(faces, len(vertices))
base = dict(vertices=vertices, faces=faces, rows=np.array([[0, 1], [1, 1], [2, 1]], np.uint32),
            weights=np.array([[0, 1], [0, 1], [0, 1]], np.float32), colors=np.ones((3, 3), np.float32),
            offsets=offsets, neighbors=neighbors)
observed = {}
cases = [('white', [1, 1, 1, 1], 255), ('red_tint', [1, 0, 0, 1], 255),
         ('hidden_material', [1, 1, 1, 0], 255), ('hidden_texture', [1, 1, 1, 1], 0)]
renderer = character_gl.CharacterRenderer()
for name, color, alpha in cases:
    pixels = np.full((2, 2, 4), 255, np.uint8)
    pixels[:, :, 3] = alpha
    textures = dict(corners=np.array([[0, 0, 0], [1, 1, 0], [2, .5, 1]], np.float32),
                    images={'image0': pixels}, batches=[dict(start=0, count=3, image='image0', color=color)])
    renderer.load(dict(base, key=name, textures=textures), 1)
    renderer.deform(np.eye(4, dtype=np.float32)[None], name)
    renderer.gl.glViewport(0, 0, 64, 64)
    renderer.gl.glClearColor(0, 0, 1, 1)
    renderer.gl.glClear(0x4000 | 0x0100)
    renderer.draw(np.array([0., 0., -3.]), np.eye(3), 64, 64)
    observed[name] = list(framebuffer.toImage().pixelColor(32, 32).getRgb())
assert observed['white'][0] > 100 and observed['white'][1] > 100, observed
if legacy:
    assert observed['red_tint'] == observed['white'] and observed['hidden_material'] == observed['white'], observed
else:
    assert observed['red_tint'][0] > 100 and observed['red_tint'][1] == 0 and observed['red_tint'][2] == 0, observed
    assert observed['hidden_material'] == [0, 0, 255, 255], observed
assert observed['hidden_texture'] == [0, 0, 255, 255], observed
renderer.close()
framebuffer.release()
del framebuffer
context.doneCurrent()
print(json.dumps(dict(status='passed', legacy_shader=legacy, center_rgba=observed)))
