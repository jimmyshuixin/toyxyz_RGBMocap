"""Output-world regression tests using real BVH/FBX files and bundle staging."""
from contextlib import ExitStack
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap import blend_export, export_copy, fbx_export
from mocap.biped_bvh import bone_names
from mocap.motion_heading import TRANSFORM


def fbx_elements(path):
    """Read binary FBX properties independently of the application's writer."""
    source = io.BytesIO(path.read_bytes())
    source.seek(27)

    def value():
        code = source.read(1)
        if code in (b'I', b'L', b'D', b'C'):
            fmt = {b'I': '<i', b'L': '<q', b'D': '<d', b'C': '<?'}[code]
            return struct.unpack(fmt, source.read(struct.calcsize(fmt)))[0]
        if code == b'S':
            size, = struct.unpack('<I', source.read(4))
            return source.read(size).decode('utf-8')
        if code in (b'i', b'l', b'f', b'd'):
            count, compressed, size = struct.unpack('<III', source.read(12))
            raw = source.read(size)
            if compressed:
                raw = zlib.decompress(raw)
            return np.frombuffer(raw, {b'i': '<i4', b'l': '<i8', b'f': '<f4', b'd': '<f8'}[code], count)
        raise AssertionError(code)

    def elements(end=None):
        result = []
        while end is None or source.tell() < end:
            stop, count, _, size = struct.unpack('<IIIB', source.read(13))
            if not stop:
                break
            name = source.read(size).decode('ascii')
            properties = [value() for _ in range(count)]
            children = elements(stop) if source.tell() < stop else []
            result.append((name, properties, children))
        return result

    return elements()


def root_fbx_channels(path):
    objects = next(row[2] for row in fbx_elements(path) if row[0] == 'Objects')
    curves = [row for row in objects if row[0] == 'AnimationCurve']
    channels = [next(child[1][0] for child in row[2] if child[0] == 'KeyValueFloat') for row in curves[:6]]
    return np.column_stack(channels)


class HeadingExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.project = self.folder / 'project'
        self.project.mkdir()
        self.names = list(bone_names())
        self.parents = np.r_[-1, np.zeros(51, dtype=int)]
        self.rest = np.zeros((52, 3))
        self.rest[1:, 0] = np.linspace(-.8, .8, 51)
        self.rest[1:, 1] = np.linspace(.1, .9, 51)
        self.rotations = Rotation.from_euler('y', [180., 205., 250.], degrees=True).as_matrix()
        self.positions = np.array([[2., 1., 3.], [2.2, 1.1, 2.6], [2.5, .9, 2.1]])
        self.child_rotation = Rotation.from_euler('x', [12., -5., 17.], degrees=True).as_matrix()
        self.frames = np.tile(np.eye(4), (3, 52, 1, 1))
        self.frames[:, :, :3, :3] = self.rotations[:, None] @ self.child_rotation[:, None]
        self.frames[:, 0, :3, :3] = self.rotations
        self.frames[:, :, :3, 3] = self.positions[:, None] + np.einsum('tij,nj->tni', self.rotations, self.rest)
        np.save(self.project / 'frames.npy', self.frames)
        identity = np.tile(np.eye(3), (52, 1, 1))
        np.savez(self.project / 'asset.npz', parents=self.parents, fbx_rest=self.rest,
                 fbx_rest_frames=identity, fbx_display_frames=identity, fbx_tip_local=np.zeros((52, 3)))
        self.info = dict(names=self.names, frames=3, fps=30., asset_file='asset.npz', frames_file='frames.npy')
        (self.project / 'initial_mixamo.json').write_text(json.dumps(self.info), encoding='utf-8')
        (self.project / 'project.json').write_text(json.dumps(dict(frames=3, fps=30., start=0.)), encoding='utf-8')
        self.source = self.project / 'source.bvh'
        lines = ['HIERARCHY', 'ROOT Hips', '{', ' OFFSET 0 0 0',
                 ' CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation']
        for name, point in zip(self.names[1:], self.rest[1:]):
            lines += [' JOINT ' + name, ' {', '  OFFSET ' + ' '.join(map(str, point)),
                      '  CHANNELS 3 Xrotation Yrotation Zrotation', ' }']
        lines += ['}', 'MOTION', 'Frames: 3', 'Frame Time: 0.0333333333']
        root_angles = Rotation.from_matrix(self.rotations).as_euler('ZXY', degrees=True)
        child_angles = Rotation.from_matrix(self.child_rotation).as_euler('XYZ', degrees=True)
        for position, root, child in zip(self.positions, root_angles, child_angles):
            lines += [' '.join(map(str, np.r_[position, root, np.tile(child, 51)]))]
        self.source.write_text('\n'.join(lines) + '\n', encoding='ascii')
        self.original = {p: p.read_bytes() for p in self.project.iterdir()}
        self.placement = np.eye(4)
        self.placement[:3, :3] = Rotation.from_euler('y', 37., degrees=True).as_matrix()
        self.placement[:3, 3] = [3., .2, -4.]
        self.scene = dict(root=str(self.folder), transform=self.placement.tolist())

    def row(self, target, **options):
        return dict(project=str(self.project), kind='initial', source=str(self.source), target=str(target), **options)

    def request(self, row):
        request = self.folder / 'request.json'
        request.write_text(json.dumps([row]), encoding='utf-8')
        return request

    def assert_originals(self):
        for path, original in self.original.items():
            self.assertEqual(path.read_bytes(), original, str(path))

    def expected(self, placed):
        transform = TRANSFORM @ (self.placement if placed else np.eye(4))
        return transform, self.positions @ transform[:3, :3].T + transform[:3, 3]

    def test_fbx_actual_channels_turn_once_and_keep_reference_pose(self):
        for placed in (False, True):
            for prepend in (False, True):
                with self.subTest(placed=placed, prepend=prepend):
                    target = self.folder / f'{placed}-{prepend}.fbx'
                    row = self.row(target, prepend_tpose=prepend, **({'scene': self.scene} if placed else {}))
                    with patch('mocap.fbx_export.selected_export', return_value=self.source), \
                         patch('mocap.scene_export.context', return_value=self.scene):
                        self.assertEqual(fbx_export.run(self.request(row)), 1)
                    result = root_fbx_channels(target)
                    transform, positions = self.expected(placed)
                    np.testing.assert_allclose(result[int(prepend):, :3], positions, atol=4e-7)
                    actual = Rotation.from_euler('xyz', result[int(prepend):, 3:], degrees=True).as_matrix()
                    np.testing.assert_allclose(actual, transform[:3, :3] @ self.rotations, atol=4e-7)
                    if prepend:
                        np.testing.assert_allclose(result[0, :3], positions[0], atol=4e-7)
                        pose = Rotation.from_euler('xyz', result[0, 3:], degrees=True).as_matrix()
                        np.testing.assert_allclose(pose, self.placement[:3, :3] if placed else np.eye(3), atol=4e-7)
                    self.assert_originals()

    def test_bvh_both_profiles_preserve_local_motion_and_reference_pose(self):
        for profile in ('generic', 'biped'):
            for placed in (False, True):
                with self.subTest(profile=profile, placed=placed):
                    target = self.folder / f'{profile}-{placed}.bvh'
                    row = self.row(target, bvh_profile=profile, prepend_tpose=True,
                                   **({'scene': self.scene} if placed else {}))
                    with patch('mocap.export_copy.selected_export', return_value=self.source), \
                         patch('mocap.scene_export.context', return_value=self.scene):
                        self.assertEqual(export_copy.run(self.request(row)), 1)
                    rows = target.read_text(encoding='ascii').split('Frame Time:', 1)[1].splitlines()[1:]
                    result = np.asarray([list(map(float, row.split())) for row in rows])
                    transform, positions = self.expected(placed)
                    np.testing.assert_allclose(result[1:, :3], positions, atol=1e-8)
                    np.testing.assert_allclose(result[0, :3], positions[0], atol=1e-8)
                    actual = Rotation.from_euler('ZXY', result[1:, 3:6], degrees=True).as_matrix()
                    np.testing.assert_allclose(actual, transform[:3, :3] @ self.rotations, atol=1e-8)
                    pose = Rotation.from_euler('ZXY', result[0, 3:6], degrees=True).as_matrix()
                    np.testing.assert_allclose(pose, self.placement[:3, :3] if placed else np.eye(3), atol=1e-8)
                    child = Rotation.from_euler('ZXY' if profile == 'biped' else 'XYZ', result[1:, 6:9], degrees=True).as_matrix()
                    np.testing.assert_allclose(child, self.child_rotation, atol=1e-8)
                    self.assert_originals()

    def test_blender_bundle_transports_camera_after_motion_scale(self):
        video = self.project / 'input.mp4'
        video.write_bytes(b'video fixture; decoder is stubbed')
        native = np.tile(np.eye(4), (3, 1, 1))
        native[:, :3, :3] = Rotation.from_euler('xyz', [[8., 12., 1.], [7., 14., 2.], [5., 17., 3.]], degrees=True).as_matrix()
        native[:, :3, 3] = [0.3, -1.5, 5.]
        displacement = np.array([[0., 0., 0.], [.1, 0., -.2], [.25, .05, -.1]])
        camera = dict(K=np.array([[800., 0., 320.], [0., 810., 240.], [0., 0., 1.]]),
                      size=(640, 480), matrices=native.copy(), start=0., fps=30., displacement=displacement)
        cameras = dict(cameras=[camera], camera_sources={}, model='exmr', note='Synthetic camera')
        captured = {}

        def build(work, script, emit, stage):
            job = json.loads((work / 'blend-job.json').read_text(encoding='utf-8'))
            captured['camera'] = np.load(work / 'camera.npy')
            captured['root'] = root_fbx_channels(work / 'animation.fbx')
            (work / job['target']).write_bytes(b'Blender builder stub')

        target = self.folder / 'bundle' / 'animation.blend'
        row = self.row(target, scene=self.scene, animate_camera=True, camera_translation_scale=2.)
        with ExitStack() as stack:
            for name, value in [('mocap.motion_tools.selected_export', self.source),
                                ('mocap.fbx_export.selected_export', self.source),
                                ('mocap.scene_export.context', self.scene),
                                ('mocap.camera_tracks.load', cameras),
                                ('mocap.video.probe', dict(frames=3, fps=30., width=640, height=480))]:
                stack.enter_context(patch(name, return_value=value))
            stack.enter_context(patch('mocap.blend_export.copy_video', side_effect=lambda a, b, *args: b.write_bytes(a.read_bytes())))
            stack.enter_context(patch('mocap.blend_export.build', side_effect=build))
            stack.enter_context(patch('mocap.blend_export.disk_check'))
            self.assertEqual(blend_export.run(self.request(row)), 1)
        moved = native.copy()
        moved[:, :3, 3] -= np.einsum('tij,tj->ti', native[:, :3, :3], displacement * 2.)
        np.testing.assert_allclose(captured['camera'], moved @ TRANSFORM, atol=1e-12)
        scene_positions = self.positions @ self.placement[:3, :3].T + self.placement[:3, 3]
        before = np.einsum('tij,tj->ti', moved[:, :3, :3], scene_positions) + moved[:, :3, 3]
        after = np.einsum('tij,tj->ti', captured['camera'][:, :3, :3], captured['root'][:, :3]) + captured['camera'][:, :3, 3]
        np.testing.assert_allclose(after, before, atol=6e-7)
        np.testing.assert_array_equal(camera['matrices'], native)
        self.assert_originals()


if __name__ == '__main__':
    unittest.main()
