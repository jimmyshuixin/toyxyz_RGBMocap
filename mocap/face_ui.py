"""Video facial capture in an isolated process, followed by transactional baking."""
from __future__ import annotations

import codecs
import json
from pathlib import Path
import uuid

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
    QPushButton, QTextEdit, QVBoxLayout)

from .paths import ROOT
from .storage import atomic_json
from .viewer import VideoView


class FaceRegionView(VideoView):
    """Choose a fixed detector region in normalized original-image coordinates."""
    def __init__(self, crop):
        super().__init__()
        self.crop = list(crop)
        self.drag_start = None

    def point(self, position):
        rect = self.image_rect()
        return (max(0., min(1., (position.x()-rect.x())/rect.width())),
                max(0., min(1., (position.y()-rect.y())/rect.height())))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.frame is not None:
            self.drag_start = self.point(event.position()); event.accept()
        else: super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_start is not None:
            x, y = self.point(event.position()); x0, y0 = self.drag_start
            self.crop = [min(x,x0), min(y,y0), max(x,x0), max(y,y0)]
            self.update(); event.accept()
        else: super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drag_start is not None:
            self.drag_start = None
            if self.crop[2]-self.crop[0]<.005 or self.crop[3]-self.crop[1]<.005:
                self.crop = [0.,0.,1.,1.]
            self.update(); event.accept()
        else: super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.frame is not None:
            painter = QPainter(self); painter.setPen(QPen(QColor('#ffce54'),2))
            rect = self.image_rect(); left,top,right,bottom = self.crop
            painter.drawRect(QRectF(rect.x()+left*rect.width(), rect.y()+top*rect.height(),
                                    (right-left)*rect.width(), (bottom-top)*rect.height()))
            painter.end()


class FaceRegionDialog(QDialog):
    def __init__(self, parent, video, seconds, crop):
        super().__init__(parent)
        self.setWindowTitle('Select Face Region'); self.resize(800,580)
        layout = QVBoxLayout(self)
        label = QLabel('Drag a box around the head. Include its movement throughout the capture interval.\n'
                       'Wheel: zoom · middle drag: pan. The source video is unchanged.')
        label.setWordWrap(True); layout.addWidget(label)
        self.view = FaceRegionView(crop); layout.addWidget(self.view,1)
        self.message = QLabel('Loading video frame...'); layout.addWidget(self.message)
        actions = QHBoxLayout(); reset = QPushButton('Full Frame')
        reset.clicked.connect(self.reset_region); actions.addWidget(reset); actions.addStretch()
        use = QPushButton('Use Region'); use.setEnabled(False); use.clicked.connect(self.accept)
        close = QPushButton('Cancel'); close.clicked.connect(self.reject)
        actions.addWidget(use); actions.addWidget(close); layout.addLayout(actions)
        self.view.frame_presented.connect(lambda _: (use.setEnabled(True), self.message.setText('Select the face region.')))
        self.view.decode_error.connect(lambda error: (use.setEnabled(False), self.message.setText(error)))
        self.view.metadata_ready.connect(lambda info:self.view.seek(max(0,min(info['frames']-1,round(seconds*info['fps'])))))
        self.view.open(Path(video))

    def reset_region(self):
        self.view.crop=[0.,0.,1.,1.]; self.view.reset_view()

    def done(self, result):
        self.view.shutdown()
        super().done(result)


def capture_source(window):
    """Return media-local range and original timeline origin, including saved trims."""
    path = Path(window.video_path) if window.video_path else None
    first, end = window.processing_range() if window.info else (0., 0.)
    origin = 0.
    if window.workspace_root:
        from .project_workspace import read
        origin = float(read(window.workspace_root).get('time_offset', 0.))
    elif window.project:
        from .project_storage import manifest
        origin = float(manifest(window.project).get('storage', {}).get('time_offset', 0.))
    if (path is None or not path.is_file()) and window.project:
        candidate = Path(window.project) / 'input.mp4'
        if candidate.is_file():
            from .project_storage import manifest
            path = candidate
            origin = float(manifest(window.project)['start'])
    return path, max(0., first-origin), max(0., end-origin), origin


class FaceCaptureDialog(QDialog):
    def __init__(self, window, selected=None):
        super().__init__(window)
        self.window = window
        self.process = None
        self.result = None
        self.apply_result = None
        self.closing = False
        self.cancelled = False
        self.crop = [0.,0.,1.,1.]
        self.setWindowTitle('Video Face Capture · MediaPipe')
        self.resize(650, 510)
        layout = QVBoxLayout(self)
        intro = QLabel('Capture one face from a video and merge expressions with a character’s body motion.\n'
                       'Best with a clear, front-facing face. Select a face region for small faces or multiple people.')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        path, first, end, origin = capture_source(window)
        self.video = QLineEdit(str(path) if path else '')
        browse = QPushButton('Browse...')
        browse.clicked.connect(self.browse)
        row = QHBoxLayout(); row.addWidget(self.video); row.addWidget(browse)
        form.addRow('Video', row)
        self.crop_label = QLabel('Full frame')
        region = QPushButton('Select Face Region...'); region.clicked.connect(self.select_region)
        crop_row = QHBoxLayout(); crop_row.addWidget(self.crop_label); crop_row.addWidget(region)
        form.addRow('Detector region', crop_row)
        self.start = self.seconds(first)
        self.end = self.seconds(end)
        self.origin = self.seconds(origin)
        form.addRow('Start in video (s)', self.start)
        form.addRow('End in video (s, 0 = end)', self.end)
        form.addRow('Video begins on timeline (s)', self.origin)
        self.origin.setToolTip('Original timeline time of the first frame in this video. Usually 0; saved trimmed clips retain their original offset.')
        self.target = QComboBox()
        self.target.addItem('Save facial motion only', None)
        for character in window.characters:
            self.target.addItem(character['name'] + ' · ' + character['kind'].upper(), character['id'])
        preferred = selected or window.selected_character()
        if preferred:
            self.target.setCurrentIndex(max(0, self.target.findData(preferred['id'])))
        form.addRow('Apply to character', self.target)
        self.smooth = QCheckBox('Smooth expressions'); self.smooth.setChecked(True)
        self.smooth.setToolTip('Reduce expression jitter while retaining fast eye movements.')
        self.predict = QCheckBox('Estimate brief detection gaps'); self.predict.setChecked(True)
        self.predict.setToolTip('Fill short gaps from nearby observations and briefly predict recent motion. Longer gaps fade to neutral. Estimates cannot recover unseen expressions.')
        self.mirror = QCheckBox('Swap left/right expressions'); self.mirror.setChecked(False)
        self.mirror.setToolTip('Off keeps MediaPipe’s anatomical left/right. Turn on only for mirrored source footage.')
        form.addRow(self.smooth); form.addRow(self.predict); form.addRow(self.mirror)
        layout.addLayout(form)
        self.controls = [self.video, browse, region, self.start, self.end, self.origin, self.target, self.smooth, self.predict, self.mirror]
        self.video.textChanged.connect(self.reset_crop)
        self.runtime_status = QLabel(); self.runtime_status.setWordWrap(True)
        layout.addWidget(self.runtime_status)
        self.status = QLabel('Facial motion is saved locally. PMX models need facial morphs; older imports may need reimporting.')
        self.status.setTextFormat(Qt.PlainText); self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar(); layout.addWidget(self.progress)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.document().setMaximumBlockCount(150)
        self.log.setMaximumHeight(110); layout.addWidget(self.log)
        actions = QHBoxLayout()
        self.setup = QPushButton('Set Up Face Capture'); self.setup.clicked.connect(self.run_setup)
        self.capture = QPushButton('Capture Face'); self.capture.clicked.connect(self.run_capture)
        self.apply = QPushButton('Apply to Character'); self.apply.clicked.connect(self.apply_track)
        self.export = QPushButton('Export Face VMD...'); self.export.clicked.connect(self.export_track)
        self.cancel = QPushButton('Close'); self.cancel.clicked.connect(self.reject)
        for button in (self.setup, self.capture, self.apply, self.export, self.cancel): actions.addWidget(button)
        layout.addLayout(actions)
        self.target.currentIndexChanged.connect(self.refresh)
        for control in (self.start, self.end, self.origin):
            control.valueChanged.connect(self.invalidate_result)
        self.smooth.toggled.connect(self.invalidate_result)
        self.predict.toggled.connect(self.invalidate_result)
        self.mirror.toggled.connect(self.invalidate_result)
        self.refresh()

    @staticmethod
    def seconds(value):
        box = QDoubleSpinBox(); box.setRange(0, 86400); box.setDecimals(3)
        box.setValue(value); return box

    def browse(self):
        name, _ = QFileDialog.getOpenFileName(self, 'Face Video', self.video.text(), 'Video (*.mp4 *.mov *.mkv *.avi *.webm)')
        if name:
            self.video.setText(name); self.start.setValue(0); self.end.setValue(0); self.origin.setValue(0)

    def reset_crop(self, *_):
        self.crop = [0.,0.,1.,1.]; self.crop_label.setText('Full frame')
        self.invalidate_result()

    def invalidate_result(self, *_):
        if self.result is not None:
            self.result = None
            self.status.setText('Capture settings changed. Capture again before applying or exporting. The previous saved track was kept.')
            self.refresh()

    def select_region(self):
        if not Path(self.video.text()).is_file():
            self.status.setText('Choose an existing video first.'); return
        dialog=FaceRegionDialog(self,self.video.text(),self.start.value(),self.crop)
        if dialog.exec():
            self.invalidate_result()
            self.crop=list(dialog.view.crop)
            self.crop_label.setText('Full frame' if self.crop==[0.,0.,1.,1.] else
                                    'Region: '+', '.join(f'{x*100:.1f}%' for x in self.crop))
        dialog.deleteLater()

    def refresh(self, *_):
        from .face import runtime
        state = runtime.status()
        running = self.process is not None
        self.runtime_status.setText(state['message'])
        self.setup.setEnabled(not running)
        self.capture.setEnabled(not running and state['ready'])
        self.apply.setEnabled(not running and self.result is not None and self.target.currentData() is not None)
        self.export.setEnabled(not running and self.result is not None)
        for control in self.controls: control.setEnabled(not running)
        self.cancel.setText('Cancel' if running else 'Close')

    def run_setup(self):
        from .face.runtime import setup_command
        try: self.start_process(setup_command(), 'setup')
        except Exception as exc: self.status.setText(str(exc))

    def run_capture(self):
        from .face.runtime import command, model_path
        try:
            video = Path(self.video.text()).expanduser().resolve()
            if not video.is_file(): raise ValueError('Choose an existing video.')
            if self.end.value() and self.end.value() <= self.start.value():
                raise ValueError('End must be later than Start.')
            self.folder = 'face-' + uuid.uuid4().hex
            output = self.window.character_path(self.folder)
            request = self.window.character_path('face-request-' + uuid.uuid4().hex + '.json')
            job = dict(video=str(video), output_dir=str(output), model_path=str(model_path()),
                       start_seconds=self.start.value(), end_seconds=self.end.value() or None,
                       timeline_offset_seconds=self.origin.value(), fps=30.,
                       crop=self.crop,
                       smoothing=self.smooth.isChecked(), prediction=self.predict.isChecked(),
                       mirror=not self.mirror.isChecked())
            atomic_json(request, job)
            self.result = None
            self.start_process(command(request), 'capture')
        except Exception as exc: self.status.setText(str(exc))

    def start_process(self, command, mode):
        self.mode = mode; self.cancelled = False; self.closing = False
        self.pending = ''; self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self.tail = ''; self.worker_error = None
        process = QProcess(self); self.process = process
        process.setWorkingDirectory(str(ROOT)); process.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONUTF8', '1'); env.insert('PYTHONIOENCODING', 'utf-8')
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self.read_output)
        process.finished.connect(self.process_finished)
        process.started.connect(lambda: self.terminate_process() if self.cancelled else None)
        process.errorOccurred.connect(lambda error: self.process_finished(-1) if error == QProcess.FailedToStart else None)
        self.progress.setRange(0, 0); self.status.setText('Preparing face runtime...' if mode == 'setup' else 'Capturing face...')
        self.refresh(); process.start(command[0], command[1:])

    def read_output(self, final=False):
        if self.process is None: return
        text = self.decoder.decode(bytes(self.process.readAllStandardOutput()), final=final)
        self.tail = (self.tail + text)[-16000:]
        self.pending = (self.pending + text)[-32768:]
        lines = self.pending.split('\n'); self.pending = lines.pop()
        if final and self.pending: lines.append(self.pending); self.pending = ''
        for line in lines:
            try: event = json.loads(line)
            except (ValueError, TypeError):
                if line.strip(): self.log.append(line[:1500])
                continue
            if not isinstance(event, dict): continue
            if event.get('type') == 'progress':
                self.progress.setRange(0, 1000)
                self.progress.setValue(round(float(event.get('progress', 0))*1000))
                self.status.setText(event.get('message', 'Capturing face...'))
            elif event.get('type') == 'error':
                self.worker_error = event.get('message', 'Face capture failed.')
                self.log.append(self.worker_error)

    def process_finished(self, code, *_):
        if self.process is None: return
        self.read_output(final=True)
        process = self.process; self.process = None; process.deleteLater()
        self.progress.setRange(0, 100); self.progress.setValue(0)
        if self.cancelled:
            self.status.setText('Cancelled. Existing character motion was kept.')
        elif code != 0:
            self.status.setText(self.worker_error or 'Face processing failed. See details below.')
            self.log.append(self.tail[-2500:])
        elif self.mode == 'setup':
            self.status.setText('Face runtime setup finished.'); self.progress.setValue(100)
        else:
            try:
                track = self.window.character_path(self.folder + '/face.npz')
                from .retarget.face_track import load_track
                data = load_track(track)
                self.result = dict(path=track, relative=self.folder + '/face.npz')
                self.progress.setValue(100)
                detected = int(data['detected'].sum())
                predicted = int(data['predicted'].sum()) if 'predicted' in data else 0
                interpolated = int(data['interpolated'].sum()) if 'interpolated' in data else 0
                self.status.setText(f'Face capture saved · detected in {detected}/{len(data["weights"])} frames.\n'
                                    f'{interpolated} interpolated · {predicted} predicted/fading. Apply or export a facial VMD.\n' + str(track))
            except Exception as exc:
                self.status.setText('Cannot read captured motion: ' + str(exc))
        self.refresh()
        if self.closing: super().reject()

    def apply_track(self):
        if self.result and self.target.currentData():
            self.apply_result = dict(self.result, character_id=self.target.currentData())
            self.accept()

    def export_track(self):
        name, _ = QFileDialog.getSaveFileName(self, 'Export Face VMD', 'face.vmd', 'VMD (*.vmd)')
        if not name: return
        try:
            from .face.vmd import export_vmd
            path = Path(name)
            if path.suffix.lower() != '.vmd': path = path.with_name(path.name + '.vmd')
            export_vmd(self.result['path'], path)
            self.status.setText('Exported facial motion (starts at VMD frame 0): ' + str(path))
        except Exception as exc: self.status.setText(str(exc))

    def reject(self):
        if self.process is None:
            super().reject(); return
        self.cancelled = True; self.closing = True
        self.status.setText('Cancelling face process...'); self.cancel.setEnabled(False)
        self.terminate_process()

    def terminate_process(self):
        process = self.process
        if process is None: return
        if process.state() == QProcess.NotRunning:
            self.process_finished(process.exitCode()); return
        if not process.processId(): return  # started signal retries after Starting
        killer = QProcess(self); self.killer = killer
        def finished(code):
            if code and self.process is process and process.state()!=QProcess.NotRunning:
                self.log.append('Process-tree cancellation failed; stopping the face process directly.')
                process.kill()
            killer.deleteLater()
        killer.finished.connect(finished)
        killer.errorOccurred.connect(lambda error: finished(-1) if error==QProcess.FailedToStart else None)
        killer.start('taskkill', ['/PID', str(process.processId()), '/T', '/F'])

    def closeEvent(self, event):
        if self.process is not None:
            event.ignore(); self.reject()
        else: event.accept()


def capture_face(window, selected=None):
    if window.busy or window.contact_saving or getattr(window, '_pear_active', False): return
    window.stop_playback(); window.sync_characters()
    dialog = FaceCaptureDialog(window, selected)
    window.busy = True
    window.update_readiness()
    try: dialog.exec()
    finally:
        window.busy = False; window.update_readiness()
    result = dialog.apply_result
    dialog.deleteLater()
    if result:
        character = next(c for c in window.characters if c['id'] == result['character_id'])
        folder = character['id'] + '/animation-' + uuid.uuid4().hex
        job = dict(mode='face', id=character['id'], output=str(window.character_path(folder)),
                   asset=str(window.character_path(character['asset'])),
                   source_animation=str(window.character_path(character['animation'])) if character.get('animation') else None,
                   animation=folder, takes=character.get('takes', []),
                   face_motion=str(result['path']), face_motion_relative=result['relative'],
                   source_name=character.get('source_name', 'Face capture'), result=character.get('result', 'face'))
        window.start_character_job(job)
