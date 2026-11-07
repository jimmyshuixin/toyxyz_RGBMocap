"""Local video-to-face capture. Detector packages stay in the optional runtime.

Input video is sampled on its constant-rate frame clock (the app's prepared
working videos are CFR). No video or landmark data is sent over the network.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Callable

from .blendshapes import BLENDSHAPE_NAMES, scores_from_categories, side_permutation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@contextmanager
def _detector(model_path: Path, confidence: float, reject_multiple_faces: bool = False):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    if model_path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Face model exceeds the 64 MB task limit")
    # MediaPipe's native Windows file reader rejects some non-ASCII paths.
    # Read through Python's Unicode-safe filesystem layer instead (official
    # FaceLandmarker task is about 3.6 MiB; bounded before reading).
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=model_path.read_bytes(),
                                        delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2 if reject_multiple_faces else 1,
        min_face_detection_confidence=confidence,
        min_face_presence_confidence=confidence,
        min_tracking_confidence=confidence,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    with vision.FaceLandmarker.create_from_options(options) as detector:
        def detect(rgb, timestamp_ms):
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
            if len(result.face_landmarks) > 1:
                return {'blocked': True}
            if not result.face_landmarks:
                return None
            if len(result.face_blendshapes) != 1:
                raise RuntimeError('FaceLandmarker did not return expression scores for the detected face.')
            return {'weights': scores_from_categories(result.face_blendshapes[0])}
        yield detect


def _finite_float(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _crop_box(value):
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("crop must be [left, top, right, bottom] normalized fractions")
    left, top, right, bottom = (_finite_float(v, "crop") for v in value)
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("crop requires 0 <= left < right <= 1 and 0 <= top < bottom <= 1")
    return left, top, right, bottom


def run(job: dict, progress: Callable[[dict], None] | None = None,
        *, cancelled: Callable[[], bool] | None = None,
        crop_for_frame: Callable | None = None) -> dict:
    """Capture into a NEW output_dir and return its paths/summary.

    Required keys: video, model_path, output_dir. Optional: start_seconds=0,
    end_seconds=video duration (exclusive), fps=30, smoothing=True, prediction=True,
    mirror=True (keep anatomical sides), confidence=.5,
    timeline_offset_seconds=0 (absolute source time of video-local frame zero),
    crop=None or normalized [left, top, right, bottom]. Cropping affects only
    detection, not the model's expression coefficient scale.
    Output face.npz: 52 morph_names, processed weights and raw_weights (N,52),
    detected/predicted/interpolated/blocked masks, reliability, fps/start_seconds.
    Internal crop_for_frame(source_frame, image_shape) supplies actor-specific
    normalized boxes; None skips that frame instead of selecting another person.
    reject_multiple_faces=True rejects ambiguous multi-face actor crops.
    Ordinary detection gaps use bounded temporal estimation. Actor absence and
    ambiguous multiple faces block propagation. No artifact is published on
    cancellation, decode error, or an all-undetected clip.
    """
    import cv2
    import numpy as np

    if not isinstance(job, dict):
        raise ValueError("Face capture request must be a JSON object")
    video, model = Path(job["video"]).resolve(), Path(job["model_path"]).resolve()
    output = Path(job["output_dir"]).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video does not exist: {video}")
    if not model.is_file():
        raise FileNotFoundError(f"Face landmarker model does not exist: {model}")
    if output.exists():
        raise FileExistsError(f"Face output directory already exists; choose a new generation: {output}")
    start = _finite_float(job.get("start_seconds", 0), "start_seconds")
    timeline_offset = _finite_float(job.get("timeline_offset_seconds", 0), "timeline_offset_seconds")
    fps = _finite_float(job.get("fps", 30), "fps")
    confidence = _finite_float(job.get("confidence", .5), "confidence")
    crop = _crop_box(job.get("crop"))
    if start < 0 or timeline_offset < 0 or not 1 <= fps <= 120 or not 0 < confidence <= 1:
        raise ValueError("Require start_seconds >= 0, 1 <= fps <= 120 and 0 < confidence <= 1")
    for flag in ("smoothing", "prediction", "mirror", "reject_multiple_faces"):
        if flag in job and not isinstance(job[flag], bool):
            raise ValueError(f"{flag} must be a JSON boolean")
    smoothing, prediction = job.get('smoothing', True), job.get('prediction', True)
    sides = side_permutation(swap=not job.get('mirror', True))
    emit = progress or (lambda event: None)
    source_stat = video.stat()
    model_digest = _sha256(model)
    cap = cv2.VideoCapture(str(video))
    staging = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(source_fps) or source_fps <= 0 or source_count <= 0:
            raise ValueError("Video has no usable frame rate/frame count; prepare a CFR working video first")
        duration = source_count / source_fps
        end = duration if job.get("end_seconds") is None else _finite_float(job["end_seconds"], "end_seconds")
        if not start < end or start >= duration or end > duration + 1 / source_fps:
            raise ValueError(f"Capture interval must lie inside the video (duration {duration:.6f}s)")
        end = min(end, duration)
        count = max(1, math.ceil((end - start) * fps - 1e-8))
        # This bounds output arrays, not detection resolution or body runtimes.
        # Match retarget.face_track.MAX_FRAMES so every successful capture can
        # be loaded by the character preview/export path (30 min at 30 fps).
        if count > 54_000:
            raise ValueError("Face capture exceeds 54,000 samples; select a shorter interval (30 minutes at 30 fps)")
        raw_weights = np.zeros((count, len(BLENDSHAPE_NAMES)), dtype=np.float32)
        detected = np.zeros(count, dtype=np.bool_)
        blocked = np.zeros(count, dtype=np.bool_)
        emit({"type": "progress", "progress": 0.0, "frame": 0, "total": count,
              "message": "Loading local face detector"})
        first_index = min(source_count - 1, int(math.floor(start * source_fps + 1e-8)))
        # OpenCV seeks to a decoded frame index; sampling thereafter is sequential.
        if first_index and not cap.set(cv2.CAP_PROP_POS_FRAMES, first_index):
            raise RuntimeError("Video decoder cannot seek to the requested start frame")
        decoded_index, bgr = first_index - 1, None
        detector_context = (_detector(model, confidence, True) if job.get('reject_multiple_faces')
                            else _detector(model, confidence))
        with detector_context as detect:
            for index in range(count):
                if cancelled is not None and cancelled():
                    raise InterruptedError("Face capture cancelled")
                source_index = min(source_count - 1, int(math.floor((start + index / fps) * source_fps + 1e-8)))
                while decoded_index < source_index:
                    ok, bgr = cap.read()
                    if not ok or bgr is None:
                        raise RuntimeError(f"Video decode failed at frame {decoded_index + 1}; output was not published")
                    decoded_index += 1
                detection_image = bgr
                pixel_box = None
                frame_crop = crop
                skip_frame = False
                if crop_for_frame is not None:
                    frame_crop = crop_for_frame(source_index, bgr.shape)
                    skip_frame = frame_crop is None
                    if not skip_frame: frame_crop = _crop_box(frame_crop)
                if frame_crop is not None:
                    height, width = bgr.shape[:2]
                    pixel_box = (math.floor(frame_crop[0] * width), math.floor(frame_crop[1] * height),
                                 math.ceil(frame_crop[2] * width), math.ceil(frame_crop[3] * height))
                    left, top, right, bottom = pixel_box
                    detection_image = bgr[top:bottom, left:right]
                timestamp = round(index * 1000 / fps)
                observation = None
                if not skip_frame:
                    rgb = np.ascontiguousarray(cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB))
                    observation = detect(rgb, timestamp)
                blocked[index] = skip_frame or bool(observation and observation.get('blocked'))
                if observation is not None and not blocked[index]:
                    raw_weights[index] = np.asarray(observation['weights'], dtype=np.float32)[sides]
                    detected[index] = True
                if index == count - 1 or index % max(1, round(fps / 4)) == 0:
                    emit({"type": "progress", "progress": (index + 1) / count,
                          "frame": index + 1, "total": count,
                          "message": f"Face capture {index + 1}/{count}"})
        if cancelled is not None and cancelled():
            raise InterruptedError("Face capture cancelled")
        if not bool(detected.any()):
            raise RuntimeError("No face was detected. Use a video with one visible face, or set a tighter face crop.")
        emit({'type': 'progress', 'progress': 1., 'frame': count, 'total': count,
              'message': 'Smoothing expressions and estimating brief detection gaps'})
        from .temporal import process_track
        processed = process_track(raw_weights, detected, fps, BLENDSHAPE_NAMES,
                                  smoothing=smoothing, prediction=prediction, blocked=blocked)
        if cancelled is not None and cancelled():
            raise InterruptedError('Face capture cancelled')
        if (video.stat().st_size, video.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise RuntimeError("Source video changed during capture; output was not published")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", suffix=".partial", dir=str(output.parent)))
        np.savez_compressed(staging / "face.npz", morph_names=np.asarray(BLENDSHAPE_NAMES),
                            weights=processed['weights'], raw_weights=raw_weights,
                            detected=detected, blocked=blocked,
                            predicted=processed['predicted'], interpolated=processed['interpolated'],
                            reliability=processed['reliability'],
                            fps=np.asarray(fps), start_seconds=np.asarray(start + timeline_offset))
        metadata = {
            "schema_version": 2, "complete": True, "provider": "mediapipe-blendshapes",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "detector": "MediaPipe FaceLandmarker CPU", "landmark_topology": 478,
            "source": {"path": str(video), "size": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns,
                       "fps": source_fps, "frame_count": source_count},
            "model": {"path": str(model), "sha256": model_digest},
            "start_seconds": start + timeline_offset, "end_seconds": end + timeline_offset,
            "video_start_seconds": start, "video_end_seconds": end,
            "timeline_offset_seconds": timeline_offset, "fps": fps, "frame_count": count,
            "detected_frames": int(detected.sum()), "morph_names": list(BLENDSHAPE_NAMES),
            "predicted_frames": int(processed['predicted'].sum()),
            "interpolated_frames": int(processed['interpolated'].sum()),
            "blocked_frames": int(blocked.sum()),
            "smoothing": smoothing, "prediction": prediction,
            "temporal_settings": processed['settings'],
            "mirror": job.get('mirror', True), "confidence": confidence,
            "crop": list(crop) if crop is not None else None,
            "crop_mode": "actor_tracking" if crop_for_frame is not None else "fixed",
            "reject_multiple_faces": bool(job.get('reject_multiple_faces', False)),
            "weights_file": "face.npz",
            "weights_sha256": _sha256(staging / "face.npz"),
            "temporal_sha256": _sha256(Path(__file__).with_name("temporal.py")),
            "blendshapes_sha256": _sha256(Path(__file__).with_name("blendshapes.py")),
            "capture_sha256": _sha256(Path(__file__)),
            "versions": {"python": platform.python_version(), "mediapipe": _version("mediapipe"),
                         "numpy": np.__version__, "opencv": cv2.__version__},
            "notes": ["Native MediaPipe 52 expression scores, including non-deforming _neutral; not a phoneme recognizer.",
                      ("Follows this actor's existing tracking region; gaps and multiple detected faces are skipped. Body-tracker identity switches remain possible."
                       if crop_for_frame is not None else "Single face; no identity association with body actors. Input uses the CFR frame clock."),
                      "raw_weights preserves observed scores in the selected anatomical side order, with zeros on missing frames.",
                      "detected records model observations only. Interpolation and bounded trend prediction are estimates, not recovered facial evidence.",
                      "Long gaps fade to neutral; actor absence or ambiguous multiple faces block temporal propagation.",
                      "reliability is a temporal support heuristic, not a calibrated detector confidence."],
        }
        (staging / "face.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        # os.rename does not replace an existing directory on Windows. Recheck
        # also prevents ordinary accidental reuse on other development systems.
        if output.exists():
            raise FileExistsError(f"Face output was created by another job: {output}")
        os.rename(staging, output)
        staging = None
        return {"output_dir": str(output), "npz": str(output / "face.npz"),
                "metadata": str(output / "face.json"), "frame_count": count, "fps": fps,
                "start_seconds": start + timeline_offset, "detected_frames": int(detected.sum()),
                "predicted_frames": int(processed['predicted'].sum()),
                "interpolated_frames": int(processed['interpolated'].sum())}
    finally:
        cap.release()
        if staging is not None and staging.exists():
            # Only this invocation's newly created staging directory is removed.
            if staging.parent.resolve() == output.parent.resolve() and staging.name.startswith(f".{output.name}-"):
                shutil.rmtree(staging)

