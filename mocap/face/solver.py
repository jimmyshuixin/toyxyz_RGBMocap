# SPDX-License-Identifier: GPL-3.0-only
# Python adaptation of AmyangXYZ/reze-mipo, src/lib/face-blendshape-solver.ts
# and the scalar OneEuroFilter in src/lib/filters.ts, commit
# 234b51553f44d2578606a91e1aecc7267b964854. Modified for RGB MoCap, 2026-11-07.
# See third_party/reze-mipo/PROVENANCE.json and LICENSE for original sources.
"""Geometric MMD expressions from normalized MediaPipe face mesh landmarks.

Preserves upstream normalized xyz distances and image-y smile threshold (these
are deliberately not aspect-ratio corrected). Missing/invalid faces become rest
and reset filter history. Eye quaternions are MMD local XYZW, q_y * q_x.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

UPSTREAM_COMMIT = "234b51553f44d2578606a91e1aecc7267b964854"
MORPH_NAMES = ("まばたき", "ウィンク", "ウィンク右", "あ", "ワ")
MORPH_ALIASES = {
    "まばたき": ("瞬き",),
    "ウィンク": ("ウィンク２",),
    "ウィンク右": ("ウィンク右２", "ウインク右"),
    "あ": ("あ２",),
    "ワ": ("にっこり", "にやり"),
}
_USED = (33, 133, 159, 145, 468, 362, 263, 386, 374, 473, 13, 14, 61, 291)
_IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class FaceThresholds:
    eye_open: float = 0.3
    eye_closed: float = 0.1
    mouth_open: float = 0.18
    smile: float = 0.008

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.eye_open, self.eye_closed, self.mouth_open, self.smile)):
            raise ValueError("Face thresholds must be finite")
        if not 0 <= self.eye_closed < self.eye_open or min(self.mouth_open, self.smile) < 0:
            raise ValueError("Require 0 <= eye_closed < eye_open and nonnegative mouth/smile thresholds")


@dataclass(frozen=True)
class FaceResult:
    morph_weights: dict[str, float]
    eye_rotation_xyzw: tuple[float, float, float, float] = _IDENTITY
    detected: bool = False


class OneEuroFilter:
    """Upstream scalar filter; timestamps in milliseconds, cutoff in Hz."""
    def __init__(self, min_cutoff: float = 2.0, beta: float = 15.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.previous = None
        self.derivative = 0.0
        self.timestamp = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))

    def filter(self, value: float, timestamp_ms: float) -> float:
        if not math.isfinite(value) or not math.isfinite(timestamp_ms):
            raise ValueError("Filter values and timestamps must be finite")
        dt = 0.0 if self.timestamp is None else (timestamp_ms - self.timestamp) / 1000.0
        if self.previous is None or dt <= 0 or dt > 1.0:
            self.previous, self.timestamp, self.derivative = value, timestamp_ms, 0.0
            return value
        a_d = self._alpha(self.d_cutoff, dt)
        self.derivative = a_d * ((value - self.previous) / dt) + (1 - a_d) * self.derivative
        a = self._alpha(self.min_cutoff + self.beta * abs(self.derivative), dt)
        self.previous = a * value + (1 - a) * self.previous
        self.timestamp = timestamp_ms
        return self.previous


class FaceSolver:
    def __init__(self, *, smoothing: bool = True, mirror: bool = True,
                 thresholds: FaceThresholds | None = None):
        self.smoothing, self.mirror = bool(smoothing), bool(mirror)
        self.thresholds = thresholds or FaceThresholds()
        self._filters = [OneEuroFilter(beta=15 if i < 4 else 10) for i in range(6)]
        self._names = dict(zip(MORPH_NAMES, MORPH_NAMES))

    def configure(self, available_morphs: Iterable[str]):
        available = set(available_morphs)
        self._names = {
            name: next((n for n in (name, *MORPH_ALIASES[name]) if n in available), name)
            for name in MORPH_NAMES
        }

    def reset(self):
        for filt in self._filters:
            filt.reset()

    def rest(self) -> FaceResult:
        return FaceResult({self._names[n]: 0.0 for n in MORPH_NAMES})

    @staticmethod
    def _point(point):
        if isinstance(point, Mapping):
            return float(point["x"]), float(point["y"]), float(point.get("z", 0.0))
        if hasattr(point, "x"):
            return float(point.x), float(point.y), float(getattr(point, "z", 0.0))
        return float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0

    def solve(self, landmarks, timestamp_ms: float = 0.0) -> FaceResult:
        if not math.isfinite(timestamp_ms):
            raise ValueError("Face timestamp must be finite")
        try:
            if landmarks is None or len(landmarks) < 474:
                raise ValueError("Missing iris landmarks")
            p = {i: self._point(landmarks[i]) for i in _USED}
            if not all(math.isfinite(v) for pt in p.values() for v in pt):
                raise ValueError("Nonfinite landmarks")
            if min(math.dist(p[33], p[133]), math.dist(p[362], p[263]),
                   math.dist(p[61], p[291])) <= 1e-8:
                raise ValueError("Degenerate face")
        except (ValueError, TypeError, IndexError, KeyError, OverflowError):
            self.reset()
            return self.rest()

        def openness(a, b, top, bottom):
            ratio = math.dist(p[top], p[bottom]) / math.dist(p[a], p[b])
            return _clamp((ratio - self.thresholds.eye_closed) /
                          (self.thresholds.eye_open - self.thresholds.eye_closed))

        def gaze(a, b, iris):
            width = abs(p[a][0] - p[b][0])
            if width <= 1e-8:
                return 0.0, 0.0
            return (_clamp((p[iris][0] - (p[a][0] + p[b][0]) / 2) / (width * .5), -1, 1),
                    _clamp((p[iris][1] - (p[a][1] + p[b][1]) / 2) / (width * .25), -.5, .5))

        left, right = openness(33, 133, 159, 145), openness(362, 263, 386, 374)
        # Keep upstream's mirror=True default: its image-side "left" indices
        # (33 etc.) are MediaPipe's anatomical RIGHT eye. Swapping those here
        # maps the anatomical LEFT eye (362 etc.) to MMD's left wink, matching
        # body retargeting. False intentionally exchanges the two eye channels.
        if self.mirror:
            left, right = right, left
        mouth = _clamp((math.dist(p[13], p[14]) / math.dist(p[61], p[291]) -
                        self.thresholds.mouth_open) / .2)
        smile = _clamp(((p[13][1] + p[14][1] - p[61][1] - p[291][1]) / 2 -
                        self.thresholds.smile) * 120)
        ga, gb = gaze(33, 133, 468), gaze(362, 263, 473)
        values = [left, right, mouth, smile, (ga[0] + gb[0]) / 2, (ga[1] + gb[1]) / 2]
        if self.smoothing:
            values = [f.filter(v, timestamp_ms) for f, v in zip(self._filters, values)]
        left, right, mouth, smile, gx, gy = values
        lb, rb = 1 - left, 1 - right
        weights = ((lb + rb) / 2, lb if lb > .5 and rb < .3 else 0.0,
                   rb if rb > .5 and lb < .3 else 0.0, mouth, smile)
        # Reze Engine Quat.fromEuler(pitch, yaw, 0): q_y * q_x (XYZW).
        sx, cx = math.sin(gy * math.pi / 24), math.cos(gy * math.pi / 24)
        sy, cy = math.sin(-gx * math.pi / 12), math.cos(-gx * math.pi / 12)
        return FaceResult({self._names[n]: _clamp(v) for n, v in zip(MORPH_NAMES, weights)},
                          (cy * sx, sy * cx, -sy * sx, cy * cx), True)

