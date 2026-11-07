"""Bounded offline continuity for observed facial expression coefficients.

Missing expressions are estimates, never new detections. Short bounded gaps use
both visible endpoints; longer gaps use a damped, limited recent trend followed
by a finite fade to neutral. Explicit identity/visibility barriers reset history.
The reliability values are processing heuristics, not detector probabilities.
"""
from __future__ import annotations

import math

import numpy as np


SETTINGS = {
    'algorithm': 'bounded-expression-continuity-v1',
    'interpolate_seconds': .20,
    'trend_window_seconds': .15,
    'trend_damping_seconds': .10,
    'predict_seconds': .20,
    'neutral_seconds': .80,
    'blink_predict_seconds': .04,
    'blink_neutral_seconds': .25,
    'max_velocity_per_second': 1.5,
    'max_extrapolated_change': .12,
    'recovery_seconds': .12,
}


def _validate(raw_weights, detected, fps, names, smoothing, prediction, blocked):
    raw = np.asarray(raw_weights, dtype=np.float64)
    valid = np.asarray(detected)
    try:
        fps = float(fps)
        names = tuple(names)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid expression timing or channel names.') from exc
    if (raw.ndim != 2 or not 1 <= raw.shape[0] <= 54_000
            or not 1 <= raw.shape[1] <= 64 or len(names) != raw.shape[1]
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('Invalid expression dimensions or channel names.')
    if (not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1)
            or not math.isfinite(fps) or not 1 <= fps <= 240):
        raise ValueError('Expression coefficients must be finite in [0, 1], with valid fps.')
    if valid.shape != (len(raw),) or valid.dtype.kind != 'b':
        raise ValueError('Expression detection flags must be a Boolean vector.')
    barriers = np.zeros(len(raw), bool) if blocked is None else np.asarray(blocked)
    if barriers.shape != valid.shape or barriers.dtype.kind != 'b':
        raise ValueError('Expression barrier flags must be a Boolean vector.')
    if np.any(barriers & valid):
        raise ValueError('A blocked frame cannot also be a detected face.')
    if not isinstance(smoothing, bool) or not isinstance(prediction, bool):
        raise ValueError('Smoothing and prediction must be booleans.')
    return raw, valid, fps, names, barriers


def _recent_velocity(raw, valid, last, segment_start, fps):
    """Least-squares trend from recent measured frames only, in weights/second."""
    first = max(segment_start, last - int(math.ceil(SETTINGS['trend_window_seconds'] * fps)))
    indexes = np.flatnonzero(valid[first:last + 1]) + first
    # Ignore old observations across a previous missing interval. They should
    # not create a fresh prediction velocity on reacquisition.
    if len(indexes) > 1:
        breaks = np.flatnonzero(np.diff(indexes) != 1)
        if len(breaks):
            indexes = indexes[breaks[-1] + 1:]
    if len(indexes) < 2:
        return np.zeros(raw.shape[1], np.float64)
    time = (indexes - last) / fps
    time -= time.mean()
    velocity = time @ raw[indexes] / float(time @ time)
    limit = SETTINGS['max_velocity_per_second']
    return np.clip(velocity, -limit, limit)


def _fill_segment(raw, valid, start, end, fps, blink, values, predicted, interpolated, reliability):
    """Fill only intervals after an observation in this identity-safe segment."""
    observations = np.flatnonzero(valid[start:end]) + start
    for position, left in enumerate(observations):
        right = observations[position + 1] if position + 1 < len(observations) else end
        if right - left <= 1:
            continue
        indexes = np.arange(left + 1, right)
        elapsed = (indexes - left) / fps
        if right < end and len(indexes) / fps <= SETTINGS['interpolate_seconds'] + 1e-9:
            alpha = ((indexes - left) / (right - left))[:, None]
            values[indexes] = raw[left] * (1 - alpha) + raw[right] * alpha
            interpolated[indexes] = True
            # Lower confidence at the centre of the gap; endpoints are measured.
            reliability[indexes] = .9 - .2 * np.sin(np.pi * alpha[:, 0])
            continue
        # A 30-minute missing tail has the same finite prediction horizon as a
        # one-second gap. Keep intermediate C-wide arrays bounded by that horizon.
        active = elapsed < SETTINGS['neutral_seconds'] - 1e-9
        indexes, elapsed = indexes[active], elapsed[active]
        if not len(indexes):
            continue
        velocity = _recent_velocity(raw, valid, left, start, fps)
        # Do not invent a new eyelid closure while the eyes are invisible. A
        # measured closure can relax; no oscillator or synthetic blink is used.
        velocity[blink] = np.minimum(velocity[blink], 0)
        hold = np.where(blink, SETTINGS['blink_predict_seconds'], SETTINGS['predict_seconds'])
        finish = np.where(blink, SETTINGS['blink_neutral_seconds'], SETTINGS['neutral_seconds'])
        damping = SETTINGS['trend_damping_seconds']
        effective_time = damping * (1 - np.exp(-np.minimum(elapsed[:, None], hold) / damping))
        change = np.clip(effective_time * velocity, -SETTINGS['max_extrapolated_change'],
                         SETTINGS['max_extrapolated_change'])
        base = np.clip(raw[left] + change, 0, 1)
        phase = np.clip((elapsed[:, None] - hold) / (finish - hold), 0, 1)
        # Smoothstep reaches neutral at a finite time with a zero end slope.
        fade = 1 - phase * phase * (3 - 2 * phase)
        values[indexes] = base * fade
        predicted[indexes] = True
        reliability[indexes] = .6 * (1 - elapsed / SETTINGS['neutral_seconds'])


def _smooth(values, valid, predicted, interpolated, blocked, fps, names):
    result = np.zeros_like(values)
    previous = np.zeros(values.shape[1], np.float64)
    blink = np.asarray([name.startswith('eyeBlink') for name in names])
    mouth_fast = np.asarray([name in ('jawOpen', 'mouthClose') for name in names])
    base_tau = np.where(blink, .012, np.where(mouth_fast, .035, .065))
    recovery_tau = np.where(blink, .020, .075)
    recovery_left = 0.
    dt = 1 / fps
    for index, target in enumerate(values):
        active = valid[index] or predicted[index] or interpolated[index]
        if blocked[index] or not active:
            previous[:] = 0
            recovery_left = 0
            continue
        if index == 0:
            previous = target.copy()
        else:
            if valid[index] and not valid[index - 1] and not interpolated[index - 1]:
                recovery_left = SETTINGS['recovery_seconds']
            # The exponential coefficient uses seconds, not a fixed per-frame
            # factor. Large measured movements get a faster response than jitter.
            tau = base_tau / (1 + 3 * np.abs(target - previous))
            if recovery_left > 0:
                tau = np.maximum(tau, recovery_tau)
                recovery_left = max(0., recovery_left - dt)
            alpha = -np.expm1(-dt / tau)
            previous += alpha * (target - previous)
        result[index] = previous
    return result


def process_track(raw_weights, detected, fps, names, *, smoothing=True, prediction=True, blocked=None):
    """Return animation weights plus honest masks for interpolation/prediction.

    ``raw_weights`` is N x C, with finite [0, 1] coefficients. Values in missing
    frames are ignored. ``detected`` and optional ``blocked`` are Boolean N-vectors.
    Barriers represent ambiguous identities or an absent tracked actor; they stay
    neutral and prevent interpolation or prediction across them. Leading missing
    frames stay neutral. An all-missing track stays entirely neutral.

    ``smoothing=False`` disables denoising and reacquisition easing, independently
    of gap prediction. ``prediction=False`` makes missing frames neutral and resets
    smoothing history there. Prediction and interpolation masks are disjoint and
    never include actual detected frames. Reliability is heuristic support in
    [0, 1], with observed=1 and unsupported/blocked=0; it is not an accuracy score.
    """
    raw, valid, fps, names, barriers = _validate(
        raw_weights, detected, fps, names, smoothing, prediction, blocked)
    values = np.zeros_like(raw)
    values[valid] = raw[valid]
    predicted = np.zeros(len(raw), bool)
    interpolated = np.zeros(len(raw), bool)
    reliability = valid.astype(np.float32)
    if prediction:
        blink = np.asarray([name.startswith('eyeBlink') for name in names])
        bounds = np.r_[-1, np.flatnonzero(barriers), len(raw)]
        for left, right in zip(bounds[:-1], bounds[1:]):
            if right > left + 1:
                _fill_segment(raw, valid, left + 1, right, fps, blink, values,
                              predicted, interpolated, reliability)
    if smoothing:
        values = _smooth(values, valid, predicted, interpolated, barriers, fps, names)
    return dict(weights=np.clip(values, 0, 1).astype(np.float32),
                predicted=predicted, interpolated=interpolated, reliability=reliability,
                settings=dict(SETTINGS, smoothing=smoothing, prediction=prediction))
