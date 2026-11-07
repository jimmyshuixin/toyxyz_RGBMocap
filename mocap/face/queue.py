"""Pure queue helpers called only by explicit offline reconstruction requests.

Do not call append_capture_jobs from generic start_jobs: preview, correction and
export actions must keep their existing behavior regardless of the checkbox.
"""
from __future__ import annotations

import json
from pathlib import Path

SUPPORTED_MODELS = frozenset(('exmr', 'flowhmr', 'gvhmr'))
STAGE = 'face_capture'
_POST_STAGES = frozenset(('correct', 'export', 'flow_export', STAGE))


def options_snapshot(options):
    """Copy just this feature's user options without retaining a mutable UI dict."""
    options = options or {}
    enabled = options.get('enabled', False)
    smoothing = options.get('smoothing', True)
    if not isinstance(enabled, bool) or not isinstance(smoothing, bool):
        raise ValueError('Face enabled and smoothing options must be booleans.')
    return dict(enabled=enabled, smoothing=smoothing)


def face_arguments(options, project=None):
    value = options_snapshot(options)
    args = ['--face-options', json.dumps(value, sort_keys=True, separators=(',', ':'))]
    if project is not None:
        args = ['--project', str(Path(project)), *args]
    return args


def _project_argument(arguments):
    arguments = arguments or []
    try:
        index = arguments.index('--project')
        return str(Path(arguments[index + 1]))
    except (ValueError, IndexError):
        return None


def append_capture_jobs(jobs, *, model, options, projects=None):
    """Append face passes after this reconstruction's body work, without mutation.

    projects=None leaves one project unresolved until prepare has selected it.
    Explicit projects are actor/take paths from person_prepared, never all rows
    in a library. Passing an empty list intentionally adds no work. Repeated
    calls do not duplicate already queued face passes for the same project.
    """
    result = list(jobs)
    value = options_snapshot(options)
    if model not in SUPPORTED_MODELS or not value['enabled']:
        return result
    existing = {_project_argument(args) for stage, args in result if stage == STAGE}
    for project in ([None] if projects is None else projects):
        key = None if project is None else str(Path(project))
        if key in existing:
            continue
        result.append((STAGE, face_arguments(value, project)))
        existing.add(key)
    return result


def split_people_stages(stages):
    """Keep facial processing after all body/contact/export phases for a take."""
    stages = list(stages)
    inference = [stage for stage in stages if stage not in _POST_STAGES]
    post = [stage for stage in stages if stage in _POST_STAGES and stage != STAGE]
    if STAGE in stages:
        post.append(STAGE)
    return inference, post
