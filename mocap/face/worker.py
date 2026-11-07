"""Dedicated optional face-runtime entry point; stdout is JSON-lines IPC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback


def _emit(event):
    print(json.dumps(event, ensure_ascii=True, allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Capture local MediaPipe expressions with temporal gap estimation")
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        job = json.loads(args.request.read_text(encoding="utf-8"))
        from .capture import run
        result = run(job, _emit)
        _emit({"type": "complete", "result": result})
        return 0
    except KeyboardInterrupt:
        _emit({"type": "error", "message": "Face capture cancelled"})
        return 130
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit({"type": "error", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
