"""Crash-safe JSON I/O for run directories (traces, run_meta, triage)."""

import json
import os
import time
from pathlib import Path


def atomic_write_json(path: Path, obj) -> None:
    """Write JSON via tmp + os.replace, so a crash or Ctrl-C mid-write never
    leaves a truncated file where a trace (or run_meta) should be."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def read_trace(path: Path) -> tuple[dict | None, str | None]:
    """(trace, None) if path holds a JSON object, else (None, why)."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"
    if not isinstance(data, dict):
        return None, f"expected a JSON object, got {type(data).__name__}"
    return data, None


def move_aside(path: Path, reason: str) -> Path:
    """Rename a trace out of the `trial_*.json` glob (kept for the record).

    The new name ends in `.bak`, so grade/report never pick it up."""
    path = Path(path)
    dest = path.with_name(f"{path.name}.{reason}-{time.strftime('%Y%m%d-%H%M%S')}.bak")
    n = 1
    while dest.exists():
        dest = dest.with_name(f"{dest.stem}-{n}.bak")
        n += 1
    os.replace(path, dest)
    return dest
