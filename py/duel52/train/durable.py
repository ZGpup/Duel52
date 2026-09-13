"""Writes that survive the process being killed halfway through them.

The training loop resumes from whatever is on disk (``loop.py``, "Pausing"), so everything it
reads back — the incumbent, the optimiser moments, the fit checkpoint, a generation's progress
file — must be either the old version or the new one and never half of each. A 14 MB
``torch.save`` interrupted by a pause would otherwise leave a file that fails to load, or
worse, one that loads.

Each helper writes a sibling ``.tmp``, syncs it, and renames it over the target. A rename
within a directory is atomic on every filesystem this project runs on.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

__all__ = ["append_line", "copy_atomically", "replace_atomically", "write_json_atomically"]


def replace_atomically(path: str | Path, write: Callable[[Path], object]) -> Path:
    """Call ``write(tmp)`` and move ``tmp`` over ``path`` once it is complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _sync_directory(path.parent)
    return path


def write_json_atomically(path: str | Path, data: Any) -> Path:
    return replace_atomically(path, lambda tmp: tmp.write_text(json.dumps(data, indent=2)))


def copy_atomically(source: str | Path, target: str | Path) -> Path:
    return replace_atomically(target, lambda tmp: shutil.copyfile(source, tmp))


def append_line(path: str | Path, line: str) -> None:
    """Append one line and sync it. A kill mid-append can still leave a torn last line, which
    is why the reader of ``log.jsonl`` tolerates exactly that and nothing else."""
    with Path(path).open("a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def _sync_directory(directory: Path) -> None:
    """Make the rename itself durable. Only matters if the machine, not the process, goes
    down, and not every platform lets a directory be opened for it."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
