"""One GPU-heavy job at a time; the OS releases the lock even if the process is killed."""

from __future__ import annotations

import fcntl
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

LOCK_PATH = Path(tempfile.gettempdir()) / "tinylab-gpu.lock"


@contextmanager
def exclusive_gpu(job: str):
    handle = LOCK_PATH.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        holder = handle.read().strip() or "unknown job"
        handle.close()
        print(f"another tinylab GPU job is running ({holder}); wait for it or stop it first", file=sys.stderr)
        raise SystemExit(75)
    handle.seek(0)
    handle.truncate()
    handle.write(f"{job}, pid {os.getpid()}")
    handle.flush()
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
