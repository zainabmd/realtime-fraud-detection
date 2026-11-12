"""Small shared helpers."""
from __future__ import annotations

import os
import time
from pathlib import Path


def atomic_replace(tmp: Path, target: Path, attempts: int = 8, base_delay: float = 0.05) -> None:
    """``os.replace`` with retries.

    The rename is atomic, but on Windows it fails with "Access is denied" while
    another process (a reader, an indexer, antivirus) briefly holds the target
    open. Those locks clear within milliseconds, so retry with backoff rather
    than crash the service.
    """
    for i in range(attempts):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * 2 ** i)
