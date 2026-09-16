"""Byte-bounded, process-local cache for deterministic NumPy targets."""

from collections import OrderedDict
from pathlib import Path

import numpy as np


def file_version(path: str | Path) -> tuple[str, int, int]:
    path = Path(path).resolve()
    info = path.stat()
    return str(path), info.st_mtime_ns, info.st_size


class TargetCache:
    def __init__(self, max_mb: int = 256) -> None:
        if max_mb < 0:
            raise ValueError("cache size must be nonnegative")
        self.max_bytes = max_mb * 1024 * 1024
        self.bytes = 0
        self.entries: OrderedDict[tuple, tuple[np.ndarray, ...]] = OrderedDict()

    def get(self, key: tuple) -> tuple[np.ndarray, ...] | None:
        value = self.entries.get(key)
        if value is not None:
            self.entries.move_to_end(key)
            return tuple(array.copy() for array in value)
        return None

    def put(self, key: tuple, value: tuple[np.ndarray, ...]) -> None:
        size = sum(array.nbytes for array in value)
        if not self.max_bytes or size > self.max_bytes:
            return
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.bytes -= sum(array.nbytes for array in previous)
        while self.entries and self.bytes + size > self.max_bytes:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= sum(array.nbytes for array in removed)
        self.entries[key] = tuple(array.copy() for array in value)
        self.bytes += size

    def __getstate__(self) -> dict:
        # Workers start with an empty cache rather than copying the parent's arrays.
        return {"max_bytes": self.max_bytes, "bytes": 0, "entries": OrderedDict()}
