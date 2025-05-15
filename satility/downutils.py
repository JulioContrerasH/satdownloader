from __future__ import annotations
from typing import Tuple
import numpy as np


def _runs_1d(vec: np.ndarray, *, min_len: int) -> list[Tuple[int, int]]:
    """Return (start, end) indices of True runs of length ≥ *min_len*."""
    runs: list[Tuple[int, int]] = []
    in_run = False
    start = 0
    for i, flag in enumerate(vec):
        if flag and not in_run:
            start, in_run = i, True
        elif not flag and in_run:
            if i - start >= min_len:
                runs.append((start, i))
            in_run = False
    if in_run and (len(vec) - start) >= min_len:
        runs.append((start, len(vec)))
    return runs

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2‑D (rows, cols), got {arr.shape}")
    return arr

def column_runs(mask, *, min_cols: int = 256) -> list[Tuple[int, int]]:
    """Contiguous valid‑column runs."""
    m = _ensure_2d(mask.values if hasattr(mask, "values") else np.asarray(mask))
    return _runs_1d(m.any(axis=0), min_len=min_cols)

def row_runs(mask, *, min_rows: int = 32) -> list[Tuple[int, int]]:
    """Contiguous valid‑row runs."""
    m = _ensure_2d(mask.values if hasattr(mask, "values") else np.asarray(mask))
    return _runs_1d(m.any(axis=1), min_len=min_rows)

def deep_find_lengths(data) -> list[int]:
    """Collect every 'length' value inside any nested structure."""
    if isinstance(data, dict):
        return sum((deep_find_lengths(v) for v in data.values()), []) + (
            [data["length"]] if "length" in data else []
        )
    if isinstance(data, list):
        return sum((deep_find_lengths(el) for el in data), [])
    return []