"""Pure NumPy helpers for constructing fixed-horizon action windows."""

from __future__ import annotations

import numpy as np


def extract_action_window(
    actions: np.ndarray,
    *,
    start: int,
    horizon: int,
    absolute_action_mask: np.ndarray,
    pad_incomplete: bool,
) -> np.ndarray | None:
    """Return one action window, optionally neutral-padding past episode end.

    Relative action dimensions are padded with zero. Absolute action dimensions
    are padded by repeating their final valid value. This represents a stationary
    robot while preserving states such as the last gripper command.
    """
    values = np.asarray(actions, dtype=np.float32)
    mask = np.asarray(absolute_action_mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"actions must have shape [T, A], got {values.shape}.")
    if values.shape[0] == 0:
        return None
    if mask.shape != (values.shape[1],):
        raise ValueError(
            f"absolute_action_mask must have shape {(values.shape[1],)}, got {mask.shape}."
        )
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    if start < 0 or start >= values.shape[0]:
        raise ValueError(
            f"start must be in [0, {values.shape[0] - 1}], got {start}."
        )

    stop = start + horizon
    if stop <= values.shape[0]:
        return values[start:stop].copy()
    if not pad_incomplete:
        return None

    valid = values[start:].copy()
    padding_length = horizon - len(valid)
    neutral = np.where(
        mask,
        values[-1],
        np.zeros(values.shape[1], dtype=np.float32),
    )
    padding = np.repeat(neutral[None], padding_length, axis=0)
    return np.concatenate([valid, padding], axis=0).astype(np.float32, copy=False)
