"""Time-domain resampling for standardized relative-EFF action trajectories."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


# Included in dataset-statistics cache hashes. Increment whenever resampling
# semantics change so stale q01/q99 files cannot silently be reused.
CONTROL_FREQUENCY_RESAMPLER_VERSION = "interval-overlap-v1"


def _validate_rates_and_mask(
    action_dim: int,
    absolute_action_mask: Sequence[bool],
    source_hz: float,
    target_hz: float,
) -> np.ndarray:
    source_hz = float(source_hz)
    target_hz = float(target_hz)
    if source_hz <= 0 or target_hz <= 0:
        raise ValueError(
            f"source_hz and target_hz must be positive, got {source_hz} and {target_hz}."
        )
    mask = np.asarray(absolute_action_mask, dtype=bool)
    if mask.shape != (action_dim,):
        raise ValueError(
            f"absolute_action_mask must have shape {(action_dim,)}, got {mask.shape}."
        )
    return mask


def resample_action_numpy(
    action: np.ndarray,
    *,
    absolute_action_mask: Sequence[bool],
    source_hz: float,
    target_hz: float,
) -> np.ndarray:
    """Resample ``[time, action_dim]`` actions to ``target_hz``.

    Relative EEF deltas are split or accumulated by interval overlap. Absolute
    dimensions (the gripper for EEF_POS/R6) use zero-order hold at the end of
    each target interval. This preserves total translational/rotational delta
    under the usual small-delta additive interpretation.
    """
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2:
        raise ValueError(f"action must have rank 2, got shape {action.shape}.")
    mask = _validate_rates_and_mask(
        action.shape[-1], absolute_action_mask, source_hz, target_hz
    )
    if action.shape[0] == 0 or np.isclose(source_hz, target_hz):
        return action.copy()

    target_length = int(np.ceil(action.shape[0] * target_hz / source_hz))
    # Integrate the source's piecewise-constant delta velocity at every target
    # boundary, then take adjacent differences. This is O(T), unlike building
    # a dense source-by-target overlap matrix.
    boundary_position = (
        np.arange(target_length + 1, dtype=np.float64) * source_hz / target_hz
    )
    boundary_position = np.clip(boundary_position, 0.0, action.shape[0])
    interval_index = np.minimum(
        np.floor(boundary_position).astype(np.int64), action.shape[0] - 1
    )
    interval_fraction = boundary_position - interval_index
    prefix = np.concatenate(
        [
            np.zeros((1, action.shape[1]), dtype=np.float64),
            np.cumsum(action.astype(np.float64), axis=0),
        ],
        axis=0,
    )
    cumulative = (
        prefix[interval_index]
        + interval_fraction[:, None] * action[interval_index]
    )
    relative = np.diff(cumulative, axis=0)
    target_end = (np.arange(target_length, dtype=np.float64) + 1.0) / target_hz
    hold_indices = np.ceil(target_end * source_hz - 1e-12).astype(np.int64) - 1
    hold_indices = np.clip(hold_indices, 0, action.shape[0] - 1)
    absolute = action[hold_indices]
    return np.where(mask[None], absolute, relative).astype(np.float32)


def resample_action_tensor(
    action: Any,
    *,
    absolute_action_mask: Sequence[bool],
    source_hz: float,
    target_hz: float,
) -> Any:
    """TensorFlow graph-compatible equivalent of :func:`resample_action_numpy`."""
    import tensorflow as tf

    action = tf.cast(action, tf.float32)
    if action.shape.rank != 2:
        raise ValueError(f"action must have rank 2, got shape {action.shape}.")
    action_dim = action.shape[-1]
    if action_dim is None:
        raise ValueError("action dimension must be statically known for resampling.")
    mask = _validate_rates_and_mask(
        int(action_dim), absolute_action_mask, source_hz, target_hz
    )
    if np.isclose(source_hz, target_hz):
        return action

    source_hz_tensor = tf.constant(float(source_hz), tf.float64)
    target_hz_tensor = tf.constant(float(target_hz), tf.float64)
    source_length = tf.shape(action)[0]

    def resample_nonempty():
        target_length = tf.cast(
            tf.math.ceil(
                tf.cast(source_length, tf.float64)
                * target_hz_tensor
                / source_hz_tensor
            ),
            tf.int32,
        )
        boundary_position = (
            tf.cast(tf.range(target_length + 1), tf.float64)
            * source_hz_tensor
            / target_hz_tensor
        )
        boundary_position = tf.clip_by_value(
            boundary_position, 0.0, tf.cast(source_length, tf.float64)
        )
        interval_index = tf.minimum(
            tf.cast(tf.floor(boundary_position), tf.int32), source_length - 1
        )
        interval_fraction = boundary_position - tf.cast(interval_index, tf.float64)
        action64 = tf.cast(action, tf.float64)
        prefix = tf.concat(
            [
                tf.zeros((1, int(action_dim)), dtype=tf.float64),
                tf.cumsum(action64, axis=0),
            ],
            axis=0,
        )
        cumulative = (
            tf.gather(prefix, interval_index)
            + interval_fraction[:, None] * tf.gather(action64, interval_index)
        )
        relative = tf.cast(cumulative[1:] - cumulative[:-1], tf.float32)
        target_end = (
            tf.cast(tf.range(target_length), tf.float64) + 1.0
        ) / target_hz_tensor
        hold_indices = tf.cast(
            tf.math.ceil(target_end * source_hz_tensor - 1e-12), tf.int32
        ) - 1
        hold_indices = tf.clip_by_value(hold_indices, 0, source_length - 1)
        absolute = tf.gather(action, hold_indices)
        result = tf.where(tf.constant(mask)[None], absolute, relative)
        result.set_shape([None, int(action_dim)])
        return result

    return tf.cond(
        source_length > 0,
        resample_nonempty,
        lambda: tf.zeros((0, int(action_dim)), dtype=tf.float32),
    )
