from __future__ import annotations

import numpy as np
import pytest

from scripts.action_vqvae.action_window import extract_action_window
from scripts.effect_tokenizer.train_effect_tokenizer import _check_resume_contract


def test_short_episode_is_stationary_padded() -> None:
    actions = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0],
            [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.0],
        ],
        dtype=np.float32,
    )
    absolute_mask = np.asarray([False] * 6 + [True])

    chunk = extract_action_window(
        actions,
        start=0,
        horizon=5,
        absolute_action_mask=absolute_mask,
        pad_incomplete=True,
    )

    assert chunk is not None
    np.testing.assert_array_equal(chunk[:2], actions)
    np.testing.assert_array_equal(chunk[2:, :6], 0.0)
    np.testing.assert_array_equal(chunk[2:, 6], 1.0)


def test_incomplete_tail_window_can_be_dropped() -> None:
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)

    chunk = extract_action_window(
        actions,
        start=1,
        horizon=3,
        absolute_action_mask=np.asarray([False] * 6 + [True]),
        pad_incomplete=False,
    )

    assert chunk is None


def test_complete_window_is_not_modified() -> None:
    actions = np.arange(35, dtype=np.float32).reshape(5, 7)

    chunk = extract_action_window(
        actions,
        start=1,
        horizon=3,
        absolute_action_mask=np.asarray([False] * 6 + [True]),
        pad_incomplete=False,
    )

    np.testing.assert_array_equal(chunk, actions[1:4])


def test_legacy_checkpoint_defaults_to_padding_enabled() -> None:
    saved = {
        "train_dataset_name": "toy",
        "target_control_hz": 10.0,
        "horizon": 10,
        "sampling_stride": 2,
        "action_dim": 7,
        "action_normalization": "q01_q99",
        "effect_descriptor": "effect",
        "effect_motion_scale": 0.1,
        "balance_weights": True,
    }
    current = {**saved, "pad_incomplete_windows": True}

    _check_resume_contract(saved, current)

    with pytest.raises(ValueError, match="pad_incomplete_windows"):
        _check_resume_contract(
            saved,
            {**current, "pad_incomplete_windows": False},
        )
