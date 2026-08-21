import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RLDS_DIR = Path(__file__).parents[1] / "scripts" / "action_vqvae" / "rlds"
frequency_resampling = _load_module(
    "frequency_resampling", RLDS_DIR / "frequency_resampling.py"
)
resample_action_numpy = frequency_resampling.resample_action_numpy
get_oxe_control_frequency_hz = _load_module(
    "control_frequencies", RLDS_DIR / "oxe" / "control_frequencies.py"
).get_oxe_control_frequency_hz


ABSOLUTE_GRIPPER = [False] * 6 + [True]


def _actions(relative_values, gripper_values):
    action = np.zeros((len(relative_values), 7), dtype=np.float32)
    action[:, 0] = relative_values
    action[:, -1] = gripper_values
    return action


def test_downsample_accumulates_relative_deltas_and_holds_gripper():
    source = _actions([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 1.0, 0.0])
    result = resample_action_numpy(
        source,
        absolute_action_mask=ABSOLUTE_GRIPPER,
        source_hz=20.0,
        target_hz=10.0,
    )
    assert result.shape == (2, 7)
    np.testing.assert_allclose(result[:, 0], [3.0, 7.0])
    np.testing.assert_allclose(result[:, -1], [1.0, 0.0])


def test_upsample_splits_relative_deltas_and_repeats_gripper():
    source = _actions([2.0, 4.0], [0.0, 1.0])
    result = resample_action_numpy(
        source,
        absolute_action_mask=ABSOLUTE_GRIPPER,
        source_hz=5.0,
        target_hz=10.0,
    )
    assert result.shape == (4, 7)
    np.testing.assert_allclose(result[:, 0], [1.0, 1.0, 2.0, 2.0])
    np.testing.assert_allclose(result[:, -1], [0.0, 0.0, 1.0, 1.0])


def test_non_integer_ratio_preserves_total_relative_motion():
    source = _actions([1.0] * 5, [1.0] * 5)
    result = resample_action_numpy(
        source,
        absolute_action_mask=ABSOLUTE_GRIPPER,
        source_hz=3.0,
        target_hz=10.0,
    )
    assert result.shape == (17, 7)
    np.testing.assert_allclose(result[:, 0].sum(), 5.0, atol=1e-6)


def test_frequency_table_rejects_unknown_rate():
    with pytest.raises(ValueError, match="invalid control frequency"):
        get_oxe_control_frequency_hz(
            "stanford_mask_vit_converted_externally_to_rlds"
        )


def test_tensorflow_and_numpy_resamplers_match():
    tf = pytest.importorskip("tensorflow")
    source = _actions([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 1.0, 1.0, 0.0])
    expected = resample_action_numpy(
        source,
        absolute_action_mask=ABSOLUTE_GRIPPER,
        source_hz=3.0,
        target_hz=10.0,
    )
    actual = frequency_resampling.resample_action_tensor(
        tf.constant(source),
        absolute_action_mask=ABSOLUTE_GRIPPER,
        source_hz=3.0,
        target_hz=10.0,
    ).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)
