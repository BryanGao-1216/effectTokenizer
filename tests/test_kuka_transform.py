import zlib

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")
pytest.importorskip("tensorflow_graphics")

from scripts.action_vqvae.rlds.oxe.transforms import _decode_kuka_state


def test_decode_kuka_state_accepts_materialized_float_values():
    expected = np.arange(14, dtype=np.float32).reshape(2, 7)

    actual = _decode_kuka_state(tf.constant(expected), width=7)

    np.testing.assert_array_equal(actual.numpy(), expected)


def test_decode_kuka_state_accepts_legacy_zlib_strings():
    expected = np.arange(14, dtype=np.float32).reshape(2, 7)
    compressed = tf.constant(
        [zlib.compress(row.tobytes()) for row in expected],
        dtype=tf.string,
    )

    actual = _decode_kuka_state(compressed, width=7)

    np.testing.assert_array_equal(actual.numpy(), expected)
