from __future__ import annotations

import numpy as np

from scripts.effect_tokenizer.effect_tokenizer import (
    EffectTokenizer,
    compute_effect_descriptors,
    fit_full_kmeans,
    fit_global_standardizer,
    standardize_for_fit,
)


def test_effect_descriptor_accumulates_motion_and_gripper_change() -> None:
    actions = np.zeros((2, 4, 7), dtype=np.float32)
    actions[0, :, :6] = np.arange(1, 7, dtype=np.float32)
    actions[0, :, 6] = [0.0, 0.0, 1.0, 1.0]
    actions[1, :, :6] = -1.0
    actions[1, :, 6] = [1.0, 1.0, 0.0, 0.0]

    descriptors = compute_effect_descriptors(actions)

    np.testing.assert_allclose(descriptors[0, :6], 4 * np.arange(1, 7))
    np.testing.assert_allclose(descriptors[1, :6], -4.0)
    np.testing.assert_allclose(descriptors[:, 6], [1.0, -1.0])


def test_pooled_standardization_is_invertible() -> None:
    descriptors = np.array(
        [[0, 1, 2, 3, 4, 5, -1], [2, 3, 4, 5, 6, 7, 1]],
        dtype=np.float32,
    )
    mean, scale = fit_global_standardizer(descriptors)
    standardized = standardize_for_fit(descriptors, mean, scale, 2.0)
    tokenizer = EffectTokenizer(
        centers=standardized,
        global_mean=mean,
        global_scale=scale,
        gripper_weight=2.0,
        config={},
    )

    np.testing.assert_allclose(tokenizer.raw_centers, descriptors, atol=1e-6)


def test_tokenizer_checkpoint_round_trip(tmp_path) -> None:
    tokenizer = EffectTokenizer(
        centers=np.zeros((2, 7), dtype=np.float32),
        global_mean=np.arange(7, dtype=np.float32),
        global_scale=np.arange(1, 8, dtype=np.float32),
        gripper_weight=1.5,
        config={"target_control_hz": 10.0},
    )
    checkpoint = tmp_path / "effect_tokenizer.pt"

    tokenizer.save(checkpoint)
    restored = EffectTokenizer.load(checkpoint)

    np.testing.assert_array_equal(restored.centers, tokenizer.centers)
    np.testing.assert_array_equal(restored.global_mean, tokenizer.global_mean)
    np.testing.assert_array_equal(restored.global_scale, tokenizer.global_scale)
    assert restored.gripper_weight == tokenizer.gripper_weight
    assert restored.config == tokenizer.config
    assert checkpoint.with_suffix(".json").is_file()


def test_full_lloyd_kmeans_uses_all_points() -> None:
    rng = np.random.default_rng(7)
    left = rng.normal(-2.0, 0.03, size=(100, 7))
    right = rng.normal(2.0, 0.03, size=(100, 7))
    values = np.concatenate([left, right]).astype(np.float32)

    centers, info = fit_full_kmeans(
        values,
        num_clusters=2,
        max_iterations=20,
        n_init=2,
        assignment_batch_size=17,
        init_candidate_samples=len(values),
        seed=3,
        device="cpu",
    )

    centers = centers[np.argsort(centers[:, 0])]
    np.testing.assert_allclose(centers[0], values[:100].mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(centers[1], values[100:].mean(axis=0), atol=1e-5)
    assert info["inertia_per_sample"] < 0.02
