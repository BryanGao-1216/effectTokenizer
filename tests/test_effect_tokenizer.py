from __future__ import annotations

import numpy as np
import torch

from scripts.effect_tokenizer.effect_tokenizer import (
    ARTIFACT_VERSION,
    EffectTokenizer,
    MLPEffectVQVAE,
    MLPVQVAEConfig,
    compute_effect_descriptors,
    load_effect_checkpoint,
    unweight_effects,
    vqvae_losses,
    weight_effects,
)


def _small_model() -> MLPEffectVQVAE:
    return MLPEffectVQVAE(
        MLPVQVAEConfig(
            input_dim=7,
            hidden_dim=16,
            latent_dim=4,
            num_hidden_layers=1,
            codebook_size=8,
        )
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


def test_only_gripper_weight_is_applied_after_per_dataset_normalization() -> None:
    effects = np.arange(14, dtype=np.float32).reshape(2, 7)

    weighted = weight_effects(effects, 2.5)

    np.testing.assert_array_equal(weighted[:, :6], effects[:, :6])
    np.testing.assert_allclose(weighted[:, 6], effects[:, 6] * 2.5)
    np.testing.assert_allclose(unweight_effects(weighted, 2.5), effects)


def test_mlp_vqvae_forward_and_loss_backward() -> None:
    torch.manual_seed(3)
    model = _small_model()
    effects = torch.randn(32, 7)

    output = model(effects, usage_temperature=0.7)
    losses = vqvae_losses(
        output,
        effects,
        codebook_loss_weight=1.0,
        commitment_loss_weight=0.25,
        usage_loss_weight=0.01,
    )
    losses["total"].backward()

    assert output.reconstruction.shape == effects.shape
    assert output.encoder_latent.shape == (32, 4)
    assert output.codes.shape == (32,)
    assert output.codes.min() >= 0
    assert output.codes.max() < 8
    assert torch.isfinite(losses["total"])
    assert model.encoder[0].weight.grad is not None
    assert model.codebook.weight.grad is not None
    assert model.decoder[0].weight.grad is not None


def test_tokenizer_checkpoint_and_inference_round_trip(tmp_path) -> None:
    model = _small_model()
    tokenizer = EffectTokenizer(
        model=model,
        gripper_weight=1.5,
        config={
            "data": {
                "action_normalization": "per_dataset_q01_q99_to_minus1_plus1_except_gripper"
            }
        },
    )
    checkpoint = tmp_path / "effect_vqvae.pt"
    tokenizer.save(
        checkpoint,
        training_state={"global_step": 12, "optimizer_state_dict": {}},
    )

    payload = load_effect_checkpoint(checkpoint)
    restored = EffectTokenizer.from_payload(payload)
    effects = np.random.default_rng(9).normal(size=(11, 7)).astype(np.float32)
    original = tokenizer.encode_reconstruct(effects, batch_size=4)
    loaded = restored.encode_reconstruct(effects, batch_size=4)

    assert payload["artifact_version"] == ARTIFACT_VERSION
    assert payload["global_step"] == 12
    assert restored.gripper_weight == tokenizer.gripper_weight
    assert restored.config == tokenizer.config
    np.testing.assert_array_equal(loaded[0], original[0])
    np.testing.assert_allclose(loaded[1], original[1], atol=1e-7)
    np.testing.assert_allclose(loaded[3], original[3], atol=1e-7)
    assert restored.raw_centers.shape == (8, 7)
    assert checkpoint.with_suffix(".json").is_file()
