"""MLP VQ-VAE for per-dataset-normalized OpenX endpoint effects."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


ARTIFACT_VERSION = 3
DESCRIPTOR_NAMES = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "delta_gripper",
)


def choose_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_effect_descriptors(actions: np.ndarray) -> np.ndarray:
    """Convert ``[..., horizon, 7]`` action chunks to signed endpoint effects.

    Translation and OpenX-standardized RPY deltas are accumulated over the
    chunk. The gripper effect is the final-minus-initial absolute command.
    """
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim < 2 or values.shape[-1] != len(DESCRIPTOR_NAMES):
        raise ValueError(
            "Expected action chunks shaped "
            f"[..., horizon, {len(DESCRIPTOR_NAMES)}], got {values.shape}."
        )
    if values.shape[-2] <= 0:
        raise ValueError("Action chunks must contain at least one timestep.")
    position = values[..., :, :3].sum(axis=-2, dtype=np.float64)
    rotation = values[..., :, 3:6].sum(axis=-2, dtype=np.float64)
    gripper = values[..., -1, 6] - values[..., 0, 6]
    return np.concatenate(
        [position, rotation, gripper[..., None]], axis=-1
    ).astype(np.float32)


def weight_effects(
    descriptors: np.ndarray,
    gripper_weight: float,
    effect_scale: np.ndarray | tuple[float, ...] | None = None,
) -> np.ndarray:
    """Apply the reversible model-conditioning scale after data normalization."""
    values = np.asarray(descriptors, dtype=np.float32).copy()
    if effect_scale is not None:
        scale = np.asarray(effect_scale, dtype=np.float32)
        if scale.shape != (len(DESCRIPTOR_NAMES),) or np.any(scale <= 0):
            raise ValueError(
                f"effect_scale must contain {len(DESCRIPTOR_NAMES)} positive values."
            )
        values *= scale
    values[..., -1] *= float(gripper_weight)
    return values


def unweight_effects(
    weighted: np.ndarray,
    gripper_weight: float,
    effect_scale: np.ndarray | tuple[float, ...] | None = None,
) -> np.ndarray:
    values = np.asarray(weighted, dtype=np.float32).copy()
    values[..., -1] /= float(gripper_weight)
    if effect_scale is not None:
        scale = np.asarray(effect_scale, dtype=np.float32)
        if scale.shape != (len(DESCRIPTOR_NAMES),) or np.any(scale <= 0):
            raise ValueError(
                f"effect_scale must contain {len(DESCRIPTOR_NAMES)} positive values."
            )
        values /= scale
    return values


def _squared_distance(values: Tensor, centers: Tensor) -> Tensor:
    values_float = values.float()
    centers_float = centers.float()
    return (
        values_float.square().sum(dim=-1, keepdim=True)
        + centers_float.square().sum(dim=-1).unsqueeze(0)
        - 2.0 * values_float @ centers_float.t()
    ).clamp_min_(0.0)


@dataclass(frozen=True)
class MLPVQVAEConfig:
    input_dim: int = len(DESCRIPTOR_NAMES)
    hidden_dim: int = 128
    latent_dim: int = 16
    num_hidden_layers: int = 2
    codebook_size: int = 256
    normalize_latents: bool = True

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "normalize_latents":
                continue
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.input_dim != len(DESCRIPTOR_NAMES):
            raise ValueError(
                f"This effect contract requires input_dim={len(DESCRIPTOR_NAMES)}."
            )
        if self.codebook_size < 2:
            raise ValueError("codebook_size must be at least 2.")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MLPVQVAEConfig":
        integer_fields = (
            "input_dim",
            "hidden_dim",
            "latent_dim",
            "num_hidden_layers",
            "codebook_size",
        )
        kwargs = {
            key: int(values[key]) for key in integer_fields if key in values
        }
        kwargs["normalize_latents"] = bool(values.get("normalize_latents", False))
        return cls(**kwargs)


def _make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(num_hidden_layers):
        layers.extend([nn.Linear(current_dim, hidden_dim), nn.GELU()])
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


@dataclass
class VQForwardOutput:
    reconstruction: Tensor
    encoder_latent: Tensor
    quantized_latent: Tensor
    straight_through_latent: Tensor
    codes: Tensor
    nearest_squared_distance: Tensor
    relative_margin: Tensor
    soft_usage: Tensor
    soft_assignment_entropy: Tensor


class MLPEffectVQVAE(nn.Module):
    """A single-codebook VQ-VAE over seven-dimensional endpoint effects."""

    def __init__(self, config: MLPVQVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _make_mlp(
            config.input_dim,
            config.latent_dim,
            config.hidden_dim,
            config.num_hidden_layers,
        )
        self.codebook = nn.Embedding(config.codebook_size, config.latent_dim)
        self.decoder = _make_mlp(
            config.latent_dim,
            config.input_dim,
            config.hidden_dim,
            config.num_hidden_layers,
        )
        bound = config.latent_dim**-0.5
        nn.init.uniform_(self.codebook.weight, -bound, bound)

    @property
    def codebook_size(self) -> int:
        return self.config.codebook_size

    def encode(self, effects: Tensor) -> Tensor:
        latent = self.encoder(effects)
        if self.config.normalize_latents:
            latent = F.normalize(latent.float(), dim=-1, eps=1e-6)
        return latent

    def normalized_codebook(self) -> Tensor:
        if self.config.normalize_latents:
            return F.normalize(self.codebook.weight.float(), dim=-1, eps=1e-6)
        return self.codebook.weight

    @torch.no_grad()
    def normalize_codebook_(self) -> None:
        if self.config.normalize_latents:
            self.codebook.weight.copy_(
                F.normalize(self.codebook.weight.float(), dim=-1, eps=1e-6)
            )

    def quantize(
        self,
        encoder_latent: Tensor,
        *,
        usage_temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if usage_temperature <= 0:
            raise ValueError("usage_temperature must be positive.")
        codebook = self.normalized_codebook()
        distances = _squared_distance(encoder_latent, codebook)
        nearest, indices = torch.topk(
            distances,
            k=min(2, self.codebook_size),
            largest=False,
            dim=-1,
        )
        codes = indices[:, 0]
        quantized = F.embedding(codes, codebook)
        if nearest.shape[1] == 1:
            margin = torch.ones_like(nearest[:, 0])
        else:
            margin = (nearest[:, 1] - nearest[:, 0]) / nearest[:, 1].clamp_min(
                1e-8
            )
        soft_assignments = torch.softmax(
            -distances / float(usage_temperature), dim=-1
        )
        soft_usage = soft_assignments.mean(dim=0)
        soft_assignment_entropy = -torch.sum(
            soft_assignments.float().clamp_min(1e-12)
            * torch.log(soft_assignments.float().clamp_min(1e-12)),
            dim=-1,
        ).mean()
        return (
            quantized,
            codes,
            nearest[:, 0],
            margin,
            soft_usage,
            soft_assignment_entropy,
        )

    def forward(
        self,
        effects: Tensor,
        *,
        usage_temperature: float = 1.0,
    ) -> VQForwardOutput:
        encoder_latent = self.encode(effects)
        (
            quantized,
            codes,
            distance,
            margin,
            soft_usage,
            soft_assignment_entropy,
        ) = self.quantize(
            encoder_latent,
            usage_temperature=usage_temperature,
        )
        straight_through = encoder_latent + (quantized - encoder_latent).detach()
        reconstruction = self.decoder(straight_through)
        return VQForwardOutput(
            reconstruction=reconstruction,
            encoder_latent=encoder_latent,
            quantized_latent=quantized,
            straight_through_latent=straight_through,
            codes=codes,
            nearest_squared_distance=distance,
            relative_margin=margin,
            soft_usage=soft_usage,
            soft_assignment_entropy=soft_assignment_entropy,
        )

    def decode_codes(self, codes: Tensor) -> Tensor:
        return self.decoder(F.embedding(codes, self.normalized_codebook()))


class DeadCodeTracker:
    """Track hard assignments and revive persistently unused codebook entries."""

    def __init__(
        self,
        codebook_size: int,
        *,
        decay: float = 0.99,
        threshold: float = 0.1,
        patience: int = 100,
        warmup_steps: int = 500,
        max_resets_per_step: int = 32,
        device: str | torch.device = "cpu",
        initial_usage: Tensor | np.ndarray | None = None,
    ) -> None:
        if codebook_size < 2:
            raise ValueError("codebook_size must be at least 2.")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1).")
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative.")
        if patience <= 0 or warmup_steps < 0 or max_resets_per_step <= 0:
            raise ValueError("Invalid dead-code timing configuration.")
        self.codebook_size = int(codebook_size)
        self.decay = float(decay)
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.warmup_steps = int(warmup_steps)
        self.max_resets_per_step = int(max_resets_per_step)
        self.device = torch.device(device)
        self.ema_usage = torch.zeros(
            self.codebook_size, dtype=torch.float32, device=self.device
        )
        if initial_usage is not None:
            usage = torch.as_tensor(
                initial_usage, dtype=torch.float32, device=self.device
            )
            if usage.shape != self.ema_usage.shape or torch.any(usage < 0):
                raise ValueError("initial_usage has an invalid shape or value.")
            if float(usage.sum()) > 0:
                self.ema_usage.copy_(usage / usage.sum())
        self.inactive_steps = torch.zeros(
            self.codebook_size, dtype=torch.long, device=self.device
        )
        self.updates = 0
        self.total_resets = 0

    @property
    def minimum_probability(self) -> float:
        """Threshold as a fraction of uniform code usage."""
        return self.threshold / self.codebook_size

    def state_dict(self) -> dict[str, Any]:
        return {
            "ema_usage": self.ema_usage.detach().cpu(),
            "inactive_steps": self.inactive_steps.detach().cpu(),
            "updates": self.updates,
            "total_resets": self.total_resets,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ema_usage = torch.as_tensor(
            state["ema_usage"], dtype=torch.float32, device=self.device
        )
        inactive_steps = torch.as_tensor(
            state["inactive_steps"], dtype=torch.long, device=self.device
        )
        if ema_usage.shape != self.ema_usage.shape:
            raise ValueError("Dead-code state does not match codebook size.")
        self.ema_usage.copy_(ema_usage)
        self.inactive_steps.copy_(inactive_steps)
        self.updates = int(state.get("updates", 0))
        self.total_resets = int(state.get("total_resets", 0))

    def metrics(self) -> dict[str, float | int]:
        probabilities = self.ema_usage / self.ema_usage.sum().clamp_min(1e-12)
        active = probabilities > 0
        entropy = -torch.sum(
            probabilities[active] * torch.log(probabilities[active])
        )
        low_usage = probabilities < self.minimum_probability
        return {
            "ema_perplexity": float(torch.exp(entropy)),
            "ema_dead_codes": int(low_usage.sum()),
            "total_code_resets": self.total_resets,
        }

    @staticmethod
    def _clear_optimizer_rows(
        optimizer: torch.optim.Optimizer | None,
        parameter: nn.Parameter,
        rows: Tensor,
    ) -> None:
        if optimizer is None or parameter not in optimizer.state:
            return
        for value in optimizer.state[parameter].values():
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value[rows] = 0

    @torch.no_grad()
    def update_and_reset(
        self,
        model: MLPEffectVQVAE,
        *,
        codes: Tensor,
        encoder_latent: Tensor,
        squared_distances: Tensor,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> Tensor:
        counts = torch.bincount(
            codes.detach().to(self.device), minlength=self.codebook_size
        ).float()
        batch_usage = counts / counts.sum().clamp_min(1.0)
        if self.updates == 0 and float(self.ema_usage.sum()) == 0.0:
            self.ema_usage.copy_(batch_usage)
        else:
            self.ema_usage.mul_(self.decay).add_(
                batch_usage, alpha=1.0 - self.decay
            )
        normalized_usage = self.ema_usage / self.ema_usage.sum().clamp_min(1e-12)
        below_threshold = normalized_usage < self.minimum_probability
        self.inactive_steps = torch.where(
            below_threshold,
            self.inactive_steps + 1,
            torch.zeros_like(self.inactive_steps),
        )
        self.updates += 1
        if self.updates < self.warmup_steps:
            return torch.empty(0, dtype=torch.long, device=self.device)

        dead = torch.nonzero(
            self.inactive_steps >= self.patience, as_tuple=False
        ).flatten()
        if len(dead) == 0:
            return dead
        dead = dead[: min(self.max_resets_per_step, len(dead), len(codes))]
        candidate_order = torch.argsort(
            squared_distances.detach().to(self.device), descending=True
        )
        candidates = encoder_latent.detach().to(self.device)[
            candidate_order[: len(dead)]
        ].float()
        if model.config.normalize_latents:
            candidates = F.normalize(candidates, dim=-1, eps=1e-6)
        model.codebook.weight[dead] = candidates.to(model.codebook.weight.dtype)
        model.normalize_codebook_()
        self._clear_optimizer_rows(optimizer, model.codebook.weight, dead)
        self.ema_usage[dead] = 1.0 / self.codebook_size
        self.inactive_steps[dead] = 0
        self.total_resets += len(dead)
        return dead


def vqvae_losses(
    output: VQForwardOutput,
    target: Tensor,
    *,
    codebook_loss_weight: float,
    commitment_loss_weight: float,
    usage_loss_weight: float,
) -> dict[str, Tensor]:
    reconstruction = F.mse_loss(output.reconstruction.float(), target.float())
    codebook = F.mse_loss(
        output.quantized_latent.float(), output.encoder_latent.detach().float()
    )
    commitment = F.mse_loss(
        output.encoder_latent.float(), output.quantized_latent.detach().float()
    )
    probabilities = output.soft_usage.float().clamp_min(1e-12)
    marginal_kl = torch.sum(
        probabilities
        * torch.log(probabilities * probabilities.new_tensor(len(probabilities)))
    )
    log_codebook_size = math.log(len(probabilities))
    assignment_entropy = output.soft_assignment_entropy.float()
    # Minimizing both terms maximizes marginal code diversity while preventing
    # the all-samples-uniform soft assignment that hid hard-code collapse.
    usage = (marginal_kl + assignment_entropy) / log_codebook_size
    total = (
        reconstruction
        + float(codebook_loss_weight) * codebook
        + float(commitment_loss_weight) * commitment
        + float(usage_loss_weight) * usage
    )
    return {
        "total": total,
        "reconstruction": reconstruction,
        "codebook": codebook,
        "commitment": commitment,
        "usage": usage,
        "usage_balance": marginal_kl / log_codebook_size,
        "assignment_entropy": assignment_entropy / log_codebook_size,
    }


def load_effect_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid effect VQ-VAE checkpoint at {path}.")
    version = int(payload.get("artifact_version", 0))
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported artifact version {version}; expected {ARTIFACT_VERSION}. "
            "Version 3 adds spherical quantization and reversible effect scaling; "
            "older checkpoints cannot be safely resumed or evaluated with this model."
        )
    return payload


@dataclass
class EffectTokenizer:
    """Inference wrapper for effects already normalized by each OXE dataset."""

    model: MLPEffectVQVAE
    gripper_weight: float
    config: dict[str, Any]
    effect_scale: tuple[float, ...] = (1.0,) * len(DESCRIPTOR_NAMES)

    def __post_init__(self) -> None:
        if self.gripper_weight <= 0:
            raise ValueError("gripper_weight must be strictly positive.")
        self.effect_scale = tuple(float(value) for value in self.effect_scale)
        if (
            len(self.effect_scale) != len(DESCRIPTOR_NAMES)
            or any(value <= 0 for value in self.effect_scale)
        ):
            raise ValueError(
                f"effect_scale must contain {len(DESCRIPTOR_NAMES)} positive values."
            )

    @property
    def codebook_size(self) -> int:
        return self.model.codebook_size

    @property
    def descriptor_dim(self) -> int:
        return self.model.config.input_dim

    @property
    def latent_dim(self) -> int:
        return self.model.config.latent_dim

    @property
    def centers(self) -> np.ndarray:
        return (
            self.model.normalized_codebook()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    @property
    def raw_centers(self) -> np.ndarray:
        """Decode each latent code into an endpoint-effect prototype."""
        was_training = self.model.training
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            codes = torch.arange(self.codebook_size, device=device)
            decoded = self.model.decode_codes(codes).float().cpu().numpy()
        if was_training:
            self.model.train()
        return self.unweight(decoded)

    def weight(self, descriptors: np.ndarray) -> np.ndarray:
        return weight_effects(
            descriptors,
            self.gripper_weight,
            self.effect_scale,
        )

    def unweight(self, weighted: np.ndarray) -> np.ndarray:
        return unweight_effects(
            weighted,
            self.gripper_weight,
            self.effect_scale,
        )

    def encode_reconstruct(
        self,
        descriptors: np.ndarray,
        *,
        batch_size: int = 65_536,
        device: str | torch.device = "cpu",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        weighted = self.weight(descriptors)
        if weighted.ndim != 2 or weighted.shape[1] != self.descriptor_dim:
            raise ValueError(
                f"Expected descriptors [N, {self.descriptor_dim}], got {weighted.shape}."
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if len(weighted) == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty((0, self.descriptor_dim), dtype=np.float32),
            )

        torch_device = torch.device(device)
        self.model.to(torch_device)
        was_training = self.model.training
        self.model.eval()
        labels: list[np.ndarray] = []
        distances: list[np.ndarray] = []
        margins: list[np.ndarray] = []
        reconstructions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(weighted), batch_size):
                batch = torch.from_numpy(
                    weighted[start : start + batch_size]
                ).to(torch_device)
                output = self.model(batch)
                labels.append(output.codes.cpu().numpy())
                distances.append(output.nearest_squared_distance.cpu().numpy())
                margins.append(output.relative_margin.cpu().numpy())
                reconstructions.append(output.reconstruction.float().cpu().numpy())
        if was_training:
            self.model.train()
        reconstructed_raw = self.unweight(
            np.concatenate(reconstructions).astype(np.float32)
        )
        return (
            np.concatenate(labels).astype(np.int64),
            np.concatenate(distances).astype(np.float32),
            np.concatenate(margins).astype(np.float32),
            reconstructed_raw,
        )

    def assign(
        self,
        descriptors: np.ndarray,
        *,
        batch_size: int = 65_536,
        device: str | torch.device = "cpu",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels, distances, margins, _ = self.encode_reconstruct(
            descriptors,
            batch_size=batch_size,
            device=device,
        )
        return labels, distances, margins

    def save(
        self,
        path: str | Path,
        *,
        training_state: dict[str, Any] | None = None,
    ) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "artifact_version": ARTIFACT_VERSION,
            "model_type": "mlp_effect_vqvae",
            "model_config": asdict(self.model.config),
            "model_state_dict": self.model.state_dict(),
            "gripper_weight": float(self.gripper_weight),
            "effect_scale": list(self.effect_scale),
            "config": dict(self.config),
            "descriptor_names": list(DESCRIPTOR_NAMES),
        }
        if training_state:
            payload.update(training_state)
        temporary = output.with_name(output.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)

        metadata = {
            "artifact_version": ARTIFACT_VERSION,
            "model_type": "mlp_effect_vqvae",
            "checkpoint": str(output),
            "model_config": asdict(self.model.config),
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "input_normalization": (
                "per_dataset_q01_q99_to_minus1_plus1_except_gripper_then_"
                "reversible_effect_scale"
            ),
            "gripper_weight": float(self.gripper_weight),
            "effect_scale": list(self.effect_scale),
            "decoded_effect_prototypes": self.raw_centers.tolist(),
            "config": dict(self.config),
            "global_step": int(payload.get("global_step", 0)),
        }
        output.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EffectTokenizer":
        model_config = MLPVQVAEConfig.from_dict(payload["model_config"])
        model = MLPEffectVQVAE(model_config)
        model.load_state_dict(payload["model_state_dict"])
        return cls(
            model=model,
            gripper_weight=float(payload["gripper_weight"]),
            config=dict(payload["config"]),
            effect_scale=tuple(
                float(value)
                for value in payload.get(
                    "effect_scale", [1.0] * len(DESCRIPTOR_NAMES)
                )
            ),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "EffectTokenizer":
        return cls.from_payload(
            load_effect_checkpoint(path, map_location=map_location)
        )
