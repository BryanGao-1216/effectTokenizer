"""Core endpoint-effect descriptor and exact Lloyd K-means implementation."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


ARTIFACT_VERSION = 1
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
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_effect_descriptors(actions: np.ndarray) -> np.ndarray:
    """Map ``[..., horizon, 7]`` clipped actions to signed 7-D effects.

    Translation and rotation are accumulated over the chunk.  Rotation is a
    first-order sum of the OpenX-standardized small RPY deltas.  Gripper is
    represented by the final-minus-initial absolute gripper command.
    """
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim < 2 or values.shape[-1] != 7:
        raise ValueError(
            f"Expected action chunks shaped [..., horizon, 7], got {values.shape}."
        )
    if values.shape[-2] <= 0:
        raise ValueError("Action chunks must contain at least one timestep.")
    position = values[..., :, :3].sum(axis=-2, dtype=np.float64)
    rotation = values[..., :, 3:6].sum(axis=-2, dtype=np.float64)
    gripper = values[..., -1, 6] - values[..., 0, 6]
    return np.concatenate(
        [position, rotation, gripper[..., None]], axis=-1
    ).astype(np.float32)


def fit_global_standardizer(
    descriptors: np.ndarray,
    *,
    minimum_scale: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one pooled z-score transform after per-source q01/q99 clipping."""
    values = np.asarray(descriptors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(DESCRIPTOR_NAMES):
        raise ValueError(
            f"Expected descriptors shaped [N, {len(DESCRIPTOR_NAMES)}], got {values.shape}."
        )
    if len(values) == 0:
        raise ValueError("Cannot fit global normalization on an empty array.")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.maximum(scale, float(minimum_scale))
    return mean.astype(np.float32), scale.astype(np.float32)


def _standardize(
    descriptors: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    gripper_weight: float,
) -> np.ndarray:
    values = (
        np.asarray(descriptors, dtype=np.float32)
        - np.asarray(mean, dtype=np.float32)
    ) / np.asarray(scale, dtype=np.float32)
    values = values.copy()
    values[..., -1] *= float(gripper_weight)
    return values


def _inverse_standardize(
    descriptors: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    gripper_weight: float,
) -> np.ndarray:
    values = np.asarray(descriptors, dtype=np.float32).copy()
    values[..., -1] /= float(gripper_weight)
    return values * np.asarray(scale, dtype=np.float32) + np.asarray(
        mean, dtype=np.float32
    )


def _squared_distance(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (
        values.square().sum(dim=-1, keepdim=True)
        + centers.square().sum(dim=-1).unsqueeze(0)
        - 2.0 * values @ centers.t()
    ).clamp_min_(0.0)


def _kmeans_plus_plus(
    values: torch.Tensor,
    num_clusters: int,
    *,
    seed: int,
    candidate_samples: int,
) -> torch.Tensor:
    """K-means++ initialization on all data or an optional candidate subset."""
    generator = torch.Generator(device=values.device)
    generator.manual_seed(int(seed))
    candidate_count = (
        len(values)
        if candidate_samples <= 0
        else min(len(values), max(num_clusters, candidate_samples))
    )
    if candidate_count < len(values):
        indices = torch.randperm(
            len(values), generator=generator, device=values.device
        )[:candidate_count]
        candidates = values[indices]
    else:
        candidates = values

    first = int(
        torch.randint(
            len(candidates), (1,), generator=generator, device=values.device
        ).item()
    )
    centers = [candidates[first].clone()]
    nearest = _squared_distance(candidates, centers[0][None]).squeeze(1)
    for _ in range(1, num_clusters):
        total = nearest.sum()
        if not torch.isfinite(total) or float(total) <= 0.0:
            selected = int(
                torch.randint(
                    len(candidates),
                    (1,),
                    generator=generator,
                    device=values.device,
                ).item()
            )
        else:
            selected = int(
                torch.multinomial(
                    nearest / total,
                    1,
                    generator=generator,
                ).item()
            )
        center = candidates[selected].clone()
        centers.append(center)
        nearest = torch.minimum(
            nearest,
            _squared_distance(candidates, center[None]).squeeze(1),
        )
    return torch.stack(centers)


def _lloyd_once(
    values: torch.Tensor,
    *,
    num_clusters: int,
    max_iterations: int,
    tolerance: float,
    assignment_batch_size: int,
    seed: int,
    init_candidate_samples: int,
    progress: Callable[[int, float, float, int], None] | None,
) -> tuple[torch.Tensor, float, int]:
    centers = _kmeans_plus_plus(
        values,
        num_clusters,
        seed=seed,
        candidate_samples=init_candidate_samples,
    )
    generator = torch.Generator(device=values.device)
    generator.manual_seed(seed + 1_000_003)
    previous_inertia = math.inf
    final_iteration = 0
    for iteration in range(1, max_iterations + 1):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(
            num_clusters, dtype=torch.float32, device=values.device
        )
        inertia = 0.0
        for start in range(0, len(values), assignment_batch_size):
            batch = values[start : start + assignment_batch_size]
            distances = _squared_distance(batch, centers)
            nearest_distance, assignments = distances.min(dim=1)
            sums.index_add_(0, assignments, batch)
            counts += torch.bincount(
                assignments, minlength=num_clusters
            ).to(dtype=counts.dtype)
            inertia += float(nearest_distance.sum().item())

        updated = centers.clone()
        nonempty = counts > 0
        updated[nonempty] = sums[nonempty] / counts[nonempty, None]
        empty_count = int((~nonempty).sum().item())
        if empty_count:
            replacements = torch.randint(
                len(values),
                (empty_count,),
                generator=generator,
                device=values.device,
            )
            updated[~nonempty] = values[replacements]

        shift = float(torch.linalg.vector_norm(updated - centers, dim=1).max())
        relative_improvement = (
            math.inf
            if not math.isfinite(previous_inertia)
            else (previous_inertia - inertia) / max(previous_inertia, 1e-12)
        )
        centers = updated
        final_iteration = iteration
        if progress is not None:
            progress(iteration, inertia / len(values), shift, empty_count)
        if shift <= tolerance or (
            math.isfinite(relative_improvement)
            and 0.0 <= relative_improvement <= tolerance
        ):
            break
        previous_inertia = inertia

    final_inertia = 0.0
    for start in range(0, len(values), assignment_batch_size):
        distances = _squared_distance(
            values[start : start + assignment_batch_size], centers
        )
        final_inertia += float(distances.min(dim=1).values.sum().item())
    return centers, final_inertia / len(values), final_iteration


def fit_full_kmeans(
    standardized_descriptors: np.ndarray,
    *,
    num_clusters: int,
    max_iterations: int = 50,
    tolerance: float = 1e-4,
    n_init: int = 1,
    assignment_batch_size: int = 65_536,
    init_candidate_samples: int = 0,
    seed: int = 0,
    device: str | torch.device = "cpu",
    progress: Callable[[int, int, float, float, int], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit regular Lloyd K-means, using every sample on every iteration.

    ``assignment_batch_size`` only bounds the temporary distance matrix.  It
    does not change the objective or update centers from subsets, so this is not
    MiniBatch K-means.
    """
    values_np = np.asarray(standardized_descriptors, dtype=np.float32)
    if values_np.ndim != 2 or values_np.shape[1] != len(DESCRIPTOR_NAMES):
        raise ValueError(
            f"Expected standardized descriptors [N, {len(DESCRIPTOR_NAMES)}], got {values_np.shape}."
        )
    if not 1 < num_clusters <= len(values_np):
        raise ValueError(
            f"num_clusters must be in [2, {len(values_np)}], got {num_clusters}."
        )
    if min(max_iterations, n_init, assignment_batch_size) <= 0:
        raise ValueError("K-means iteration, initialization, and batch values must be positive.")
    if init_candidate_samples < 0:
        raise ValueError("init_candidate_samples must be non-negative; zero means all samples.")
    torch_device = torch.device(device)
    values = torch.from_numpy(values_np).to(torch_device)
    best_centers = None
    best_inertia = math.inf
    runs: list[dict[str, Any]] = []
    for run in range(n_init):
        callback = None
        if progress is not None:
            callback = lambda iteration, inertia, shift, empty, run=run: progress(
                run + 1, iteration, inertia, shift, empty
            )
        centers, inertia, iterations = _lloyd_once(
            values,
            num_clusters=num_clusters,
            max_iterations=max_iterations,
            tolerance=tolerance,
            assignment_batch_size=assignment_batch_size,
            seed=seed + 104_729 * run,
            init_candidate_samples=init_candidate_samples,
            progress=callback,
        )
        runs.append(
            {"run": run + 1, "iterations": iterations, "inertia": inertia}
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.detach().cpu().numpy().copy()
    assert best_centers is not None
    return best_centers.astype(np.float32), {
        "inertia_per_sample": float(best_inertia),
        "runs": runs,
    }


@dataclass
class EffectTokenizer:
    """Frozen global standardizer plus direct K-means centers."""

    centers: np.ndarray
    global_mean: np.ndarray
    global_scale: np.ndarray
    gripper_weight: float
    config: dict[str, Any]

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=np.float32)
        self.global_mean = np.asarray(self.global_mean, dtype=np.float32)
        self.global_scale = np.asarray(self.global_scale, dtype=np.float32)
        if self.centers.ndim != 2 or self.centers.shape[1] != len(DESCRIPTOR_NAMES):
            raise ValueError(
                f"Expected centers [K, {len(DESCRIPTOR_NAMES)}], got {self.centers.shape}."
            )
        expected_shape = (self.centers.shape[1],)
        if self.global_mean.shape != expected_shape:
            raise ValueError(
                f"Expected global_mean shape {expected_shape}, got {self.global_mean.shape}."
            )
        if self.global_scale.shape != expected_shape:
            raise ValueError(
                f"Expected global_scale shape {expected_shape}, got {self.global_scale.shape}."
            )
        if np.any(self.global_scale <= 0):
            raise ValueError("global_scale must be strictly positive.")
        if self.gripper_weight <= 0:
            raise ValueError("gripper_weight must be strictly positive.")

    @property
    def codebook_size(self) -> int:
        return int(self.centers.shape[0])

    @property
    def descriptor_dim(self) -> int:
        return int(self.centers.shape[1])

    @property
    def raw_centers(self) -> np.ndarray:
        return _inverse_standardize(
            self.centers,
            self.global_mean,
            self.global_scale,
            self.gripper_weight,
        )

    def standardize(self, descriptors: np.ndarray) -> np.ndarray:
        return _standardize(
            descriptors,
            self.global_mean,
            self.global_scale,
            self.gripper_weight,
        )

    def assign(
        self,
        descriptors: np.ndarray,
        *,
        batch_size: int = 65_536,
        device: str | torch.device = "cpu",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return labels, nearest squared distance, and relative top-2 margin."""
        standardized = self.standardize(descriptors)
        if standardized.ndim != 2 or standardized.shape[1] != self.descriptor_dim:
            raise ValueError(
                f"Expected descriptors [N, {self.descriptor_dim}], got {standardized.shape}."
            )
        if len(standardized) == 0:
            empty_labels = np.empty(0, dtype=np.int64)
            empty_values = np.empty(0, dtype=np.float32)
            return empty_labels, empty_values, empty_values.copy()
        values = torch.from_numpy(standardized).to(device)
        centers = torch.from_numpy(np.asarray(self.centers, dtype=np.float32)).to(
            device
        )
        labels: list[np.ndarray] = []
        nearest_distances: list[np.ndarray] = []
        margins: list[np.ndarray] = []
        for start in range(0, len(values), batch_size):
            distances = _squared_distance(values[start : start + batch_size], centers)
            nearest, indices = torch.topk(
                distances, k=min(2, self.codebook_size), largest=False, dim=1
            )
            labels.append(indices[:, 0].cpu().numpy())
            nearest_distances.append(nearest[:, 0].cpu().numpy())
            if nearest.shape[1] == 1:
                margins.append(np.ones(len(nearest), dtype=np.float32))
            else:
                margins.append(
                    (
                        (nearest[:, 1] - nearest[:, 0])
                        / nearest[:, 1].clamp_min(1e-8)
                    )
                    .cpu()
                    .numpy()
                )
        return (
            np.concatenate(labels).astype(np.int64),
            np.concatenate(nearest_distances).astype(np.float32),
            np.concatenate(margins).astype(np.float32),
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_version": ARTIFACT_VERSION,
            "centers": torch.from_numpy(np.asarray(self.centers, dtype=np.float32)),
            "global_mean": torch.from_numpy(
                np.asarray(self.global_mean, dtype=np.float32)
            ),
            "global_scale": torch.from_numpy(
                np.asarray(self.global_scale, dtype=np.float32)
            ),
            "gripper_weight": float(self.gripper_weight),
            "config": dict(self.config),
            "descriptor_names": list(DESCRIPTOR_NAMES),
        }
        torch.save(payload, output)
        metadata = {
            "artifact_version": ARTIFACT_VERSION,
            "checkpoint": str(output),
            "codebook_size": self.codebook_size,
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "global_normalization": "pooled_zscore_after_per_dataset_q01_q99_clip",
            "global_mean": np.asarray(self.global_mean).tolist(),
            "global_scale": np.asarray(self.global_scale).tolist(),
            "gripper_weight": float(self.gripper_weight),
            "raw_centers": self.raw_centers.tolist(),
            "config": dict(self.config),
        }
        output.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "EffectTokenizer":
        try:
            payload = torch.load(
                path, map_location=map_location, weights_only=False
            )
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        version = int(payload.get("artifact_version", 0))
        if version != ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported effect tokenizer artifact version {version}; expected {ARTIFACT_VERSION}."
            )
        return cls(
            centers=payload["centers"].detach().cpu().numpy().astype(np.float32),
            global_mean=payload["global_mean"].detach().cpu().numpy().astype(np.float32),
            global_scale=payload["global_scale"].detach().cpu().numpy().astype(np.float32),
            gripper_weight=float(payload["gripper_weight"]),
            config=dict(payload["config"]),
        )


def standardize_for_fit(
    descriptors: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    gripper_weight: float,
) -> np.ndarray:
    return _standardize(descriptors, mean, scale, gripper_weight)
