"""PyTorch wrapper around the VQ-VLA Open X-Embodiment RLDS pipeline.

The underlying RLDS implementation in :mod:`rlds` is vendored from VQ-VLA and
keeps its important data contract: every dataset is standardized to relative
EEF actions and future actions are chunked online.  Callers may either use the
original per-dataset q01/q99 scaling or only clip to those bounds before a
later pooled normalization stage.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

if __package__:
    from .rlds import make_interleaved_action_dataset
    from .rlds.frequency_resampling import (
        CONTROL_FREQUENCY_RESAMPLER_VERSION,
        resample_action_numpy,
    )
    from .rlds.oxe import (
        OXE_NAMED_MIXTURES,
        get_oxe_dataset_kwargs_and_weights,
    )
    from .rlds.oxe.configs import OXE_DATASET_CONFIGS
    from .rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
    from .rlds.utils.data_utils import NormalizationType
    from .rlds.webdataset import (
        iter_openx_tar_episodes,
        load_or_compute_openx_action_statistics,
        openx_tar_manifest,
        resolve_openx_tar_paths,
        transform_openx_tar_episode,
    )
else:  # Support direct execution of train_action_vqvae.py.
    from rlds import make_interleaved_action_dataset
    from rlds.frequency_resampling import (
        CONTROL_FREQUENCY_RESAMPLER_VERSION,
        resample_action_numpy,
    )
    from rlds.oxe import (
        OXE_NAMED_MIXTURES,
        get_oxe_dataset_kwargs_and_weights,
    )
    from rlds.oxe.configs import OXE_DATASET_CONFIGS
    from rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
    from rlds.utils.data_utils import NormalizationType
    from rlds.webdataset import (
        iter_openx_tar_episodes,
        load_or_compute_openx_action_statistics,
        openx_tar_manifest,
        resolve_openx_tar_paths,
        transform_openx_tar_episode,
    )


class OXEActionDataset(IterableDataset[tuple[Tensor]]):
    """Stream normalized action chunks from one OXE dataset or a named mix."""

    def __init__(
        self,
        data_root_dir: str | Path,
        data_mix: str,
        *,
        horizon: int,
        sampling_stride: int | None = None,
        target_control_hz: float | None = None,
        action_dim: int = 7,
        train: bool = True,
        shuffle_buffer_size: int = 200_000,
        sample_ratio: float = 1.0,
        balance_weights: bool = True,
        traj_transform_threads: int | None = None,
        traj_read_threads: int | None = None,
        storage_format: str = "auto",
        action_normalization: str = "bounds_q99",
        seed: int = 0,
    ) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if sampling_stride is None:
            sampling_stride = max(1, horizon // 4)
        if sampling_stride <= 0:
            raise ValueError(
                f"sampling_stride must be positive, got {sampling_stride}"
            )
        if target_control_hz is not None and target_control_hz <= 0:
            raise ValueError(
                f"target_control_hz must be positive or None, got {target_control_hz}"
            )
        if shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive")
        if not 0.0 < sample_ratio <= 1.0:
            raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
        if storage_format not in {"auto", "tfds", "webdataset", "hybrid"}:
            raise ValueError(
                "storage_format must be 'auto', 'tfds', 'webdataset', or 'hybrid', "
                f"got {storage_format!r}"
            )
        if action_normalization not in {"bounds_q99", "clip_q99"}:
            raise ValueError(
                "action_normalization must be 'bounds_q99' or 'clip_q99', "
                f"got {action_normalization!r}"
            )

        self.data_root_dir = Path(data_root_dir)
        self.data_mix = data_mix
        self.horizon = int(horizon)
        self.sampling_stride = int(sampling_stride)
        self.target_control_hz = (
            None if target_control_hz is None else float(target_control_hz)
        )
        self.action_dim = int(action_dim)
        self.train = bool(train)
        self.sample_ratio = float(sample_ratio)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.action_normalization = action_normalization
        self.seed = int(seed)

        if data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[data_mix]
        elif data_mix in OXE_DATASET_CONFIGS:
            mixture_spec = [(data_mix, 1.0)]
        else:
            known_mixes = ", ".join(sorted(OXE_NAMED_MIXTURES))
            raise ValueError(
                f"Unknown OXE dataset or mixture {data_mix!r}. "
                f"Known named mixtures: {known_mixes}"
            )

        # Action-only VQ-VAE training does not decode images, proprioception, or
        # language. Empty camera views avoids unnecessary observation plumbing.
        normalization_type = (
            NormalizationType.BOUNDS_Q99
            if self.action_normalization == "bounds_q99"
            else NormalizationType.BOUNDS_Q99_CLIP
        )
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=(),
            load_depth=False,
            load_proprio=False,
            load_language=False,
            action_proprio_normalization_type=normalization_type,
            target_control_hz=self.target_control_hz,
        )
        if self.action_normalization == "clip_q99":
            # The direct effect tokenizer clusters gripper changes as well as
            # translation and rotation, so clip every action dimension.  The
            # absolute mask still controls frequency resampling and tail fill.
            for dataset_kwargs in per_dataset_kwargs:
                dataset_kwargs["action_normalization_mask"] = [
                    True
                ] * len(dataset_kwargs["absolute_action_mask"])
        if not per_dataset_kwargs:
            raise ValueError(
                f"No supported EEF action datasets remain in mixture {data_mix!r}"
            )

        if self.target_control_hz is not None:
            print(
                f"[data] control-frequency plan: target={self.target_control_hz:g}Hz",
                flush=True,
            )
            for item in per_dataset_kwargs:
                source_hz = float(item["source_control_hz"])
                if np.isclose(source_hz, self.target_control_hz):
                    mode = "unchanged"
                elif source_hz > self.target_control_hz:
                    mode = "downsample"
                else:
                    mode = "upsample"
                print(
                    f"[data]   {item['name']}: {source_hz:g}Hz -> "
                    f"{self.target_control_hz:g}Hz ({mode})",
                    flush=True,
                )
        self.dataset_names = tuple(item["name"] for item in per_dataset_kwargs)
        self.source_control_frequencies = {
            item["name"]: float(item["source_control_hz"])
            for item in per_dataset_kwargs
            if item.get("source_control_hz") is not None
        }
        self.absolute_action_masks = {
            item["name"]: np.asarray(item["absolute_action_mask"], dtype=bool)
            for item in per_dataset_kwargs
            if item.get("absolute_action_mask") is not None
        }
        has_tar_shards = [
            bool(resolve_openx_tar_paths(self.data_root_dir, item["name"]))
            for item in per_dataset_kwargs
        ]
        if storage_format in {"auto", "hybrid"}:
            if all(has_tar_shards):
                self.storage_format = "webdataset"
            elif not any(has_tar_shards):
                self.storage_format = "tfds"
            else:
                self.storage_format = "hybrid"
        else:
            self.storage_format = storage_format
        if self.storage_format == "webdataset" and not all(has_tar_shards):
            missing = [
                item["name"]
                for item, has_tar in zip(
                    per_dataset_kwargs, has_tar_shards, strict=True
                )
                if not has_tar
            ]
            raise FileNotFoundError(
                f"storage_format='webdataset' but no tar shards were found for: {missing}"
            )

        self.dataset = None
        self._tfds_dataset = None
        self._webdataset_sources: list[dict[str, Any]] = []
        self._webdataset_sample_weights = np.empty(0, dtype=np.float64)

        tar_indices = [index for index, has_tar in enumerate(has_tar_shards) if has_tar]
        tfds_indices = [
            index for index, has_tar in enumerate(has_tar_shards) if not has_tar
        ]
        if self.storage_format == "webdataset":
            tar_indices = list(range(len(per_dataset_kwargs)))
            tfds_indices = []
        elif self.storage_format == "tfds":
            tar_indices = []
            tfds_indices = list(range(len(per_dataset_kwargs)))

        dataset_statistics: dict[str, dict[str, Any]] = {}
        dataset_sizes_by_name: dict[str, int] = {}
        if tar_indices:
            tar_kwargs = [per_dataset_kwargs[index] for index in tar_indices]
            tar_weights = [weights[index] for index in tar_indices]
            tar_statistics, tar_sizes = self._initialize_webdataset(
                tar_kwargs, tar_weights, balance_weights
            )
            dataset_statistics.update(tar_statistics)
            dataset_sizes_by_name.update(
                {
                    kwargs["name"]: size
                    for kwargs, size in zip(tar_kwargs, tar_sizes, strict=True)
                }
            )

        if tfds_indices:
            tfds_kwargs = [per_dataset_kwargs[index] for index in tfds_indices]
            tfds_weights = [weights[index] for index in tfds_indices]
            self._tfds_dataset, _, tfds_statistics = make_interleaved_action_dataset(
                dataset_kwargs_list=tfds_kwargs,
                sample_weights=tfds_weights,
                train=self.train,
                shuffle_buffer_size=shuffle_buffer_size,
                traj_transform_kwargs={
                    "window_size": 1,
                    "future_action_window_size": self.horizon - 1,
                    "sampling_stride": self.sampling_stride,
                    "skip_unlabeled": False,
                    "goal_relabeling_strategy": None,
                },
                balance_weights=balance_weights,
                traj_transform_threads=(
                    traj_transform_threads
                    if traj_transform_threads is not None
                    else len(tfds_kwargs)
                ),
                traj_read_threads=(
                    traj_read_threads
                    if traj_read_threads is not None
                    else len(tfds_kwargs)
                ),
                only_action=True,
                apply_shuffle=self.storage_format != "hybrid",
            )
            dataset_statistics.update(tfds_statistics)
            dataset_sizes_by_name.update(
                {
                    name: int(statistics["num_transitions"])
                    for name, statistics in tfds_statistics.items()
                }
            )
            if self.storage_format == "tfds":
                self.dataset = self._tfds_dataset

        self.dataset_statistics = dataset_statistics
        dataset_sizes = np.asarray(
            [dataset_sizes_by_name[name] for name in self.dataset_names],
            dtype=np.float64,
        )
        sampled_dataset_sizes = np.ceil(dataset_sizes / self.sampling_stride)
        base_weights = np.asarray(weights, dtype=np.float64)
        effective_weights = base_weights.copy()
        if balance_weights:
            effective_weights *= dataset_sizes
        effective_weights /= effective_weights.sum()
        self.sample_weights = effective_weights

        if tar_indices:
            tar_probabilities = effective_weights[tar_indices]
            self._webdataset_sample_weights = (
                tar_probabilities / tar_probabilities.sum()
            )
        if self.storage_format == "hybrid":
            self._hybrid_backend_weights = np.asarray(
                [
                    effective_weights[tar_indices].sum(),
                    effective_weights[tfds_indices].sum(),
                ],
                dtype=np.float64,
            )

        primary = np.flatnonzero(base_weights == 1.0)
        if primary.size == 0:
            primary = np.arange(len(base_weights))
        dataset_length = int(
            (sampled_dataset_sizes / effective_weights)[primary].max()
        )

        observed_dims = {
            int(np.asarray(statistics["action"]["mean"]).shape[-1])
            for statistics in self.dataset_statistics.values()
        }
        if observed_dims != {self.action_dim}:
            raise ValueError(
                f"OXE mixture {data_mix!r} produced action dimensions {sorted(observed_dims)}, "
                f"but --action-dim={self.action_dim}. Use a homogeneous EEF action mixture."
            )

        if self._tfds_dataset is not None and self.sample_ratio < 1.0:
            import tensorflow as tf

            ratio = tf.constant(self.sample_ratio, dtype=tf.float32)
            self._tfds_dataset = self._tfds_dataset.filter(
                lambda _: tf.random.uniform((), dtype=tf.float32) < ratio
            )
            if self.storage_format == "tfds":
                self.dataset = self._tfds_dataset
        if self.sample_ratio < 1.0:
            dataset_length = max(1, int(dataset_length * self.sample_ratio))

        self.dataset_length = int(dataset_length)

    def _initialize_webdataset(
        self,
        per_dataset_kwargs: list[dict[str, Any]],
        weights: list[float],
        balance_weights: bool,
    ) -> tuple[dict[str, dict[str, Any]], list[int]]:
        import tensorflow as tf

        statistics_by_name: dict[str, dict[str, Any]] = {}
        dataset_sizes = []
        for kwargs in per_dataset_kwargs:
            name = kwargs["name"]
            paths = resolve_openx_tar_paths(self.data_root_dir, name)
            standardize_fn = kwargs["standardize_fn"]
            source_control_hz = kwargs.get("source_control_hz")
            target_control_hz = kwargs.get("target_control_hz")
            absolute_action_mask = np.asarray(
                kwargs["absolute_action_mask"], dtype=bool
            )
            if name == "bridge_orig" and paths[0].parent.name == "bridge":
                # The jxu124/OpenX-Embodiment ``bridge`` tar release follows
                # the Open-X schema (nested action components), not the flat
                # action schema of the separately converted bridge_orig data.
                standardize_fn = OXE_STANDARDIZATION_TRANSFORMS["bridge_oxe"]
                print(
                    "[data] OpenX tar directory 'bridge' uses the bridge_oxe standardizer "
                    "for mixture source 'bridge_orig'.",
                    flush=True,
                )
            hash_dependencies = [
                name,
                repr(openx_tar_manifest(paths)),
                inspect.getsource(standardize_fn),
            ]
            if target_control_hz is not None:
                hash_dependencies.extend(
                    [
                        CONTROL_FREQUENCY_RESAMPLER_VERSION,
                        str(source_control_hz),
                        str(target_control_hz),
                    ]
                )
            statistics = load_or_compute_openx_action_statistics(
                paths=paths,
                tf=tf,
                standardize_fn=standardize_fn,
                hash_dependencies=tuple(hash_dependencies),
                action_transform=(
                    None
                    if target_control_hz is None
                    else lambda action,
                    absolute_action_mask=absolute_action_mask,
                    source_control_hz=source_control_hz,
                    target_control_hz=target_control_hz: resample_action_numpy(
                        action,
                        absolute_action_mask=absolute_action_mask,
                        source_hz=source_control_hz,
                        target_hz=target_control_hz,
                    )
                ),
            )
            statistics = {
                **statistics,
                "action": {
                    key: np.asarray(value)
                    for key, value in statistics["action"].items()
                },
            }
            normalization_mask = np.asarray(
                kwargs["action_normalization_mask"], dtype=bool
            )
            if normalization_mask.size != statistics["action"]["mean"].size:
                raise ValueError(
                    f"OXE source {name!r} action mask has {normalization_mask.size} dimensions, "
                    f"but its action statistics have {statistics['action']['mean'].size}."
                )
            statistics["action"]["mask"] = normalization_mask
            statistics_by_name[name] = statistics
            dataset_sizes.append(int(statistics["num_transitions"]))
            self._webdataset_sources.append(
                {
                    "name": name,
                    "paths": paths,
                    "standardize_fn": standardize_fn,
                    "absolute_action_mask": absolute_action_mask,
                    "source_control_hz": source_control_hz,
                    "target_control_hz": target_control_hz,
                    "statistics": statistics,
                }
            )

        base_weights = np.asarray(weights, dtype=np.float64)
        effective_weights = base_weights.copy()
        if balance_weights:
            effective_weights *= np.asarray(dataset_sizes, dtype=np.float64)
        effective_weights /= effective_weights.sum()
        self._webdataset_sample_weights = effective_weights
        return statistics_by_name, dataset_sizes

    def _iter_webdataset_source(
        self, source: dict[str, Any], source_index: int
    ) -> Iterator[np.ndarray]:
        import tensorflow as tf

        statistics = source["statistics"]["action"]
        low = np.asarray(statistics["q01"], dtype=np.float32)
        high = np.asarray(statistics["q99"], dtype=np.float32)
        minimum = np.asarray(statistics["min"], dtype=np.float32)
        maximum = np.asarray(statistics["max"], dtype=np.float32)
        normalization_mask = np.asarray(statistics["mask"], dtype=bool)
        absolute_action_mask = np.asarray(source["absolute_action_mask"], dtype=bool)
        num_trajectories = int(source["statistics"]["num_trajectories"])
        train_end = max(1, int(num_trajectories * 0.95)) if num_trajectories > 1 else 1
        epoch = 0
        while True:
            yielded = False
            frame_rng = np.random.default_rng(self.seed + 10_007 * source_index + epoch)
            for episode_index, payload in enumerate(
                iter_openx_tar_episodes(source["paths"])
            ):
                in_train_split = episode_index < train_end
                if in_train_split != self.train:
                    continue
                trajectory = transform_openx_tar_episode(
                    payload,
                    tf=tf,
                    transform=source["standardize_fn"],
                )
                action = np.asarray(trajectory["action"], dtype=np.float32)
                if source["target_control_hz"] is not None:
                    action = resample_action_numpy(
                        action,
                        absolute_action_mask=absolute_action_mask,
                        source_hz=source["source_control_hz"],
                        target_hz=source["target_control_hz"],
                    )
                if self.action_normalization == "clip_q99":
                    normalized = np.where(
                        normalization_mask,
                        np.clip(action, low, high),
                        action,
                    ).astype(np.float32)
                else:
                    normalized = np.clip(
                        2.0 * (action - low) / (high - low + 1e-8) - 1.0,
                        -1.0,
                        1.0,
                    )
                    normalized = np.where(normalization_mask, normalized, action)
                    normalized = np.where(
                        minimum == maximum, 0.0, normalized
                    ).astype(np.float32)
                frame_indices = np.arange(
                    0, normalized.shape[0], self.sampling_stride
                )
                if self.train:
                    frame_rng.shuffle(frame_indices)
                for frame_index in frame_indices:
                    chunk_indices = np.arange(frame_index, frame_index + self.horizon)
                    past_end = chunk_indices >= normalized.shape[0]
                    chunk = normalized[
                        np.minimum(chunk_indices, normalized.shape[0] - 1)
                    ].copy()
                    if np.any(past_end):
                        chunk[past_end] = np.where(
                            absolute_action_mask,
                            chunk[past_end],
                            np.zeros_like(chunk[past_end]),
                        )
                    yielded = True
                    yield chunk
            if not yielded:
                split = "train" if self.train else "validation"
                raise ValueError(
                    f"OpenX tar source {source['name']!r} has no episodes in its {split} split."
                )
            epoch += 1

    def _iter_weighted_webdataset_chunks(self) -> Iterator[np.ndarray]:
        iterators = [
            iter(self._iter_webdataset_source(source, index))
            for index, source in enumerate(self._webdataset_sources)
        ]
        rng = np.random.default_rng(self.seed)
        while True:
            source_index = int(
                rng.choice(len(iterators), p=self._webdataset_sample_weights)
            )
            chunk = next(iterators[source_index])
            if self.sample_ratio >= 1.0 or rng.random() < self.sample_ratio:
                yield chunk

    def _iter_tfds_chunks(self, *, repeat: bool = False) -> Iterator[np.ndarray]:
        if self._tfds_dataset is None:
            raise RuntimeError("The TFDS stream has not been initialized.")
        while True:
            yielded = False
            for batch in self._tfds_dataset.as_numpy_iterator():
                yielded = True
                yield np.asarray(batch["action"], dtype=np.float32)
            if not repeat:
                return
            if not yielded:
                raise ValueError("The TFDS/RLDS stream is empty.")

    def _shuffle_chunk_stream(
        self, stream: Iterator[np.ndarray], *, description: str
    ) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self.seed + 91_003)
        if not self.train:
            # Match the existing DLimp validation path: repeat the held-out
            # stream, take one fixed buffer, cache it, then shuffle it.
            cached = [next(stream) for _ in range(self.shuffle_buffer_size)]
            rng.shuffle(cached)
            yield from cached
            return

        print(
            f"[data] filling {description} action shuffle buffer: "
            f"{self.shuffle_buffer_size} chunks",
            flush=True,
        )
        buffer = []
        while len(buffer) < self.shuffle_buffer_size:
            buffer.append(next(stream))
            if len(buffer) % 10_000 == 0:
                print(
                    f"[data] {description} action shuffle buffer: "
                    f"{len(buffer)}/{self.shuffle_buffer_size}",
                    flush=True,
                )
        print(f"[data] {description} action shuffle buffer is ready", flush=True)
        while True:
            index = int(rng.integers(len(buffer)))
            chunk = buffer[index]
            buffer[index] = next(stream)
            yield chunk

    def _iter_webdataset_chunks(self) -> Iterator[np.ndarray]:
        yield from self._shuffle_chunk_stream(
            iter(self._iter_weighted_webdataset_chunks()),
            description="OpenX tar",
        )

    def _iter_hybrid_chunks(self) -> Iterator[np.ndarray]:
        backend_iterators = [
            iter(self._iter_weighted_webdataset_chunks()),
            iter(self._iter_tfds_chunks(repeat=True)),
        ]
        rng = np.random.default_rng(self.seed + 47_021)

        def weighted_stream() -> Iterator[np.ndarray]:
            while True:
                backend_index = int(
                    rng.choice(len(backend_iterators), p=self._hybrid_backend_weights)
                )
                yield next(backend_iterators[backend_index])

        yield from self._shuffle_chunk_stream(
            iter(weighted_stream()),
            description="hybrid OXE",
        )

    def __iter__(self) -> Iterator[tuple[Tensor]]:
        worker = get_worker_info()
        if worker is not None:
            raise RuntimeError(
                "OXEActionDataset must use DataLoader(num_workers=0); its backend manages streaming "
                "and distributed sharding."
            )

        if dist.is_available() and dist.is_initialized():
            rank, world_size = dist.get_rank(), dist.get_world_size()
        else:
            rank, world_size = 0, 1

        if self.storage_format == "webdataset":
            stream = self._iter_webdataset_chunks()
        elif self.storage_format == "hybrid":
            stream = self._iter_hybrid_chunks()
        else:
            stream = self._iter_tfds_chunks()
        for index, actions in enumerate(stream):
            if index % world_size != rank:
                continue
            actions = np.asarray(actions, dtype=np.float32)
            if actions.shape != (self.horizon, self.action_dim):
                raise ValueError(
                    f"Expected OXE action chunk shape {(self.horizon, self.action_dim)}, got {actions.shape}"
                )
            yield (torch.from_numpy(actions.copy()),)

    def __len__(self) -> int:
        return self.dataset_length

    def summary(self) -> str:
        datasets = ",".join(self.dataset_names)
        split = "train" if self.train else "validation"
        return (
            f"rlds:{self.data_mix} storage={self.storage_format} split={split} datasets=[{datasets}] "
            f"estimated_samples={self.dataset_length} horizon={self.horizon} "
            f"target_control_hz={self.target_control_hz or 'native'} "
            f"sampling_stride={self.sampling_stride} action_dim={self.action_dim}"
            f" action_normalization={self.action_normalization}"
        )

    @property
    def is_rlds(self) -> bool:
        return True

    def statistics_for_json(self) -> dict[str, Any]:
        """Return JSON-safe action statistics for reproducibility metadata."""
        return {
            dataset_name: {
                "action": {
                    key: np.asarray(value).tolist()
                    for key, value in statistics["action"].items()
                },
                "num_transitions": int(statistics["num_transitions"]),
                "num_trajectories": int(statistics["num_trajectories"]),
                "source_control_hz": float(
                    self.source_control_frequencies[dataset_name]
                )
                if dataset_name in self.source_control_frequencies
                else None,
                "target_control_hz": self.target_control_hz,
                "absolute_action_mask": self.absolute_action_masks[
                    dataset_name
                ].tolist()
                if dataset_name in self.absolute_action_masks
                else None,
            }
            for dataset_name, statistics in self.dataset_statistics.items()
        }
