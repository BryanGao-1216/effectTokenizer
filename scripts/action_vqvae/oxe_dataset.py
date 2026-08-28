"""PyTorch wrapper around the VQ-VLA Open X-Embodiment RLDS pipeline.

The underlying RLDS implementation in :mod:`rlds` is vendored from VQ-VLA and
keeps its important data contract: every dataset is standardized to relative
EEF actions, dimensions other than the gripper are normalized with per-dataset
1st/99th percentiles, and future actions are chunked online.
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
    from .action_window import (
        extract_action_window,
        frames_for_duration,
        frames_for_stride,
    )
    from .rlds import make_interleaved_action_dataset
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
    from action_window import (
        extract_action_window,
        frames_for_duration,
        frames_for_stride,
    )
    from rlds import make_interleaved_action_dataset
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


class OXEActionDataset(IterableDataset[tuple[Tensor, Tensor, Tensor]]):
    """Stream native-rate, fixed-duration normalized OXE action chunks."""

    def __init__(
        self,
        data_root_dir: str | Path,
        data_mix: str,
        *,
        window_duration_seconds: float,
        sampling_stride_seconds: float | None = None,
        pad_incomplete_windows: bool = True,
        action_dim: int = 7,
        train: bool = True,
        shuffle_buffer_size: int = 200_000,
        sample_ratio: float = 1.0,
        balance_weights: bool = True,
        traj_transform_threads: int | None = None,
        traj_read_threads: int | None = None,
        storage_format: str = "auto",
        seed: int = 0,
    ) -> None:
        super().__init__()
        if window_duration_seconds <= 0:
            raise ValueError(
                "window_duration_seconds must be positive, got "
                f"{window_duration_seconds}"
            )
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if sampling_stride_seconds is None:
            sampling_stride_seconds = window_duration_seconds / 4.0
        if sampling_stride_seconds <= 0:
            raise ValueError(
                "sampling_stride_seconds must be positive, got "
                f"{sampling_stride_seconds}"
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

        self.data_root_dir = Path(data_root_dir)
        self.data_mix = data_mix
        self.window_duration_seconds = float(window_duration_seconds)
        self.sampling_stride_seconds = float(sampling_stride_seconds)
        self.pad_incomplete_windows = bool(pad_incomplete_windows)
        self.action_dim = int(action_dim)
        self.train = bool(train)
        self.sample_ratio = float(sample_ratio)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
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
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=(),
            load_depth=False,
            load_proprio=False,
            load_language=False,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        if not per_dataset_kwargs:
            raise ValueError(
                f"No supported EEF action datasets remain in mixture {data_mix!r}"
            )

        self.window_frames_by_name = {
            item["name"]: frames_for_duration(
                self.window_duration_seconds,
                float(item["source_control_hz"]),
            )
            for item in per_dataset_kwargs
        }
        self.stride_frames_by_name = {
            item["name"]: frames_for_stride(
                self.sampling_stride_seconds,
                float(item["source_control_hz"]),
            )
            for item in per_dataset_kwargs
        }
        self.output_action_window_size = max(self.window_frames_by_name.values())
        print(
            "[data] native-rate time-window plan: "
            f"duration={self.window_duration_seconds:g}s "
            f"stride={self.sampling_stride_seconds:g}s "
            f"batch_frames={self.output_action_window_size}",
            flush=True,
        )
        for item in per_dataset_kwargs:
            name = item["name"]
            source_hz = float(item["source_control_hz"])
            print(
                f"[data]   {name}: {source_hz:g}Hz, "
                f"window={self.window_frames_by_name[name]} frames, "
                f"stride={self.stride_frames_by_name[name]} frames",
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
                    "pad_incomplete_action_windows": self.pad_incomplete_windows,
                    "skip_unlabeled": False,
                    "goal_relabeling_strategy": None,
                },
                per_dataset_traj_transform_kwargs=[
                    {
                        "future_action_window_size": self.window_frames_by_name[
                            item["name"]
                        ]
                        - 1,
                        "sampling_stride": self.stride_frames_by_name[
                            item["name"]
                        ],
                        "output_action_window_size": self.output_action_window_size,
                    }
                    for item in tfds_kwargs
                ],
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
        stride_frames = np.asarray(
            [self.stride_frames_by_name[name] for name in self.dataset_names],
            dtype=np.float64,
        )
        sampled_dataset_sizes = np.ceil(dataset_sizes / stride_frames)
        base_weights = np.asarray(weights, dtype=np.float64)
        effective_weights = base_weights.copy()
        if balance_weights:
            # Balance by the number of fixed-duration window starts rather
            # than native transitions. Otherwise a 20 Hz source would receive
            # roughly twice the weight of a 10 Hz source solely for having
            # twice as many frames per second.
            effective_weights *= sampled_dataset_sizes
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
            source_control_hz = float(kwargs["source_control_hz"])
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
            statistics = load_or_compute_openx_action_statistics(
                paths=paths,
                tf=tf,
                standardize_fn=standardize_fn,
                hash_dependencies=tuple(hash_dependencies),
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
                    "window_frames": self.window_frames_by_name[name],
                    "stride_frames": self.stride_frames_by_name[name],
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
    ) -> Iterator[tuple[np.ndarray, int, float]]:
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
                normalized = np.clip(
                    2.0 * (action - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0
                )
                normalized = np.where(normalization_mask, normalized, action)
                normalized = np.where(minimum == maximum, 0.0, normalized).astype(
                    np.float32
                )
                frame_stop = (
                    normalized.shape[0]
                    if self.pad_incomplete_windows
                    else max(
                        normalized.shape[0] - source["window_frames"] + 1,
                        0,
                    )
                )
                frame_indices = np.arange(
                    0,
                    frame_stop,
                    source["stride_frames"],
                )
                if self.train:
                    frame_rng.shuffle(frame_indices)
                for frame_index in frame_indices:
                    chunk = extract_action_window(
                        normalized,
                        start=int(frame_index),
                        horizon=source["window_frames"],
                        absolute_action_mask=absolute_action_mask,
                        pad_incomplete=self.pad_incomplete_windows,
                    )
                    if chunk is None:
                        continue
                    if source["window_frames"] < self.output_action_window_size:
                        chunk = extract_action_window(
                            chunk,
                            start=0,
                            horizon=self.output_action_window_size,
                            absolute_action_mask=absolute_action_mask,
                            pad_incomplete=True,
                        )
                        if chunk is None:  # pragma: no cover - defensive contract
                            raise RuntimeError("Failed to add batch-only action padding.")
                    yielded = True
                    yield (
                        chunk,
                        int(source["window_frames"]),
                        float(source["source_control_hz"]),
                    )
            if not yielded:
                split = "train" if self.train else "validation"
                raise ValueError(
                    f"OpenX tar source {source['name']!r} has no episodes in its {split} split."
                )
            epoch += 1

    def _iter_weighted_webdataset_chunks(
        self,
    ) -> Iterator[tuple[np.ndarray, int, float]]:
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

    def _iter_tfds_chunks(
        self, *, repeat: bool = False
    ) -> Iterator[tuple[np.ndarray, int, float]]:
        if self._tfds_dataset is None:
            raise RuntimeError("The TFDS stream has not been initialized.")
        while True:
            yielded = False
            for batch in self._tfds_dataset.as_numpy_iterator():
                yielded = True
                yield (
                    np.asarray(batch["action"], dtype=np.float32),
                    int(np.asarray(batch["action_window_length"]).item()),
                    float(np.asarray(batch["source_control_hz"]).item()),
                )
            if not repeat:
                return
            if not yielded:
                raise ValueError("The TFDS/RLDS stream is empty.")

    def _shuffle_chunk_stream(
        self,
        stream: Iterator[tuple[np.ndarray, int, float]],
        *,
        description: str,
    ) -> Iterator[tuple[np.ndarray, int, float]]:
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

    def _iter_webdataset_chunks(
        self,
    ) -> Iterator[tuple[np.ndarray, int, float]]:
        yield from self._shuffle_chunk_stream(
            iter(self._iter_weighted_webdataset_chunks()),
            description="OpenX tar",
        )

    def _iter_hybrid_chunks(
        self,
    ) -> Iterator[tuple[np.ndarray, int, float]]:
        backend_iterators = [
            iter(self._iter_weighted_webdataset_chunks()),
            iter(self._iter_tfds_chunks(repeat=True)),
        ]
        rng = np.random.default_rng(self.seed + 47_021)

        def weighted_stream() -> Iterator[tuple[np.ndarray, int, float]]:
            while True:
                backend_index = int(
                    rng.choice(len(backend_iterators), p=self._hybrid_backend_weights)
                )
                yield next(backend_iterators[backend_index])

        yield from self._shuffle_chunk_stream(
            iter(weighted_stream()),
            description="hybrid OXE",
        )

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
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
        for index, (actions, action_window_length, source_control_hz) in enumerate(
            stream
        ):
            if index % world_size != rank:
                continue
            actions = np.asarray(actions, dtype=np.float32)
            if actions.shape != (self.output_action_window_size, self.action_dim):
                raise ValueError(
                    "Expected OXE action chunk shape "
                    f"{(self.output_action_window_size, self.action_dim)}, "
                    f"got {actions.shape}"
                )
            if not 0 < action_window_length <= self.output_action_window_size:
                raise ValueError(
                    f"Invalid native action window length {action_window_length}."
                )
            yield (
                torch.from_numpy(actions.copy()),
                torch.tensor(action_window_length, dtype=torch.int64),
                torch.tensor(source_control_hz, dtype=torch.float32),
            )

    def __len__(self) -> int:
        return self.dataset_length

    def summary(self) -> str:
        datasets = ",".join(self.dataset_names)
        split = "train" if self.train else "validation"
        return (
            f"rlds:{self.data_mix} storage={self.storage_format} split={split} datasets=[{datasets}] "
            f"estimated_samples={self.dataset_length} "
            f"window_duration_seconds={self.window_duration_seconds:g} "
            f"sampling_stride_seconds={self.sampling_stride_seconds:g} "
            f"batch_frames={self.output_action_window_size} "
            f"pad_incomplete_windows={self.pad_incomplete_windows} "
            f"action_dim={self.action_dim}"
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
                "window_duration_seconds": self.window_duration_seconds,
                "window_frames": self.window_frames_by_name[dataset_name],
                "sampling_stride_seconds": self.sampling_stride_seconds,
                "sampling_stride_frames": self.stride_frames_by_name[
                    dataset_name
                ],
                "absolute_action_mask": self.absolute_action_masks[
                    dataset_name
                ].tolist()
                if dataset_name in self.absolute_action_masks
                else None,
            }
            for dataset_name, statistics in self.dataset_statistics.items()
        }
