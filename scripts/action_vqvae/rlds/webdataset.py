"""Direct reader for the local ``jxu124/OpenX-Embodiment`` tar release.

Each tar member is a trusted Python pickle containing one episode with a
``steps`` list. The helpers here stream members without extraction, restore a
dense RLDS-style trajectory, and compute the action statistics consumed by the
vendored VQ-VLA/OXE standardizers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import random
import tarfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_OPENX_DIRECTORY_ALIASES = {
    "bridge_orig": "bridge",
    "fmb": "fmb_dataset",
}

logger = logging.getLogger(__name__)


def resolve_openx_tar_paths(root: str | Path, dataset_name: str) -> tuple[Path, ...]:
    """Return sorted tar shards for one OXE source."""
    root = Path(root).expanduser()
    candidates = [root / dataset_name]
    alias = _OPENX_DIRECTORY_ALIASES.get(dataset_name)
    if alias is not None:
        candidates.append(root / alias)
    for dataset_dir in candidates:
        paths = tuple(sorted(dataset_dir.glob("*.tar")))
        if paths:
            return paths
    return ()


def openx_tar_manifest(paths: Sequence[Path]) -> tuple[tuple[str, int, int], ...]:
    """Return shard metadata used to invalidate cached statistics."""
    return tuple(
        (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
        for path in paths
    )


def iter_openx_tar_episodes(
    paths: Sequence[Path],
    *,
    seed: int = 0,
    shuffle_shards: bool = False,
) -> Iterator[Mapping[str, Any]]:
    """Yield trusted local OpenX episode payloads without extracting them."""
    ordered_paths = list(paths)
    if shuffle_shards:
        random.Random(seed).shuffle(ordered_paths)
    for path in ordered_paths:
        try:
            with tarfile.open(path, mode="r|*") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".data.pickle"):
                        continue
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise RuntimeError(
                            f"Could not read {member.name!r} from OpenX shard {path}."
                        )
                    try:
                        # The release uses pickle by design. Only trusted local
                        # OpenX shards should be supplied as data_root_dir.
                        payload = pickle.loads(stream.read())  # noqa: S301  # nosec B301
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to deserialize {member.name!r} from OpenX shard {path}: {exc}"
                        ) from exc
                    if not isinstance(payload, Mapping) or not isinstance(
                        payload.get("steps"), list
                    ):
                        raise ValueError(
                            f"OpenX member {member.name!r} in {path} must contain a mapping with a 'steps' list."
                        )
                    if payload["steps"]:
                        yield payload
        except (tarfile.TarError, OSError) as exc:
            raise RuntimeError(
                f"Failed to open OpenX WebDataset shard {path}: {exc}"
            ) from exc


def _is_encoded_image(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and "bytes" in value
        and set(value).issubset({"bytes", "path"})
    )


def _encoded_image_bytes(value: Mapping[str, Any], path: str) -> bytes:
    data = value.get("bytes")
    if data is not None:
        return bytes(data)
    image_path = value.get("path")
    if image_path:
        return Path(image_path).read_bytes()
    raise ValueError(f"Encoded image at {path} has neither bytes nor a readable path.")


def _stack_step_values(values: Sequence[Any], path: str) -> Any:
    first = values[0]
    if all(_is_encoded_image(value) for value in values):
        return np.asarray(
            [_encoded_image_bytes(value, path) for value in values], dtype=object
        )
    if isinstance(first, Mapping):
        expected_keys = set(first)
        for index, value in enumerate(values[1:], start=1):
            if not isinstance(value, Mapping) or set(value) != expected_keys:
                raise ValueError(
                    f"OpenX episode field {path} changes mapping keys at step {index}."
                )
        result = {}
        for key in first:
            nested = [value[key] for value in values]
            if all(value is None for value in nested):
                continue
            if any(value is None for value in nested):
                raise ValueError(
                    f"OpenX episode field {path}/{key} is None only at some steps."
                )
            result[key] = _stack_step_values(nested, f"{path}/{key}")
        return result
    if isinstance(first, str):
        if not all(isinstance(value, str) for value in values):
            raise ValueError(
                f"OpenX episode field {path} mixes string and non-string values."
            )
        return np.asarray(values, dtype=object)
    if isinstance(first, (bytes, bytearray, memoryview)):
        if not all(
            isinstance(value, (bytes, bytearray, memoryview)) for value in values
        ):
            raise ValueError(
                f"OpenX episode field {path} mixes byte and non-byte values."
            )
        return np.asarray([bytes(value) for value in values], dtype=object)
    if first.__class__.__module__.startswith("PIL."):
        values = [np.asarray(value) for value in values]
    try:
        return np.stack([np.asarray(value) for value in values])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"OpenX episode field {path} cannot be stacked into a dense tensor."
        ) from exc


def stack_openx_episode_steps(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a pickled episode's step list to an RLDS-style trajectory."""
    steps = payload["steps"]
    if not steps or not all(isinstance(step, Mapping) for step in steps):
        raise ValueError("OpenX episode 'steps' must be a non-empty list of mappings.")
    expected_keys = set(steps[0])
    if any(set(step) != expected_keys for step in steps[1:]):
        raise ValueError("OpenX episode top-level step keys are inconsistent.")
    return {
        key: _stack_step_values([step[key] for step in steps], key)
        for key in steps[0]
        if not all(step[key] is None for step in steps)
    }


def _trajectory_to_tensors(trajectory: Mapping[str, Any], tf: Any) -> dict[str, Any]:
    def convert(value: Any):
        if isinstance(value, Mapping):
            return {key: convert(nested) for key, nested in value.items()}
        array = np.asarray(value)
        if array.dtype.kind in {"O", "S", "U"}:
            return tf.convert_to_tensor(array.tolist(), dtype=tf.string)
        return tf.convert_to_tensor(array)

    return {key: convert(value) for key, value in trajectory.items()}


def transform_openx_tar_episode(
    payload: Mapping[str, Any],
    *,
    tf: Any,
    transform: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Run one vendored OXE standardizer eagerly on a tar episode."""
    return transform(_trajectory_to_tensors(stack_openx_episode_steps(payload), tf))


def _cache_paths(
    paths: Sequence[Path], dependencies: Sequence[str]
) -> tuple[Path, Path]:
    digest = hashlib.sha256(
        "".join(dependencies).encode(), usedforsecurity=False
    ).hexdigest()
    filename = f"dataset_statistics_{digest}.json"
    return paths[
        0
    ].parent / filename, Path.home() / ".cache" / "myStudy" / "rlds" / filename


def load_or_compute_openx_action_statistics(
    *,
    paths: Sequence[Path],
    tf: Any,
    standardize_fn: Callable[[dict[str, Any]], Mapping[str, Any]],
    hash_dependencies: Sequence[str],
) -> dict[str, Any]:
    """Compute action mean/std/min/max/q01/q99 directly from tar episodes."""
    if not paths:
        raise ValueError(
            "At least one OpenX tar shard is required to compute statistics."
        )
    primary, fallback = _cache_paths(paths, hash_dependencies)
    for path in (primary, fallback):
        if path.is_file():
            print(f"[data] loading cached OpenX tar statistics: {path}", flush=True)
            with path.open(encoding="utf-8") as stream:
                return json.load(stream)

    print(
        f"[data] computing OpenX action statistics from {len(paths)} tar shard(s); no extraction",
        flush=True,
    )
    actions: list[np.ndarray] = []
    num_transitions = 0
    num_trajectories = 0
    for payload in iter_openx_tar_episodes(paths):
        trajectory = transform_openx_tar_episode(
            payload, tf=tf, transform=standardize_fn
        )
        action = np.asarray(trajectory["action"], dtype=np.float32)
        if action.ndim != 2:
            raise ValueError(
                f"OXE standardizer must produce rank-2 actions, got {action.shape}."
            )
        actions.append(action)
        num_transitions += action.shape[0]
        num_trajectories += 1
        if num_trajectories % 100 == 0:
            print(
                f"[data] statistics scan: {num_trajectories} episodes, {num_transitions} transitions",
                flush=True,
            )
    if not actions:
        raise ValueError("OpenX tar shards contain no non-empty episodes.")
    values = np.concatenate(actions, axis=0)
    metadata = {
        "action": {
            "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(),
            "max": values.max(0).tolist(),
            "min": values.min(0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        },
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }
    for path in (primary, fallback):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream)
            temporary.replace(path)
            print(f"[data] cached OpenX tar statistics: {path}", flush=True)
            return metadata
        except OSError as exc:
            logger.warning("Could not cache OpenX tar statistics at %s: %s", path, exc)
    raise OSError(f"Could not cache OpenX tar statistics at {primary} or {fallback}.")
