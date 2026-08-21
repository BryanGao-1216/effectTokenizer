"""Fit a direct K-means tokenizer on pooled OpenX endpoint effects."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.effect_tokenizer.effect_tokenizer import (  # noqa: E402
    DESCRIPTOR_NAMES,
    EffectTokenizer,
    choose_device,
    compute_effect_descriptors,
    fit_full_kmeans,
    fit_global_standardizer,
    set_seed,
    standardize_for_fit,
)


def _log(message: str) -> None:
    print(f"[effect] {message}", flush=True)


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _collect_descriptors(
    loader: DataLoader,
    *,
    num_samples: int,
    log_every_batches: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    collected = 0
    started = time.perf_counter()
    for batch_index, (actions,) in enumerate(loader, start=1):
        remaining = num_samples - collected
        if remaining <= 0:
            break
        action_array = actions.numpy().astype(np.float32, copy=False)[:remaining]
        descriptor = compute_effect_descriptors(action_array)
        chunks.append(descriptor)
        collected += len(descriptor)
        if (
            batch_index == 1
            or batch_index % log_every_batches == 0
            or collected >= num_samples
        ):
            elapsed = time.perf_counter() - started
            rate = collected / max(elapsed, 1e-8)
            eta = (num_samples - collected) / max(rate, 1e-8)
            _log(
                f"collect descriptors: {collected}/{num_samples} "
                f"({100.0 * collected / num_samples:.1f}%) "
                f"speed={rate:.0f} sample/s elapsed={_duration(elapsed)} "
                f"eta={_duration(eta)}"
            )
        if collected >= num_samples:
            break
    if collected < num_samples:
        raise ValueError(
            f"Training stream returned {collected} samples, fewer than {num_samples}."
        )
    return np.concatenate(chunks, axis=0)


def _usage_metrics(labels: np.ndarray, codebook_size: int) -> dict[str, Any]:
    counts = np.bincount(labels, minlength=codebook_size)
    probabilities = counts / max(counts.sum(), 1)
    nonzero = probabilities > 0
    entropy = float(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))
    return {
        "counts": counts.tolist(),
        "used_codes": int(nonzero.sum()),
        "usage_fraction": float(nonzero.mean()),
        "perplexity": float(np.exp(entropy)),
        "normalized_entropy": float(entropy / math.log(codebook_size)),
        "top_probability": float(probabilities.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit ordinary full-data Lloyd K-means on OpenX endpoint effects.",
        allow_abbrev=False,
    )
    parser.add_argument("--data-root-dir", required=True)
    parser.add_argument("--train-dataset-name", required=True)
    parser.add_argument(
        "--rlds-storage-format",
        choices=("auto", "tfds", "webdataset", "hybrid"),
        default="auto",
    )
    parser.add_argument("--target-control-hz", type=float, default=10.0)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--sampling-stride", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--shuffle-buffer-size", type=int, default=100_000)
    parser.add_argument("--fit-samples", type=int, default=500_000)
    parser.add_argument("--data-batch-size", type=int, default=4_096)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--gripper-weight", type=float, default=1.0)
    parser.add_argument("--kmeans-max-iterations", type=int, default=50)
    parser.add_argument("--kmeans-tolerance", type=float, default=1e-4)
    parser.add_argument("--kmeans-n-init", type=int, default=3)
    parser.add_argument("--kmeans-assignment-batch-size", type=int, default=65_536)
    parser.add_argument(
        "--kmeans-init-candidate-samples",
        type=int,
        default=0,
        help="K-means++ candidates; zero uses every fit sample.",
    )
    parser.add_argument(
        "--balance-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rlds-traj-transform-threads", type=int, default=0)
    parser.add_argument("--rlds-traj-read-threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    positive_fields = (
        "target_control_hz",
        "horizon",
        "sampling_stride",
        "action_dim",
        "shuffle_buffer_size",
        "fit_samples",
        "data_batch_size",
        "codebook_size",
        "gripper_weight",
        "kmeans_max_iterations",
        "kmeans_tolerance",
        "kmeans_n_init",
        "kmeans_assignment_batch_size",
        "log_every_batches",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.action_dim != 7:
        parser.error("The endpoint effect descriptor currently requires --action-dim=7")
    if args.fit_samples < args.codebook_size:
        parser.error("--fit-samples must be at least --codebook-size")
    if args.rlds_traj_transform_threads < 0 or args.rlds_traj_read_threads < 0:
        parser.error("RLDS thread counts must be non-negative")
    if args.kmeans_init_candidate_samples < 0:
        parser.error("--kmeans-init-candidate-samples must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    from scripts.action_vqvae.oxe_dataset import OXEActionDataset

    started = time.perf_counter()
    set_seed(args.seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(args.seed)
    except ModuleNotFoundError:
        pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(
        "[1/5] building clipped OpenX stream: "
        f"dataset={args.train_dataset_name} storage={args.rlds_storage_format} "
        f"target_hz={args.target_control_hz:g} horizon={args.horizon} "
        f"stride={args.sampling_stride}"
    )
    dataset = OXEActionDataset(
        args.data_root_dir,
        args.train_dataset_name,
        horizon=args.horizon,
        sampling_stride=args.sampling_stride,
        target_control_hz=args.target_control_hz,
        action_dim=args.action_dim,
        train=True,
        shuffle_buffer_size=args.shuffle_buffer_size,
        sample_ratio=1.0,
        balance_weights=args.balance_weights,
        traj_transform_threads=(
            args.rlds_traj_transform_threads
            if args.rlds_traj_transform_threads > 0
            else None
        ),
        traj_read_threads=(
            args.rlds_traj_read_threads
            if args.rlds_traj_read_threads > 0
            else None
        ),
        storage_format=args.rlds_storage_format,
        action_normalization="clip_q99",
        seed=args.seed,
    )
    _log(dataset.summary())
    loader = DataLoader(
        dataset,
        batch_size=args.data_batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    _log(f"[2/5] collecting {args.fit_samples} endpoint-effect descriptors")
    descriptors = _collect_descriptors(
        loader,
        num_samples=args.fit_samples,
        log_every_batches=args.log_every_batches,
    )
    _log(
        "[3/5] fitting one pooled z-score transform after per-dataset q01/q99 clipping"
    )
    global_mean, global_scale = fit_global_standardizer(descriptors)
    standardized = standardize_for_fit(
        descriptors,
        global_mean,
        global_scale,
        args.gripper_weight,
    )
    _log(
        "global descriptor mean="
        + np.array2string(global_mean, precision=6, separator=", ")
    )
    _log(
        "global descriptor scale="
        + np.array2string(global_scale, precision=6, separator=", ")
    )

    device = choose_device(args.device)
    _log(
        f"[4/5] fitting regular Lloyd K-means: K={args.codebook_size} "
        f"samples={len(standardized)} device={device}; every iteration uses all samples"
    )
    centers, fit_info = fit_full_kmeans(
        standardized,
        num_clusters=args.codebook_size,
        max_iterations=args.kmeans_max_iterations,
        tolerance=args.kmeans_tolerance,
        n_init=args.kmeans_n_init,
        assignment_batch_size=args.kmeans_assignment_batch_size,
        init_candidate_samples=args.kmeans_init_candidate_samples,
        seed=args.seed,
        device=device,
        progress=lambda run, iteration, inertia, shift, empty: _log(
            f"kmeans run={run}/{args.kmeans_n_init} iteration={iteration}/"
            f"{args.kmeans_max_iterations} mse={inertia / len(DESCRIPTOR_NAMES):.8f} "
            f"center_shift={shift:.7g} empty={empty}"
        ),
    )
    config = {
        "data_root_dir": str(Path(args.data_root_dir).resolve()),
        "train_dataset_name": args.train_dataset_name,
        "rlds_storage_format": dataset.storage_format,
        "target_control_hz": args.target_control_hz,
        "horizon": args.horizon,
        "sampling_stride": args.sampling_stride,
        "action_dim": args.action_dim,
        "action_preprocessing": "per_dataset_q01_q99_clip_then_pooled_effect_zscore",
        "effect_descriptor": "sum_xyz_sum_rpy_final_minus_initial_gripper",
        "fit_samples": args.fit_samples,
        "balance_weights": args.balance_weights,
        "codebook_size": args.codebook_size,
        "gripper_weight": args.gripper_weight,
        "seed": args.seed,
        "kmeans": {
            "algorithm": "full_lloyd",
            "max_iterations": args.kmeans_max_iterations,
            "tolerance": args.kmeans_tolerance,
            "n_init": args.kmeans_n_init,
            "assignment_batch_size": args.kmeans_assignment_batch_size,
            "init_candidate_samples": args.kmeans_init_candidate_samples,
            **fit_info,
        },
    }
    tokenizer = EffectTokenizer(
        centers=centers,
        global_mean=global_mean,
        global_scale=global_scale,
        gripper_weight=args.gripper_weight,
        config=config,
    )
    labels, distances, margins = tokenizer.assign(
        descriptors,
        batch_size=args.kmeans_assignment_batch_size,
        device=device,
    )
    training_metrics = {
        "samples": len(descriptors),
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "scaled_mse": float(distances.mean() / len(DESCRIPTOR_NAMES)),
        "relative_margin_mean": float(margins.mean()),
        "usage": _usage_metrics(labels, args.codebook_size),
        "elapsed_seconds": float(time.perf_counter() - started),
    }

    _log(f"[5/5] saving tokenizer: {Path(args.checkpoint).resolve()}")
    tokenizer.save(args.checkpoint)
    (output_dir / "training_metrics.json").write_text(
        json.dumps(training_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "dataset_statistics.json").write_text(
        json.dumps(dataset.statistics_for_json(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    usage = training_metrics["usage"]
    _log(
        f"complete: used={usage['used_codes']}/{args.codebook_size} "
        f"ppl={usage['perplexity']:.2f} H={usage['normalized_entropy']:.4f} "
        f"top={usage['top_probability']:.4f} "
        f"scaled_mse={training_metrics['scaled_mse']:.8f} "
        f"elapsed={_duration(time.perf_counter() - started)}"
    )


if __name__ == "__main__":
    main()
