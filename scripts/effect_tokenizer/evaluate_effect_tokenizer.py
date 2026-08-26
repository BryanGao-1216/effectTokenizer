"""Evaluate MLP effect VQ-VAE tokens and render assigned trajectories."""

from __future__ import annotations

import argparse
import csv
import html
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
    load_effect_checkpoint,
    set_seed,
)


def _log(message: str) -> None:
    print(f"[effect-eval] {message}", flush=True)


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _collect_actions(
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
        values = actions.numpy().astype(np.float32, copy=False)[:remaining]
        chunks.append(values.copy())
        collected += len(values)
        if (
            batch_index == 1
            or batch_index % log_every_batches == 0
            or collected >= num_samples
        ):
            elapsed = time.perf_counter() - started
            rate = collected / max(elapsed, 1e-8)
            eta = (num_samples - collected) / max(rate, 1e-8)
            _log(
                f"collect validation actions: {collected}/{num_samples} "
                f"({100.0 * collected / num_samples:.1f}%) "
                f"speed={rate:.0f} sample/s elapsed={_duration(elapsed)} "
                f"eta={_duration(eta)}"
            )
        if collected >= num_samples:
            break
    if collected < num_samples:
        raise ValueError(
            f"Validation stream returned {collected} samples, fewer than {num_samples}."
        )
    return np.concatenate(chunks, axis=0)


def _usage_metrics(labels: np.ndarray, codebook_size: int) -> dict[str, Any]:
    counts = np.bincount(labels, minlength=codebook_size)
    probabilities = counts / max(counts.sum(), 1)
    active = probabilities > 0
    entropy = float(-np.sum(probabilities[active] * np.log(probabilities[active])))
    return {
        "counts": counts,
        "probabilities": probabilities,
        "used_codes": int(active.sum()),
        "usage_fraction": float(active.mean()),
        "perplexity": float(np.exp(entropy)),
        "normalized_entropy": float(entropy / math.log(codebook_size)),
        "top_probability": float(probabilities.max()),
    }


def _evaluation_metrics(
    tokenizer: EffectTokenizer,
    descriptors: np.ndarray,
    labels: np.ndarray,
    latent_squared_distances: np.ndarray,
    margins: np.ndarray,
    reconstruction: np.ndarray,
    training_reconstruction_mse: float | None,
) -> dict[str, Any]:
    raw_prediction = reconstruction
    error = raw_prediction - descriptors
    per_dimension_mse = np.square(error).mean(axis=0)
    baseline_error = descriptors - descriptors.mean(axis=0, keepdims=True)
    weighted_target = tokenizer.weight(descriptors)
    weighted_prediction = tokenizer.weight(raw_prediction)
    mse = float(np.square(error).mean())
    baseline_mse = float(np.square(baseline_error).mean())
    weighted_mse = float(np.square(weighted_prediction - weighted_target).mean())
    weighted_baseline_mse = float(
        np.square(weighted_target - weighted_target.mean(axis=0, keepdims=True)).mean()
    )
    gripper_threshold = 0.25
    target_gripper = np.where(
        descriptors[:, -1] > gripper_threshold,
        1,
        np.where(descriptors[:, -1] < -gripper_threshold, -1, 0),
    )
    predicted_gripper = np.where(
        raw_prediction[:, -1] > gripper_threshold,
        1,
        np.where(raw_prediction[:, -1] < -gripper_threshold, -1, 0),
    )
    changed_gripper = target_gripper != 0
    usage = _usage_metrics(labels, tokenizer.codebook_size)
    prototype_codes, _, _ = tokenizer.assign(
        tokenizer.raw_centers,
        batch_size=tokenizer.codebook_size,
        device=next(tokenizer.model.parameters()).device,
    )
    cycle_consistency = float(
        np.mean(prototype_codes == np.arange(tokenizer.codebook_size))
    )
    return {
        "weighted_reconstruction_mse": weighted_mse,
        "weighted_reconstruction_r2": float(
            1.0 - weighted_mse / max(weighted_baseline_mse, 1e-12)
        ),
        "training_weighted_reconstruction_mse": training_reconstruction_mse,
        "validation_to_training_reconstruction_ratio": (
            None
            if training_reconstruction_mse is None
            else weighted_mse / max(training_reconstruction_mse, 1e-12)
        ),
        "effect_mse": mse,
        "effect_rmse": float(np.sqrt(mse)),
        "effect_r2": float(
            1.0 - mse / max(baseline_mse, 1e-12)
        ),
        "position_mse": float(per_dimension_mse[:3].mean()),
        "rotation_mse": float(per_dimension_mse[3:6].mean()),
        "gripper_mse": float(per_dimension_mse[6]),
        "gripper_direction_accuracy": float(
            np.mean(target_gripper == predicted_gripper)
        ),
        "gripper_change_rate": float(changed_gripper.mean()),
        "gripper_changed_direction_accuracy": (
            float(np.mean(target_gripper[changed_gripper] == predicted_gripper[changed_gripper]))
            if np.any(changed_gripper)
            else math.nan
        ),
        "gripper_change_threshold": gripper_threshold,
        "per_dimension_mse": per_dimension_mse,
        "relative_margin_mean": float(margins.mean()),
        "relative_margin_p10": float(np.quantile(margins, 0.1)),
        "latent_commitment_mse": float(
            latent_squared_distances.mean() / tokenizer.latent_dim
        ),
        "code_cycle_consistency": cycle_consistency,
        "usage": usage,
    }


def _sanity_checks(
    metrics: dict[str, Any],
    *,
    codebook_size: int,
) -> dict[str, bool]:
    usage = metrics["usage"]
    return {
        "no_codebook_collapse": bool(
            usage["usage_fraction"] >= 0.5
            and usage["perplexity"] >= 0.25 * codebook_size
            and usage["top_probability"] <= 0.2
        ),
        "vqvae_explains_effect_variance": bool(
            metrics["weighted_reconstruction_r2"] > 0.0
        ),
        "assignments_have_separation": bool(
            metrics["relative_margin_mean"] >= 0.05
        ),
        "code_decode_encode_cycle": bool(
            metrics["code_cycle_consistency"] >= 0.9
        ),
    }


def _trajectory_samples(
    actions: np.ndarray,
    labels: np.ndarray,
    *,
    codebook_size: int,
    examples_per_token: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    examples = np.zeros(
        (codebook_size, examples_per_token, *actions.shape[1:]), dtype=np.float32
    )
    example_counts = np.zeros(codebook_size, dtype=np.int64)
    mean_actions = np.zeros((codebook_size, *actions.shape[1:]), dtype=np.float64)
    counts = np.bincount(labels, minlength=codebook_size)
    for code_id in range(codebook_size):
        indices = np.flatnonzero(labels == code_id)
        if len(indices) == 0:
            continue
        mean_actions[code_id] = actions[indices].mean(axis=0, dtype=np.float64)
        selected_count = min(examples_per_token, len(indices))
        selected = rng.choice(indices, size=selected_count, replace=False)
        examples[code_id, :selected_count] = actions[selected]
        example_counts[code_id] = selected_count
    return examples, example_counts, mean_actions.astype(np.float32)


def _cumulative(action: np.ndarray, start: int) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)[:, start : start + 3]
    return np.concatenate(
        [np.zeros((1, 3), dtype=np.float32), np.cumsum(values, axis=0)], axis=0
    )


_ASSIGNED_TRAJECTORY_COLORS = (
    "#1e3a8a",
    "#1d4ed8",
    "#2563eb",
    "#0369a1",
    "#075985",
    "#3730a3",
    "#1e40af",
    "#0e7490",
)


def _token_figure(
    *,
    code_id: int,
    raw_center: np.ndarray,
    assigned_count: int,
    examples: np.ndarray,
    mean_action: np.ndarray,
    target_control_hz: float,
) -> Any:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise RuntimeError(
            "Plotly is required. Install dependencies from requirements-oxe.txt."
        ) from error

    figure = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "xy"}]],
        subplot_titles=(
            "Position trajectory (dataset-normalized)",
            "Rotation trajectory (dataset-normalized)",
            "Gripper open/close",
        ),
        horizontal_spacing=0.045,
    )
    for sample_id, action in enumerate(examples):
        position = _cumulative(action, 0)
        rotation = _cumulative(action, 3)
        name = f"assigned validation trajectories (shown={len(examples)})"
        sample_color = _ASSIGNED_TRAJECTORY_COLORS[
            sample_id % len(_ASSIGNED_TRAJECTORY_COLORS)
        ]
        figure.add_trace(
            go.Scatter3d(
                x=position[:, 0],
                y=position[:, 1],
                z=position[:, 2],
                mode="lines+markers",
                name=name,
                legendgroup="examples",
                showlegend=sample_id == 0,
                line={"color": sample_color, "width": 4},
                marker={"color": sample_color, "size": 2.5, "opacity": 0.9},
                hovertemplate=(
                    f"sample {sample_id}<br>step=%{{pointNumber}}<br>"
                    "XYZ=(%{x:.5f}, %{y:.5f}, %{z:.5f})<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter3d(
                x=rotation[:, 0],
                y=rotation[:, 1],
                z=rotation[:, 2],
                mode="lines+markers",
                name=name,
                legendgroup="examples",
                showlegend=False,
                line={"color": sample_color, "width": 4},
                marker={"color": sample_color, "size": 2.5, "opacity": 0.9},
                hovertemplate=(
                    f"sample {sample_id}<br>step=%{{pointNumber}}<br>"
                    "RPY=(%{x:.5f}, %{y:.5f}, %{z:.5f})<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
        time_axis = np.arange(action.shape[0], dtype=np.float32) / target_control_hz
        figure.add_trace(
            go.Scatter(
                x=time_axis,
                y=action[:, 6],
                mode="lines+markers",
                name=name,
                legendgroup="examples",
                showlegend=False,
                line={"color": sample_color, "width": 2.5, "shape": "hv"},
                marker={"color": sample_color, "size": 4, "opacity": 0.9},
                hovertemplate=(
                    f"sample {sample_id}<br>t=%{{x:.3f}}s<br>"
                    "gripper=%{y:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=3,
        )

    if assigned_count > 0:
        mean_position = _cumulative(mean_action, 0)
        mean_rotation = _cumulative(mean_action, 3)
        for column, curve, coordinates in (
            (1, mean_position, "XYZ"),
            (2, mean_rotation, "RPY"),
        ):
            figure.add_trace(
                go.Scatter3d(
                    x=curve[:, 0],
                    y=curve[:, 1],
                    z=curve[:, 2],
                    mode="lines+markers",
                    name="mean assigned trajectory",
                    legendgroup="mean",
                    showlegend=column == 1,
                    line={"color": "#dc2626", "width": 7},
                    marker={"color": "#dc2626", "size": 3},
                    hovertemplate=(
                        f"mean<br>step=%{{pointNumber}}<br>{coordinates}="
                        "(%{x:.5f}, %{y:.5f}, %{z:.5f})<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        time_axis = (
            np.arange(mean_action.shape[0], dtype=np.float32) / target_control_hz
        )
        figure.add_trace(
            go.Scatter(
                x=time_axis,
                y=mean_action[:, 6],
                mode="lines+markers",
                name="mean assigned trajectory",
                legendgroup="mean",
                showlegend=False,
                line={"color": "#dc2626", "width": 4, "shape": "hv"},
                marker={"color": "#dc2626", "size": 5},
                hovertemplate="mean<br>t=%{x:.3f}s<br>gripper=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=3,
        )

    # The VQ-VAE decoder predicts an endpoint effect, not a full path. Display
    # it as a dashed reference so it cannot be mistaken for a generated action.
    for column, endpoint in ((1, raw_center[:3]), (2, raw_center[3:6])):
        figure.add_trace(
            go.Scatter3d(
                x=[0.0, float(endpoint[0])],
                y=[0.0, float(endpoint[1])],
                z=[0.0, float(endpoint[2])],
                mode="lines+markers",
                name="VQ-VAE decoded effect",
                legendgroup="center",
                showlegend=column == 1,
                line={"color": "#111827", "width": 4, "dash": "dash"},
                marker={"color": "#111827", "size": 5, "symbol": "diamond"},
                hovertemplate="center=(%{x:.5f}, %{y:.5f}, %{z:.5f})<extra></extra>",
            ),
            row=1,
            col=column,
        )

    figure.update_scenes(
        aspectmode="data",
        dragmode="orbit",
        camera={"eye": {"x": 1.45, "y": 1.45, "z": 1.15}},
    )
    figure.update_layout(
        title={
            "text": (
                f"Effect token {code_id:03d}<br><sup>assigned={assigned_count} · "
                        f"shown={len(examples)} · decoded Δgripper={raw_center[6]:.4f}</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        height=650,
        width=1600,
        margin={"l": 25, "r": 25, "b": 55, "t": 110},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.07},
        scene={
            "xaxis": {"title": "normalized ΔX"},
            "yaxis": {"title": "normalized ΔY"},
            "zaxis": {"title": "normalized ΔZ"},
        },
        scene2={
            "xaxis": {"title": "normalized ΔRoll"},
            "yaxis": {"title": "normalized ΔPitch"},
            "zaxis": {"title": "normalized ΔYaw"},
        },
    )
    figure.update_xaxes(title_text="time (s)", row=1, col=3)
    figure.update_yaxes(title_text="gripper value", row=1, col=3)
    return figure


def _usage_figure(usage: dict[str, Any], dataset_name: str) -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise RuntimeError(
            "Plotly is required. Install dependencies from requirements-oxe.txt."
        ) from error
    counts = np.asarray(usage["counts"])
    frequencies = np.asarray(usage["probabilities"]) * 100.0
    code_ids = np.arange(len(counts))
    angles = code_ids / len(code_ids) * 360.0
    figure = go.Figure(
        go.Scatterpolar(
            r=np.concatenate([frequencies, frequencies[:1]]),
            theta=np.concatenate([angles, [360.0]]),
            thetaunit="degrees",
            mode="lines",
            line={"color": "#2563eb", "width": 2.5},
            customdata=np.column_stack(
                [np.concatenate([code_ids, code_ids[:1]]), np.concatenate([counts, counts[:1]])]
            ),
            hovertemplate=(
                "Token: %{customdata[0]:.0f}<br>Usage: %{r:.4f}%<br>"
                "Count: %{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title={
            "text": (
                "Effect-token validation usage<br>"
                f"<sup>{dataset_name} · used={usage['used_codes']}/{len(counts)} · "
                f"PPL={usage['perplexity']:.2f}</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        height=850,
        margin={"l": 65, "r": 65, "b": 55, "t": 100},
        showlegend=False,
        polar={
            "angularaxis": {"direction": "clockwise", "rotation": 90},
            "radialaxis": {"rangemode": "tozero", "ticksuffix": "%"},
        },
    )
    return figure


def _write_index(
    path: Path,
    *,
    counts: np.ndarray,
    example_counts: np.ndarray,
    checkpoint: Path,
    dataset_name: str,
) -> None:
    links = []
    for code_id, (assigned, shown) in enumerate(
        zip(counts, example_counts, strict=True)
    ):
        links.append(
            "<a class='token' target='viewer' "
            f"href='token_{code_id:03d}.html'>Token {code_id:03d}"
            f"<small>{int(assigned)} assigned · {int(shown)} shown</small></a>"
        )
    used = np.flatnonzero(counts > 0)
    initial_code = int(used[0]) if len(used) else 0
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Effect-token trajectories</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font: 14px system-ui, sans-serif; color: #0f172a; background: #f8fafc; }}
    header {{ padding: 14px 18px; background: white; border-bottom: 1px solid #cbd5e1; }}
    header h1 {{ margin: 0 0 5px; font-size: 20px; }}
    header p {{ margin: 2px 0; color: #475569; word-break: break-all; }}
    main {{ display: grid; grid-template-columns: 245px minmax(0, 1fr); height: calc(100vh - 92px); }}
    nav {{ overflow-y: auto; padding: 10px; border-right: 1px solid #cbd5e1; background: white; }}
    .token {{ display: block; margin-bottom: 6px; padding: 8px 10px; color: #1d4ed8; text-decoration: none; border: 1px solid #dbeafe; border-radius: 7px; }}
    .token:hover {{ background: #eff6ff; border-color: #93c5fd; }}
    .token small {{ display: block; margin-top: 2px; color: #64748b; }}
    iframe {{ width: 100%; height: 100%; border: 0; background: white; }}
  </style>
</head>
<body>
  <header>
    <h1>Effect-token trajectories</h1>
    <p>Dataset: {html.escape(dataset_name)}</p>
    <p>Checkpoint: {html.escape(str(checkpoint.resolve()))}</p>
  </header>
  <main>
    <nav>{''.join(links)}</nav>
    <iframe name="viewer" src="token_{initial_code:03d}.html" title="Token trajectory"></iframe>
  </main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _write_token_csv(
    path: Path,
    tokenizer: EffectTokenizer,
    labels: np.ndarray,
    descriptors: np.ndarray,
) -> None:
    counts = np.bincount(labels, minlength=tokenizer.codebook_size)
    centers = tokenizer.raw_centers
    with path.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "token_id",
            "count",
            "frequency",
            "within_effect_mse",
            *[f"decoded_{name}" for name in DESCRIPTOR_NAMES],
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for code_id in range(tokenizer.codebook_size):
            selected = descriptors[labels == code_id]
            within = (
                float(np.square(selected - centers[code_id]).mean())
                if len(selected)
                else math.nan
            )
            row = {
                "token_id": code_id,
                "count": int(counts[code_id]),
                "frequency": float(counts[code_id] / max(len(labels), 1)),
                "within_effect_mse": within,
            }
            row.update(
                {
                    f"decoded_{name}": float(centers[code_id, dimension])
                    for dimension, name in enumerate(DESCRIPTOR_NAMES)
                }
            )
            writer.writerow(row)


def _write_summary(path: Path, metrics: dict[str, Any]) -> None:
    result = metrics["vqvae"]
    usage = result["usage"]
    checks = metrics["sanity_checks"]
    passed = sum(checks.values())
    training_mse = result["training_weighted_reconstruction_mse"]
    validation_to_training = result[
        "validation_to_training_reconstruction_ratio"
    ]
    training_mse_text = (
        "n/a" if training_mse is None else f"{training_mse:.8f}"
    )
    validation_to_training_text = (
        "n/a"
        if validation_to_training is None
        else f"{validation_to_training:.6f}"
    )
    lines = [
        "# MLP Effect VQ-VAE Evaluation",
        "",
        f"Checkpoint: `{metrics['checkpoint']}`",
        "",
        "## Data contract",
        "",
        "- Actions are resampled before normalization and chunking.",
        "- Each dataset independently maps its action q01/q99 to [-1, 1], exactly as myStudy.",
        "- The gripper remains an absolute value; no pooled global z-score is fitted.",
        (
            "- Incomplete tail windows are stationary-padded."
            if metrics["data"]["pad_incomplete_windows"]
            else "- Incomplete tail windows are dropped."
        ),
        "- A simple MLP encoder, one learned codebook, and an MLP decoder are trained jointly.",
        "",
        "## Sanity checks",
        "",
        f"Passed **{passed}/{len(checks)}** broad checks.",
        "",
        "| Check | Result |",
        "|---|---:|",
        *[
            f"| {name} | {'PASS' if result else 'FAIL'} |"
            for name, result in checks.items()
        ],
        "",
        "## Validation VQ-VAE",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Samples | {metrics['data']['samples']} |",
        f"| Used tokens | {usage['used_codes']}/{metrics['model']['codebook_size']} |",
        f"| Perplexity | {usage['perplexity']:.4f} |",
        f"| Normalized entropy | {usage['normalized_entropy']:.6f} |",
        f"| Top probability | {usage['top_probability']:.6f} |",
        f"| Weighted reconstruction MSE | {result['weighted_reconstruction_mse']:.8f} |",
        f"| Weighted reconstruction R² | {result['weighted_reconstruction_r2']:.6f} |",
        f"| Training weighted reconstruction MSE | {training_mse_text} |",
        f"| Validation/training reconstruction ratio | {validation_to_training_text} |",
        f"| Unweighted effect MSE | {result['effect_mse']:.8f} |",
        f"| Unweighted effect R² | {result['effect_r2']:.6f} |",
        f"| Position MSE | {result['position_mse']:.8f} |",
        f"| Rotation MSE | {result['rotation_mse']:.8f} |",
        f"| Gripper MSE | {result['gripper_mse']:.8f} |",
        f"| Gripper direction accuracy | {result['gripper_direction_accuracy']:.6f} |",
        f"| Gripper change rate | {result['gripper_change_rate']:.6f} |",
        f"| Changed-gripper direction accuracy | {result['gripper_changed_direction_accuracy']:.6f} |",
        f"| Latent commitment MSE | {result['latent_commitment_mse']:.8f} |",
        f"| Assignment margin mean | {result['relative_margin_mean']:.6f} |",
        f"| Assignment margin p10 | {result['relative_margin_p10']:.6f} |",
        f"| Code decode/encode cycle | {result['code_cycle_consistency']:.6f} |",
        "",
        "The per-token HTML pages show assigned validation trajectories, their mean trajectory, and the endpoint effect decoded from that VQ code.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a single-codebook MLP effect VQ-VAE.",
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root-dir", required=True)
    parser.add_argument("--test-dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--examples-per-token", type=int, default=8)
    parser.add_argument("--assignment-batch-size", type=int, default=65_536)
    parser.add_argument("--rlds-traj-transform-threads", type=int, default=0)
    parser.add_argument("--rlds-traj-read-threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-batches", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--include-plotlyjs", choices=("directory", "cdn"), default="directory"
    )
    args = parser.parse_args()
    for field in (
        "num_samples",
        "batch_size",
        "examples_per_token",
        "assignment_batch_size",
        "log_every_batches",
    ):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.rlds_traj_transform_threads < 0 or args.rlds_traj_read_threads < 0:
        parser.error("RLDS thread counts must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    from scripts.action_vqvae.oxe_dataset import OXEActionDataset

    started = time.perf_counter()
    set_seed(args.seed)
    _log(f"[1/7] loading effect VQ-VAE: {Path(args.checkpoint).resolve()}")
    payload = load_effect_checkpoint(args.checkpoint, map_location="cpu")
    tokenizer = EffectTokenizer.from_payload(payload)
    config = tokenizer.config
    data_config = config["data"]
    device = choose_device(args.device)
    _log(
        f"VQ-VAE loaded: K={tokenizer.codebook_size} latent={tokenizer.latent_dim} "
        f"effect_dim={tokenizer.descriptor_dim} step={payload.get('global_step', 0)} device={device}"
    )

    try:
        import tensorflow as tf

        tf.random.set_seed(args.seed)
    except ModuleNotFoundError:
        pass
    _log(
        f"[2/7] building per-dataset q01/q99-normalized validation stream: "
        f"dataset={args.test_dataset_name} target_hz={data_config['target_control_hz']} "
        f"horizon={data_config['horizon']} stride={data_config['sampling_stride']} "
        f"pad_incomplete={data_config.get('pad_incomplete_windows', True)}"
    )
    dataset = OXEActionDataset(
        args.data_root_dir,
        args.test_dataset_name,
        horizon=int(data_config["horizon"]),
        sampling_stride=int(data_config["sampling_stride"]),
        pad_incomplete_windows=bool(
            data_config.get("pad_incomplete_windows", True)
        ),
        target_control_hz=float(data_config["target_control_hz"]),
        action_dim=int(data_config["action_dim"]),
        train=False,
        shuffle_buffer_size=args.num_samples,
        sample_ratio=1.0,
        balance_weights=bool(data_config.get("balance_weights", True)),
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
        storage_format=str(data_config["rlds_storage_format"]),
        seed=args.seed,
    )
    _log(dataset.summary())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    actions = _collect_actions(
        loader,
        num_samples=args.num_samples,
        log_every_batches=args.log_every_batches,
    )

    _log("[3/7] encoding effects, quantizing tokens, and decoding reconstructions")
    descriptors = compute_effect_descriptors(actions)
    labels, squared_distances, margins, reconstruction = tokenizer.encode_reconstruct(
        descriptors,
        batch_size=args.assignment_batch_size,
        device=device,
    )
    last_validation = payload.get("last_validation") or {}
    result = _evaluation_metrics(
        tokenizer,
        descriptors,
        labels,
        squared_distances,
        margins,
        reconstruction,
        (
            float(last_validation["reconstruction"])
            if "reconstruction" in last_validation
            else None
        ),
    )
    usage = result["usage"]
    _log(
        f"usage: used={usage['used_codes']}/{tokenizer.codebook_size} "
        f"ppl={usage['perplexity']:.2f} H={usage['normalized_entropy']:.4f} "
        f"top={usage['top_probability']:.4f}"
    )

    _log(
        f"[4/7] selecting up to {args.examples_per_token} trajectories per token"
    )
    examples, example_counts, mean_actions = _trajectory_samples(
        actions,
        labels,
        codebook_size=tokenizer.codebook_size,
        examples_per_token=args.examples_per_token,
        seed=args.seed + 50_021,
    )
    output_dir = Path(args.output_dir)
    trajectory_dir = output_dir / "token_trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[5/7] rendering {tokenizer.codebook_size} interactive token pages")
    plotlyjs_mode = args.include_plotlyjs
    render_started = time.perf_counter()
    for code_id in range(tokenizer.codebook_size):
        shown = int(example_counts[code_id])
        figure = _token_figure(
            code_id=code_id,
            raw_center=tokenizer.raw_centers[code_id],
            assigned_count=int(usage["counts"][code_id]),
            examples=examples[code_id, :shown],
            mean_action=mean_actions[code_id],
            target_control_hz=float(data_config["target_control_hz"]),
        )
        figure.write_html(
            trajectory_dir / f"token_{code_id:03d}.html",
            include_plotlyjs=plotlyjs_mode,
            full_html=True,
            auto_open=False,
            config={
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"effect_token_{code_id:03d}_trajectories",
                    "scale": 2,
                },
            },
        )
        if code_id == 0 or (code_id + 1) % 32 == 0:
            _log(
                f"render token pages: {code_id + 1}/{tokenizer.codebook_size} "
                f"elapsed={_duration(time.perf_counter() - render_started)}"
            )
    _write_index(
        trajectory_dir / "index.html",
        counts=np.asarray(usage["counts"]),
        example_counts=example_counts,
        checkpoint=Path(args.checkpoint),
        dataset_name=args.test_dataset_name,
    )

    _log("[6/7] rendering usage distribution and writing per-token data")
    usage_figure = _usage_figure(usage, args.test_dataset_name)
    usage_figure.write_html(
        output_dir / "token_usage_polar.html",
        include_plotlyjs=args.include_plotlyjs,
        full_html=True,
        auto_open=False,
        config={"displaylogo": False, "responsive": True},
    )
    _write_token_csv(
        output_dir / "per_token_metrics.csv", tokenizer, labels, descriptors
    )
    np.savez_compressed(
        output_dir / "trajectory_examples.npz",
        actions=examples,
        example_counts=example_counts,
        mean_actions=mean_actions,
        decoded_effect_prototypes=tokenizer.raw_centers,
        reconstructed_effects=reconstruction,
        assignment_counts=np.asarray(usage["counts"]),
    )

    metrics = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "elapsed_seconds": float(time.perf_counter() - started),
        "model": {
            "type": "mlp_effect_vqvae",
            "codebook_size": tokenizer.codebook_size,
            "descriptor_dim": tokenizer.descriptor_dim,
            "latent_dim": tokenizer.latent_dim,
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "gripper_weight": tokenizer.gripper_weight,
            "global_step": int(payload.get("global_step", 0)),
        },
        "data": {
            "dataset_name": args.test_dataset_name,
            "samples": len(actions),
            "dataset_summary": dataset.summary(),
            "preprocessing": "per_dataset_q01_q99_to_minus1_plus1_except_gripper",
            "target_control_hz": data_config["target_control_hz"],
            "horizon": data_config["horizon"],
            "sampling_stride": data_config["sampling_stride"],
            "pad_incomplete_windows": bool(
                data_config.get("pad_incomplete_windows", True)
            ),
        },
        "vqvae": result,
        "sanity_checks": _sanity_checks(
            result,
            codebook_size=tokenizer.codebook_size,
        ),
    }
    _log("[7/7] writing metrics and summary")
    (output_dir / "metrics.json").write_text(
        json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_summary(output_dir / "summary.md", metrics)
    _log(
        f"complete: output={output_dir.resolve()} "
        f"trajectory_index={(trajectory_dir / 'index.html').resolve()} "
        f"elapsed={_duration(time.perf_counter() - started)}"
    )


if __name__ == "__main__":
    main()
