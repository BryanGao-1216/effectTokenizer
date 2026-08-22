"""Train a single-codebook MLP VQ-VAE on OpenX endpoint effects."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.effect_tokenizer.effect_tokenizer import (  # noqa: E402
    DESCRIPTOR_NAMES,
    EffectTokenizer,
    MLPEffectVQVAE,
    MLPVQVAEConfig,
    choose_device,
    compute_effect_descriptors,
    load_effect_checkpoint,
    set_seed,
    vqvae_losses,
    weight_effects,
)


def _log(message: str) -> None:
    print(f"[effect-vqvae] {message}", flush=True)


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _usage_metrics(counts: np.ndarray) -> dict[str, float | int]:
    probabilities = counts / max(int(counts.sum()), 1)
    active = probabilities > 0
    entropy = float(-np.sum(probabilities[active] * np.log(probabilities[active])))
    return {
        "used_codes": int(active.sum()),
        "perplexity": float(np.exp(entropy)),
        "normalized_entropy": float(entropy / math.log(len(counts))),
        "top_probability": float(probabilities.max()),
    }


def _next_batch(
    loader: DataLoader,
    iterator: Iterator[tuple[Tensor]],
) -> tuple[tuple[Tensor], Iterator[tuple[Tensor]]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _actions_to_effect_tensor(
    actions: Tensor,
    *,
    gripper_weight: float,
    device: torch.device,
) -> Tensor:
    descriptors = compute_effect_descriptors(
        actions.numpy().astype(np.float32, copy=False)
    )
    weighted = weight_effects(descriptors, gripper_weight)
    return torch.from_numpy(weighted).to(device, non_blocking=True)


def _autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_grad_scaler(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda" and amp_dtype == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _lr_factor(step: int, *, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def _collect_validation_effects(
    loader: DataLoader,
    *,
    num_samples: int,
    gripper_weight: float,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    collected = 0
    for actions, in loader:
        remaining = num_samples - collected
        values = actions.numpy().astype(np.float32, copy=False)[:remaining]
        descriptors = compute_effect_descriptors(values)
        chunks.append(weight_effects(descriptors, gripper_weight))
        collected += len(descriptors)
        if collected >= num_samples:
            break
    if collected < num_samples:
        raise ValueError(
            f"Validation stream returned {collected} effects, expected {num_samples}."
        )
    return np.concatenate(chunks).astype(np.float32)


@torch.no_grad()
def _validate(
    model: MLPEffectVQVAE,
    effects: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    usage_temperature: float,
    codebook_loss_weight: float,
    commitment_loss_weight: float,
    usage_loss_weight: float,
) -> dict[str, float | int]:
    model.eval()
    totals = {
        "total": 0.0,
        "reconstruction": 0.0,
        "codebook": 0.0,
        "commitment": 0.0,
        "usage": 0.0,
        "margin": 0.0,
    }
    counts = np.zeros(model.codebook_size, dtype=np.int64)
    seen = 0
    for start in range(0, len(effects), batch_size):
        batch = torch.from_numpy(effects[start : start + batch_size]).to(device)
        output = model(batch, usage_temperature=usage_temperature)
        losses = vqvae_losses(
            output,
            batch,
            codebook_loss_weight=codebook_loss_weight,
            commitment_loss_weight=commitment_loss_weight,
            usage_loss_weight=usage_loss_weight,
        )
        size = len(batch)
        seen += size
        for key in ("total", "reconstruction", "codebook", "commitment", "usage"):
            totals[key] += float(losses[key]) * size
        totals["margin"] += float(output.relative_margin.mean()) * size
        counts += np.bincount(
            output.codes.cpu().numpy(), minlength=model.codebook_size
        )
    model.train()
    result: dict[str, float | int] = {
        key: value / max(seen, 1) for key, value in totals.items()
    }
    result.update(_usage_metrics(counts))
    return result


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _data_contract(args: argparse.Namespace, storage_format: str) -> dict[str, Any]:
    return {
        "data_root_dir": str(Path(args.data_root_dir).resolve()),
        "train_dataset_name": args.train_dataset_name,
        "rlds_storage_format": storage_format,
        "target_control_hz": args.target_control_hz,
        "horizon": args.horizon,
        "sampling_stride": args.sampling_stride,
        "action_dim": args.action_dim,
        "action_normalization": "per_dataset_q01_q99_to_minus1_plus1_except_gripper",
        "effect_descriptor": "sum_xyz_sum_rpy_final_minus_initial_gripper",
        "balance_weights": args.balance_weights,
    }


def _check_resume_contract(saved: dict[str, Any], current: dict[str, Any]) -> None:
    keys = (
        "train_dataset_name",
        "target_control_hz",
        "horizon",
        "sampling_stride",
        "action_dim",
        "action_normalization",
        "effect_descriptor",
        "balance_weights",
    )
    mismatches = [
        f"{key}: checkpoint={saved.get(key)!r}, current={current.get(key)!r}"
        for key in keys
        if saved.get(key) != current.get(key)
    ]
    if mismatches:
        raise ValueError(
            "Resume data contract does not match:\n  " + "\n  ".join(mismatches)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MLP VQ-VAE on per-dataset-normalized endpoint effects.",
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
    parser.add_argument("--val-shuffle-buffer-size", type=int, default=4_096)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--val-samples", type=int, default=4_096)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--gripper-weight", type=float, default=1.0)
    parser.add_argument("--codebook-loss-weight", type=float, default=1.0)
    parser.add_argument("--commitment-loss-weight", type=float, default=0.25)
    parser.add_argument("--usage-loss-weight", type=float, default=0.01)
    parser.add_argument("--usage-temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--balance-weights", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--rlds-traj-transform-threads", type=int, default=0)
    parser.add_argument("--rlds-traj-read-threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--val-every-steps", type=int, default=1_000)
    parser.add_argument("--save-every-steps", type=int, default=10_000)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard-log-dir", default="outputs/tensorboard")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()

    positive = (
        "target_control_hz",
        "horizon",
        "sampling_stride",
        "action_dim",
        "shuffle_buffer_size",
        "val_shuffle_buffer_size",
        "batch_size",
        "val_samples",
        "total_steps",
        "hidden_dim",
        "latent_dim",
        "num_hidden_layers",
        "codebook_size",
        "gripper_weight",
        "codebook_loss_weight",
        "commitment_loss_weight",
        "usage_temperature",
        "lr",
        "min_lr",
        "grad_clip_norm",
        "log_every_steps",
        "val_every_steps",
        "save_every_steps",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.action_dim != len(DESCRIPTOR_NAMES):
        parser.error(f"--action-dim must be {len(DESCRIPTOR_NAMES)}")
    if args.codebook_size < 2:
        parser.error("--codebook-size must be at least 2")
    if args.val_samples > args.val_shuffle_buffer_size:
        parser.error("--val-samples cannot exceed --val-shuffle-buffer-size")
    if args.warmup_steps < 0 or args.weight_decay < 0 or args.usage_loss_weight < 0:
        parser.error("warmup, weight decay, and usage loss weight must be non-negative")
    if args.min_lr > args.lr:
        parser.error("--min-lr cannot exceed --lr")
    if args.rlds_traj_transform_threads < 0 or args.rlds_traj_read_threads < 0:
        parser.error("RLDS thread counts must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    from scripts.action_vqvae.oxe_dataset import OXEActionDataset

    started = time.perf_counter()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    _log(
        "[1/5] building OpenX training stream with myStudy normalization: "
        f"dataset={args.train_dataset_name} target={args.target_control_hz:g}Hz "
        f"horizon={args.horizon} stride={args.sampling_stride}"
    )
    train_dataset = OXEActionDataset(
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
        seed=args.seed,
    )
    _log(train_dataset.summary())
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    train_iterator = iter(train_loader)
    data_contract = _data_contract(args, train_dataset.storage_format)

    payload = None
    if args.resume:
        _log(f"[2/5] loading full training state: {Path(args.resume).resolve()}")
        payload = load_effect_checkpoint(args.resume, map_location="cpu")
        _check_resume_contract(payload["config"]["data"], data_contract)
        tokenizer = EffectTokenizer.from_payload(payload)
        model = tokenizer.model.to(device)
        global_step = int(payload.get("global_step", 0))
        gripper_weight = tokenizer.gripper_weight
    else:
        _log("[2/5] initializing a new single-codebook MLP VQ-VAE")
        model_config = MLPVQVAEConfig(
            input_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_hidden_layers=args.num_hidden_layers,
            codebook_size=args.codebook_size,
        )
        model = MLPEffectVQVAE(model_config).to(device)
        global_step = 0
        gripper_weight = args.gripper_weight

    if global_step >= args.total_steps:
        raise ValueError(
            f"Checkpoint is already at step {global_step}, but total_steps={args.total_steps}."
        )
    if payload is not None:
        saved_loss = payload["config"]["loss"]
        saved_optimization = payload["config"]["optimization"]
        codebook_loss_weight = float(saved_loss["codebook_loss_weight"])
        commitment_loss_weight = float(saved_loss["commitment_loss_weight"])
        usage_loss_weight = float(saved_loss["usage_loss_weight"])
        usage_temperature = float(saved_loss["usage_temperature"])
        learning_rate = float(saved_optimization["lr"])
        minimum_learning_rate = float(saved_optimization["min_lr"])
        warmup_steps = int(saved_optimization["warmup_steps"])
        weight_decay = float(saved_optimization["weight_decay"])
        grad_clip_norm = float(saved_optimization["grad_clip_norm"])
        amp_dtype = str(saved_optimization["amp_dtype"])
    else:
        codebook_loss_weight = args.codebook_loss_weight
        commitment_loss_weight = args.commitment_loss_weight
        usage_loss_weight = args.usage_loss_weight
        usage_temperature = args.usage_temperature
        learning_rate = args.lr
        minimum_learning_rate = args.min_lr
        warmup_steps = args.warmup_steps
        weight_decay = args.weight_decay
        grad_clip_norm = args.grad_clip_norm
        amp_dtype = args.amp_dtype

    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_factor(
            step,
            warmup_steps=warmup_steps,
            total_steps=args.total_steps,
            min_ratio=minimum_learning_rate / learning_rate,
        ),
    )
    scaler = _make_grad_scaler(device, amp_dtype)
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to(optimizer, device)
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        if payload.get("grad_scaler_state_dict"):
            scaler.load_state_dict(payload["grad_scaler_state_dict"])
        _log(f"resumed at global_step={global_step}")

    _log(f"[3/5] caching {args.val_samples} held-out effects for validation")
    val_dataset = OXEActionDataset(
        args.data_root_dir,
        args.train_dataset_name,
        horizon=args.horizon,
        sampling_stride=args.sampling_stride,
        target_control_hz=args.target_control_hz,
        action_dim=args.action_dim,
        train=False,
        shuffle_buffer_size=args.val_shuffle_buffer_size,
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
        storage_format=train_dataset.storage_format,
        seed=args.seed + 1,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    validation_effects = _collect_validation_effects(
        val_loader,
        num_samples=args.val_samples,
        gripper_weight=gripper_weight,
    )

    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError("TensorBoard is enabled but unavailable.") from error
        writer = SummaryWriter(log_dir=args.tensorboard_log_dir)

    config = {
        "data": data_contract,
        "model": {
            "type": "mlp_effect_vqvae",
            **vars(model.config),
            "gripper_weight": gripper_weight,
        },
        "loss": {
            "codebook_loss_weight": codebook_loss_weight,
            "commitment_loss_weight": commitment_loss_weight,
            "usage_loss_weight": usage_loss_weight,
            "usage_temperature": usage_temperature,
        },
        "optimization": {
            "lr": learning_rate,
            "min_lr": minimum_learning_rate,
            "warmup_steps": warmup_steps,
            "total_steps": args.total_steps,
            "weight_decay": weight_decay,
            "grad_clip_norm": grad_clip_norm,
            "amp_dtype": amp_dtype,
        },
        "seed": args.seed,
    }
    tokenizer = EffectTokenizer(
        model=model,
        gripper_weight=gripper_weight,
        config=config,
    )

    _log(
        f"[4/5] training on {device}: steps={global_step + 1}..{args.total_steps} "
        f"batch={args.batch_size} K={model.codebook_size} latent={model.config.latent_dim}"
    )
    window_sums = {
        "total": 0.0,
        "reconstruction": 0.0,
        "codebook": 0.0,
        "commitment": 0.0,
        "usage": 0.0,
        "margin": 0.0,
        "grad": 0.0,
    }
    window_counts = np.zeros(model.codebook_size, dtype=np.int64)
    window_steps = 0
    log_started = time.perf_counter()
    last_validation: dict[str, float | int] | None = (
        payload.get("last_validation") if payload is not None else None
    )

    while global_step < args.total_steps:
        (actions,), train_iterator = _next_batch(train_loader, train_iterator)
        effects = _actions_to_effect_tensor(
            actions,
            gripper_weight=gripper_weight,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_dtype):
            output = model(effects, usage_temperature=usage_temperature)
            losses = vqvae_losses(
                output,
                effects,
                codebook_loss_weight=codebook_loss_weight,
                commitment_loss_weight=commitment_loss_weight,
                usage_loss_weight=usage_loss_weight,
            )
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip_norm
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1

        for key in ("total", "reconstruction", "codebook", "commitment", "usage"):
            window_sums[key] += float(losses[key].detach())
        window_sums["margin"] += float(output.relative_margin.detach().mean())
        window_sums["grad"] += float(grad_norm)
        window_counts += np.bincount(
            output.codes.detach().cpu().numpy(), minlength=model.codebook_size
        )
        window_steps += 1

        if global_step % args.log_every_steps == 0 or global_step == 1:
            elapsed = time.perf_counter() - started
            interval = time.perf_counter() - log_started
            usage = _usage_metrics(window_counts)
            averages = {
                key: value / max(window_steps, 1)
                for key, value in window_sums.items()
            }
            speed = window_steps / max(interval, 1e-8)
            eta = (args.total_steps - global_step) / max(speed, 1e-8)
            _log(
                f"step={global_step:07d}/{args.total_steps} "
                f"lr={optimizer.param_groups[0]['lr']:.6g} "
                f"loss={averages['total']:.5f} recon={averages['reconstruction']:.5f} "
                f"codebook={averages['codebook']:.5f} commit={averages['commitment']:.5f} "
                f"usage={averages['usage']:.4f} margin={averages['margin']:.3f} "
                f"ppl={usage['perplexity']:.1f} used={usage['used_codes']}/{model.codebook_size} "
                f"H={usage['normalized_entropy']:.3f} top={usage['top_probability']:.3f} "
                f"grad={averages['grad']:.3f} speed={speed:.2f}step/s "
                f"elapsed={_duration(elapsed)} eta={_duration(eta)}"
            )
            if writer is not None:
                for key, value in averages.items():
                    writer.add_scalar(f"train/{key}", value, global_step)
                for key, value in usage.items():
                    writer.add_scalar(f"train/{key}", value, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            window_sums = {key: 0.0 for key in window_sums}
            window_counts.fill(0)
            window_steps = 0
            log_started = time.perf_counter()

        if global_step % args.val_every_steps == 0 or global_step == args.total_steps:
            last_validation = _validate(
                model,
                validation_effects,
                batch_size=args.batch_size,
                device=device,
                usage_temperature=usage_temperature,
                codebook_loss_weight=codebook_loss_weight,
                commitment_loss_weight=commitment_loss_weight,
                usage_loss_weight=usage_loss_weight,
            )
            _log(
                f"val step={global_step:07d} loss={last_validation['total']:.5f} "
                f"recon={last_validation['reconstruction']:.5f} "
                f"ppl={last_validation['perplexity']:.1f} "
                f"used={last_validation['used_codes']}/{model.codebook_size} "
                f"H={last_validation['normalized_entropy']:.3f} "
                f"top={last_validation['top_probability']:.3f}"
            )
            if writer is not None:
                for key, value in last_validation.items():
                    writer.add_scalar(f"val/{key}", value, global_step)

        should_save = (
            global_step % args.save_every_steps == 0
            or global_step == args.total_steps
        )
        if should_save:
            _log(f"[5/5] saving full training checkpoint at step {global_step}")
            training_state = {
                "global_step": global_step,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "grad_scaler_state_dict": scaler.state_dict(),
                "last_validation": last_validation,
            }
            numbered = output_dir / f"effect_vqvae-step-{global_step:07d}.pt"
            tokenizer.save(numbered, training_state=training_state)
            tokenizer.save(args.checkpoint, training_state=training_state)
            _log(f"checkpoint={Path(args.checkpoint).resolve()}")

    if writer is not None:
        writer.flush()
        writer.close()
    (output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "dataset_statistics.json").write_text(
        json.dumps(train_dataset.statistics_for_json(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _log(
        f"complete: step={global_step} elapsed={_duration(time.perf_counter() - started)}"
    )


if __name__ == "__main__":
    main()
