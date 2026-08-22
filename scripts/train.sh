#!/usr/bin/env bash
set -euo pipefail

# Resume example:
# RESUME_CHECKPOINT=outputs/effect_vqvae-step-0050000.pt bash scripts/train.sh
resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    resume_args=(--resume "${RESUME_CHECKPOINT}")
fi

python scripts/effect_tokenizer/train_effect_tokenizer.py \
    --data-root-dir /mnt/data27T/media/gwb/datasets/OpenX \
    --train-dataset-name action_tokenizer_plus \
    --rlds-storage-format hybrid \
    --target-control-hz 10 \
    --horizon 10 \
    --sampling-stride 2 \
    --action-dim 7 \
    --shuffle-buffer-size 100000 \
    --val-shuffle-buffer-size 4096 \
    --batch-size 4096 \
    --val-samples 4096 \
    --total-steps 100000 \
    --hidden-dim 128 \
    --latent-dim 16 \
    --num-hidden-layers 2 \
    --codebook-size 256 \
    --gripper-weight 1.0 \
    --codebook-loss-weight 1.0 \
    --commitment-loss-weight 0.25 \
    --usage-loss-weight 0.01 \
    --usage-temperature 1.0 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --warmup-steps 1000 \
    --weight-decay 1e-5 \
    --grad-clip-norm 1.0 \
    --seed 42 \
    --amp-dtype bf16 \
    --tensorboard \
    --tensorboard-log-dir outputs/tensorboard \
    --log-every-steps 50 \
    --val-every-steps 1000 \
    --save-every-steps 10000 \
    --device cuda \
    --checkpoint outputs/effect_vqvae.pt \
    --output-dir outputs \
    "${resume_args[@]}" \
    "$@"
