#!/usr/bin/env bash
set -euo pipefail

# Native-rate time-windowing and per-dataset q01/q99 normalization are loaded
# from the VQ-VAE checkpoint so evaluation uses the training data contract.
python scripts/effect_tokenizer/evaluate_effect_tokenizer.py \
    --checkpoint outputs/effect_vqvae.pt \
    --data-root-dir /mnt/data27T/media/gwb/datasets/OpenX \
    --test-dataset-name action_tokenizer_plus \
    --output-dir outputs/evaluation \
    --num-samples 50000 \
    --batch-size 2048 \
    --examples-per-token 8 \
    --seed 42 \
    --device cuda \
    --include-plotlyjs directory \
    "$@"
