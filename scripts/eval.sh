#!/usr/bin/env bash
set -euo pipefail

# 10 Hz, chunk length, stride, clipping and pooled normalization are loaded
# from the checkpoint so evaluation uses the exact training data contract.
python scripts/effect_tokenizer/evaluate_effect_tokenizer.py \
    --checkpoint outputs/effect_tokenizer.pt \
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
