#!/usr/bin/env bash
set -euo pipefail

# Run from the effectTokenizer project root. Extra CLI arguments may override
# values below, for example: bash scripts/train.sh --fit-samples 100000
python scripts/effect_tokenizer/train_effect_tokenizer.py \
    --data-root-dir /mnt/data27T/media/gwb/datasets/OpenX \
    --train-dataset-name action_tokenizer_plus \
    --rlds-storage-format hybrid \
    --target-control-hz 10 \
    --horizon 10 \
    --sampling-stride 2 \
    --action-dim 7 \
    --shuffle-buffer-size 100000 \
    --fit-samples 500000 \
    --data-batch-size 4096 \
    --codebook-size 256 \
    --gripper-weight 1.0 \
    --kmeans-max-iterations 50 \
    --kmeans-tolerance 1e-4 \
    --kmeans-n-init 3 \
    --seed 42 \
    --log-every-batches 20 \
    --device cuda \
    --checkpoint outputs/effect_tokenizer.pt \
    --output-dir outputs \
    "$@"
