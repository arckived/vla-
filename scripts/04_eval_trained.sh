#!/usr/bin/env bash
# STAGE 3: full evaluation of both trained checkpoints.
source "$(dirname "$0")/common.sh"
lerobot-eval $(eval_args "$CKPT_A" "outputs/eval/01_expert_only")
if [ -d "$CKPT_B" ]; then lerobot-eval $(eval_args "$CKPT_B" "outputs/eval/02_expert_plus_vlm"); else echo "skip B (no checkpoint)"; fi
