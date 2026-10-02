#!/usr/bin/env bash
# STAGE 1: prove the evaluation pipeline works BEFORE training anything.
# Evaluate an existing SmolVLA LIBERO checkpoint. If this gives ~0% success, your setup
# (cameras, rendering, action space) is broken - fix it now, not after 8 hours of training.
source "$(dirname "$0")/common.sh"
BASELINE="${BASELINE:-HuggingFaceVLA/smolvla_libero}"   # verify on huggingface.co that this repo exists
lerobot-eval $(eval_args "$BASELINE" "outputs/eval/00_baseline_pretrained")
