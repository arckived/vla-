#!/usr/bin/env bash
# STAGE 4: quantization. Cheap offline benchmark first, then closed-loop evals.
# FIRST run:  python -m vla_study.quantize --list "$CKPT_A"
# and set VLM_RE / EXPERT_RE so they separate the VLM backbone from the action expert
# in YOUR checkpoint's module names. The defaults below are guesses.
source "$(dirname "$0")/common.sh"
CKPT="${CKPT:-$CKPT_A}"
VLM_RE="${VLM_RE:-\.vlm\.}"
EXPERT_RE="${EXPERT_RE:-expert}"

python -m vla_study.offline_bench --policy-path "$CKPT" --n-samples 200 \
  --configs "fp:none" "w8_all:8:" "w4_all:4:" "w4_vlm:4:$VLM_RE" "w4_expert:4:$EXPERT_RE" \
  --out outputs/offline_bench.json

run () {  # $1 name, $2 bits, $3 include-regex (may be empty)
  local extra=()
  [ -n "$3" ] && extra+=(--quant-include "$3")
  [ "$2" -le 4 ] && extra+=(--quant-group-size 128)
  python -m vla_study.patched_eval --quant-bits "$2" "${extra[@]}" -- $(eval_args "$CKPT" "outputs/eval/10_$1")
}
run w8_all 8 ""
run w4_all 4 ""
run w4_vlm 4 "$VLM_RE"
run w4_expert 4 "$EXPERT_RE"
