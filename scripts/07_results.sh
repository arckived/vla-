#!/usr/bin/env bash
# STAGE 6: results table, chart, and GIFs for the README.
set -euo pipefail
python -m vla_study.results --eval-root outputs/eval --out results
mkdir -p results/gifs
# lerobot-eval saves rollout videos; convert a few to GIFs (adjust if the folder layout differs)
find outputs/eval -name "*.mp4" | head -n 8 | while read -r v; do
  name=$(echo "$v" | sed 's#outputs/eval/##; s#/#_#g; s#\.mp4$##')
  ffmpeg -loglevel error -y -i "$v" -vf "fps=10,scale=320:-1" "results/gifs/$name.gif"
done
ls results results/gifs
