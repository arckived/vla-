#!/usr/bin/env bash
# STAGE 2a - Experiment A: pretrained VLM frozen, train only the action expert (SmolVLA default).
# Trains on all LIBERO suites (multi-task), evaluates on $SUITE during training.
source "$(dirname "$0")/common.sh"
lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --policy.device="$DEVICE" \
  --dataset.repo_id=lerobot/libero \
  --env.type=libero --env.task="$SUITE" \
  --eval_freq="$EVAL_FREQ" --eval.batch_size=1 --eval.n_episodes=1 \
  --save_freq="$SAVE_FREQ" --steps="$STEPS" --batch_size="$BS" --seed=1000 \
  --output_dir=outputs/train/expert_only --job_name=expert_only \
  --wandb.enable="$WANDB" $EXTRA_TRAIN_ARGS
