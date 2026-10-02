# Shared settings. Override any of these on the command line, e.g.  N_EP=20 bash scripts/01_eval_baseline.sh
set -euo pipefail
export MUJOCO_GL="${MUJOCO_GL:-egl}"      # headless GPU rendering for the MuJoCo simulator
SUITE="${SUITE:-libero_object}"           # LIBERO suite to evaluate on
N_EP="${N_EP:-10}"                        # episodes PER TASK (libero_object has 10 tasks -> 100 episodes)
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-20000}"                   # training steps
BS="${BS:-32}"                            # batch size; lower to 16 if you run out of GPU memory
WANDB="${WANDB:-false}"                   # set true after `wandb login` to get training curves
SAVE_FREQ="${SAVE_FREQ:-5000}"            # save a checkpoint every N steps (lower = less lost on a crash)
EVAL_FREQ="${EVAL_FREQ:-5000}"            # in-training sim eval every N steps; 0 disables it (saves GPU time)
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"  # anything else for lerobot-train, e.g. --dataset.video_backend=pyav
CKPT_A="outputs/train/expert_only/checkpoints/last/pretrained_model"
CKPT_B="outputs/train/expert_plus_vlm/checkpoints/last/pretrained_model"

eval_args () {  # $1 = policy path, $2 = output dir
  echo --policy.path="$1" --env.type=libero --env.task="$SUITE" \
       --eval.batch_size=1 --eval.n_episodes="$N_EP" --policy.device="$DEVICE" --output_dir="$2"
}
