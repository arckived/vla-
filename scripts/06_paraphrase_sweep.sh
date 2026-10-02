#!/usr/bin/env bash
# STAGE 5: instruction robustness. Same checkpoint, reworded instructions.
# Optional: LLM=Qwen/Qwen2.5-1.5B-Instruct bash scripts/06_paraphrase_sweep.sh
source "$(dirname "$0")/common.sh"
CKPT="${CKPT:-$CKPT_A}"
LLM="${LLM:-}"
TASKS="data/tasks_${SUITE}.json"
PARA="data/paraphrases_${SUITE}.json"
[ -f "$TASKS" ] || python -m vla_study.paraphrase extract --suite "$SUITE" --out "$TASKS"
if [ ! -f "$PARA" ]; then
  python -m vla_study.paraphrase generate --tasks "$TASKS" --out "$PARA" ${LLM:+--llm "$LLM"}
  echo ">>> Open $PARA, fix any paraphrase that changes the meaning, then re-run this script."; exit 0
fi
LEVELS="original synonym restruct verbose swapped"
[ -n "$LLM" ] && LEVELS="$LEVELS llm"
for level in $LEVELS; do
  python -m vla_study.patched_eval --paraphrase-file "$PARA" --paraphrase-level "$level" -- \
    $(eval_args "$CKPT" "outputs/eval/20_para_$level")
done
