#!/bin/bash
# Called by the trainer (env F5_ON_CHECKPOINT) right after it writes a named checkpoint model_<N>.pt.
# Usage: repro/on_checkpoint.sh <LABEL> <CFG> <CKPT_PATH> <UPDATE>
# Queues a 1-GPU Seed-TTS test-en eval of that checkpoint unless it was already evaluated or queued:
#   UPDATE % F5_FULL_EVAL_EVERY (default 50000) == 0 -> full: WER/SIM/UTMOS + held-out loss, run twice:
#       EMA weights (label <LABEL>) and online weights (label <LABEL>_online)
#   UPDATE % F5_LOSS_EVAL_EVERY (default 10000) == 0 -> loss-only: held-out loss, no generation
#   otherwise -> checkpoint kept, no eval
F5=/data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS-sem
LABEL=$1 CFG=$2 CKPT=$3 UPDATE=$4
if [ $((UPDATE % ${F5_FULL_EVAL_EVERY:-50000})) -eq 0 ]; then MODE=full
elif [ $((UPDATE % ${F5_LOSS_EVAL_EVERY:-10000})) -eq 0 ]; then MODE=loss
else echo "no eval scheduled for $LABEL@$UPDATE"; exit 0; fi
queue() {  # $1 label, $2 weights
  local S=$F5/results/$1/summary.jsonl
  if [ -f "$S" ] && grep -q "\"update\": $UPDATE," "$S"; then echo "eval $1@$UPDATE already done"; return; fi
  if squeue -h -u $USER -n f5-eval-ckpt -o %k | grep -qx "$1@$UPDATE"; then echo "eval $1@$UPDATE already queued"; return; fi
  mkdir -p $F5/results/$1
  local id=$(env $(env | grep -o '^SLURM_[A-Za-z0-9_]*' | sed 's/^/-u /') \
       sbatch --parsable --comment="$1@$UPDATE" $F5/repro/eval_ckpt.sbatch "$1" "$CFG" "$CKPT" "$UPDATE" "$MODE" "$2")
  echo "[$(date '+%F %T')] queued $MODE eval ($2) $id for $1@$UPDATE ($CKPT)" | tee -a $F5/repro/logs/on_checkpoint.log
}
queue "$LABEL" ema
[ "$MODE" = full ] && queue "${LABEL}_online" online
