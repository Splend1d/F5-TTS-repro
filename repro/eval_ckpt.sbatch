#!/bin/bash
# Evaluate ONE training checkpoint on Seed-TTS test-en with ONE GPU (WER/SIM/UTMOS + held-out loss), write
# results, log them to W&B.
# Usage: sbatch repro/eval_ckpt.sbatch <LABEL> <CFG> <CKPT_PATH> <UPDATE> [full|loss] [ema|online]
#   MODE   full (default): generate + WER/SIM/UTMOS + held-out loss; loss: held-out loss only (no generation)
#   WEIGHTS ema (default, what F5-TTS inference uses) | online (model_state_dict); use LABEL <run>_online for the latter
#   LABEL  run label (main | libritts); results go to results/<LABEL>/update_<UPDATE>/seedtts_test_en/
#   CFG    model config name (F5TTS_v1_Base | F5TTS_v1_Small) -> architecture + tokenizer/vocab
# Submitted automatically by repro/on_checkpoint.sh when training saves model_<N>.pt; safe to rerun or requeue
# (each finished stage is skipped).
#SBATCH --job-name=f5-eval-ckpt
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS-sem/repro/logs/%x-%j.out
set -eo pipefail
source /data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS-sem/repro/env.sh
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
# queued from inside a training job (F5_ON_CHECKPOINT hook): drop its distributed-training env, otherwise
# accelerate tries to join the training job's process group (MASTER_ADDR) and times out
unset MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK NODE_RANK \
      ACCELERATE_MIXED_PRECISION F5_SAVE_ON_SIGTERM F5_SAVES_PER_SESSION F5_ON_CHECKPOINT \
      NCCL_IB_DISABLE NCCL_SOCKET_IFNAME NCCL_P2P_DISABLE \
      WANDB_SERVICE WANDB_RUN_ID WANDB_NAME WANDB_RESUME  # training job's W&B session (socket on another node)
# faster-whisper (ctranslate2) dlopens cuDNN 9 / cuBLAS; use the ones bundled with torch so no node depends on
# system CUDA libs (babel-w9-26 had none on the default path and the WER step hung)
NV=$(python -c "import site;print(site.getsitepackages()[0])")/nvidia
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}
LABEL=${1:?label}; CFG=${2:?config}; CKPT=${3:?ckpt}; UPDATE=${4:?update}; MODE=${5:-full}; WEIGHTS=${6:-ema}
D=$F5/results/$LABEL/update_$UPDATE/seedtts_test_en
mkdir -p "$D"
echo "[$(date '+%F %T')] $MODE eval ($WEIGHTS weights) $LABEL update $UPDATE ($CFG) ckpt=$CKPT on $(nvidia-smi --query-gpu=name --format=csv,noheader)"

if [ "$MODE" = full ]; then
  if [ ! -f "$D/.infer_done" ]; then
    accelerate launch --num_processes 1 --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
      src/f5_tts/eval/eval_infer_batch.py -s 0 -n "$CFG" -c "$UPDATE" -t seedtts_test_en \
      --ckpt_path "$CKPT" --output_dir "$D" $([ "$WEIGHTS" = online ] && echo --no_ema)
    touch "$D/.infer_done"
  fi
  [ -f "$D/_wer_results.jsonl" ] || python src/f5_tts/eval/eval_seedtts_testset.py -e wer -l en -g "$D" -n "[0]"
  [ -f "$D/_sim_results.jsonl" ] || python src/f5_tts/eval/eval_seedtts_testset.py -e sim -l en -g "$D" -n "[0]"
  [ -f "$D/_utmos_results.jsonl" ] || python src/f5_tts/eval/eval_utmos.py --audio_dir "$D"
fi
# held-out flow-matching loss on the ground-truth audio (same loss as training; online + EMA weights)
[ -f "$D/_loss_results.json" ] || python repro/eval_loss.py "$CFG" "$CKPT" "$D/_loss_results.json"

python repro/eval_log.py "$LABEL" "$CFG" "$UPDATE" "$CKPT" "$D" "$MODE"
