#!/bin/bash
# Generate and score one system on both MSP-IMPROV test sets (data/msp_improv_emo/{fixed,improv}): WER/SIM/UTMOS
# with the F5-TTS evaluators, outputs in results/msp_emo/<NAME>/<set>/. EMA weights, seed 0, NFE 32, CFG 2.
# Usage: sbatch repro/msp_emo_eval.sbatch <NAME> <CFG> <CKPT>
#   NAME ground_truth (CFG/CKPT "-"): the real target recordings, scored the same way (SIM/UTMOS ceilings)
# Safe to requeue: finished stages are skipped.
#SBATCH --job-name=f5-msp-eval
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --output=/data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS-sem/repro/logs/%x-%j.out
set -eo pipefail
source /data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS-sem/repro/env.sh
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
NV=$(python -c "import site;print(site.getsitepackages()[0])")/nvidia
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}
NAME=${1:?name}; CFG=${2:?config}; CKPT=${3:?ckpt}
echo "[$(date '+%F %T')] $NAME ($CFG) ckpt=$CKPT on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
for SET in fixed improv; do
  M=$F5/data/msp_improv_emo/$SET/meta.lst
  D=$F5/results/msp_emo/$NAME/$SET
  mkdir -p "$D"
  if [ "$NAME" = ground_truth ]; then
    for w in $F5/data/msp_improv_emo/$SET/wavs/*.wav; do ln -sfn "$(readlink -f "$w")" "$D/$(basename "$w")"; done
  elif [ ! -f "$D/.infer_done" ]; then
    accelerate launch --num_processes 1 --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
      src/f5_tts/eval/eval_infer_batch.py -s 0 -n "$CFG" -t msp_$SET --metalst "$M" --ckpt_path "$CKPT" \
      --output_dir "$D"
    touch "$D/.infer_done"
  fi
  [ -f "$D/_wer_results.jsonl" ] || python src/f5_tts/eval/eval_seedtts_testset.py -e wer -l en -g "$D" -n "[0]" --metalst "$M"
  [ -f "$D/_sim_results.jsonl" ] || python src/f5_tts/eval/eval_seedtts_testset.py -e sim -l en -g "$D" -n "[0]" --metalst "$M"
  [ -f "$D/_utmos_results.jsonl" ] || python src/f5_tts/eval/eval_utmos.py --audio_dir "$D"
  echo "[$(date '+%F %T')] $NAME $SET done: $(tail -n1 $D/_wer_results.jsonl) $(tail -n1 $D/_sim_results.jsonl) $(tail -n1 $D/_utmos_results.jsonl)"
done
