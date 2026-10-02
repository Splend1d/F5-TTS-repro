#!/bin/bash
# Usage: MODEL=F5TTS_v1_Base_official STEP=1250000 SEEDS="0 1 2" sbatch repro/eval.sbatch
#SBATCH --job-name=f5-eval
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS/repro/logs/%x-%j.out
set -eo pipefail
source /data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS/repro/env.sh
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
MODEL=${MODEL:-F5TTS_v1_Base_official}; STEP=${STEP:-1250000}; SEEDS=${SEEDS:-"0 1 2"}
TASKS=${TASKS:-"seedtts_test_zh seedtts_test_en ls_pc_test_clean"}
NG=$(nvidia-smi -L | wc -l); GPUS="[$(seq -s, 0 $((NG-1)))]"
LS=data/LibriSpeech/test-clean
nvidia-smi -L; echo "MODEL=$MODEL STEP=$STEP SEEDS=$SEEDS NG=$NG"
for seed in $SEEDS; do for task in $TASKS; do
  d="results/${MODEL}_${STEP}/${task}/seed${seed}_euler_nfe32_vocos_ss-1.0_cfg2.0_speed1.0"
  # skip inference if already complete (requeue-safe)
  if [ -f "$d/.infer_done" ]; then echo "skip infer $d"; else
    accelerate launch --multi_gpu --num_processes $NG --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
      src/f5_tts/eval/eval_infer_batch.py -s $seed -n $MODEL -t $task -c $STEP -p $LS
    d=$(ls -d results/${MODEL}_${STEP}/${task}/seed${seed}_euler_nfe32_vocos_ss*_cfg2.0_speed1.0 | head -1)
    touch "$d/.infer_done"
  fi
  d=$(ls -d results/${MODEL}_${STEP}/${task}/seed${seed}_euler_nfe32_vocos_ss*_cfg2.0_speed1.0 | head -1)
  case $task in
    seedtts_test_zh) E="python src/f5_tts/eval/eval_seedtts_testset.py -l zh";;
    seedtts_test_en) E="python src/f5_tts/eval/eval_seedtts_testset.py -l en";;
    ls_pc_test_clean) E="python src/f5_tts/eval/eval_librispeech_test_clean.py -p $LS";;
  esac
  [ -f "$d/_wer_results.jsonl" ] || $E -e wer -g "$d" -n "$GPUS"
  [ -f "$d/_sim_results.jsonl" ] || $E -e sim -g "$d" -n "$GPUS"
  [ -f "$d/_utmos_results.jsonl" ] || python src/f5_tts/eval/eval_utmos.py --audio_dir "$d"
done; done
python repro/summarize.py results/${MODEL}_${STEP}
