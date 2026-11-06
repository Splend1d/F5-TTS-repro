#!/bin/bash
# Prosody correlation, emotion metrics and the summary for the MSP-IMPROV test sets, after
# repro/msp_emo_eval.sbatch has run for every system. Outputs: results/_analysis/msp_emo/<set>/summary.md
#SBATCH --job-name=f5-msp-analysis
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS-sem/repro/logs/%x-%j.out
set -eo pipefail
source /data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS-sem/repro/env.sh
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
SYSTEMS=${SYSTEMS:-"small250k colbert250k bert250k v1base_released"}
CMP="colbert250k:small250k bert250k:small250k colbert250k:bert250k v1base_released:small250k"
for SET in fixed improv; do
  A=$F5/results/_analysis/msp_emo/$SET
  ARGS=(); for s in $SYSTEMS; do ARGS+=("$s=$F5/results/msp_emo/$s/$SET"); done
  python repro/prosody_corr.py --workers 7 --testset $F5/data/msp_improv_emo/$SET --speaker-field 3 \
    --out $A/prosody "${ARGS[@]}" --compare $CMP
  python repro/msp_emo_eval.py --set $SET --workers 7 "${ARGS[@]}" \
    "ground_truth=$F5/results/msp_emo/ground_truth/$SET" --compare $CMP
done
