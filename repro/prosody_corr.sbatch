#!/bin/bash
# Word-level prosody correlation (repro/prosody_corr.py) of every evaluated Seed-TTS test-en output against the
# ground-truth recording: emilia_small 150k/200k/250k, +ColBERT and +ModernBERT 200k/250k/300k (EMA and online), and
# the released F5-TTS v1 Base as a reference. Safe to requeue: finished systems are cached in OUT/cache.
#SBATCH --job-name=f5-prosody-corr
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS-sem/repro/logs/%x-%j.out
set -eo pipefail
source /data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS-sem/repro/env.sh
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
R=results; T=seedtts_test_en
SYS=()
for u in 150000 200000 250000; do
  SYS+=("small${u%000}k=$R/emilia_small/update_$u/$T" "small${u%000}k_online=$R/emilia_small_online/update_$u/$T")
done
for u in 200000 250000 300000; do
  SYS+=("colbert${u%000}k=$R/emilia_small_sem/update_$u/$T" "colbert${u%000}k_online=$R/emilia_small_sem_online/update_$u/$T")
  SYS+=("bert${u%000}k=$R/emilia_small_bert/update_$u/$T" "bert${u%000}k_online=$R/emilia_small_bert_online/update_$u/$T")
done
SYS+=("v1base_released=$R/../../F5-TTS/results/F5TTS_v1_Base_official_1250000/$T/seed0_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0")
python repro/prosody_corr.py --workers 7 "${SYS[@]}" --compare \
  colbert200k:small200k bert200k:small200k colbert250k:small250k bert250k:small250k \
  colbert200k:bert200k colbert250k:bert250k colbert300k:bert300k \
  small200k:small150k small250k:small200k colbert300k:colbert250k bert300k:bert250k \
  colbert250k_online:small250k_online bert250k_online:small250k_online v1base_released:small250k
