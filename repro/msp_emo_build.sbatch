#!/bin/bash
# Build the MSP-IMPROV emotional test sets (repro/build_msp_emo.py): extract + trim clips, Whisper large-v3 on GPU,
# pair prompts. Safe to requeue: clips and transcripts are cached.
#SBATCH --job-name=f5-msp-build
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS-sem/repro/logs/%x-%j.out
set -eo pipefail
source /data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS-sem/repro/env.sh
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
NV=$(python -c "import site;print(site.getsitepackages()[0])")/nvidia  # cuDNN/cuBLAS for faster-whisper
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:${LD_LIBRARY_PATH:-}
python repro/build_msp_emo.py
