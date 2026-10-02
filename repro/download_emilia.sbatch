#!/bin/bash
#SBATCH --job-name=f5-emilia-dl
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --qos=preempt_cpu_qos
#SBATCH --output=/data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS/repro/logs/%x-%j.out
source /data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS/repro/env.sh
python repro/download_emilia.py 12
