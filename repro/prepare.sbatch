#!/bin/bash
#SBATCH --job-name=f5-prep
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS/repro/logs/%x-%j.out
source /data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS/repro/env.sh
/usr/bin/time -v python repro/prepare_emilia_repro.py
python -c "o=set(open('data/Emilia_ZH_EN_pinyin/vocab.official.txt').read().split('\\n'));p=set(open('data/Emilia_ZH_EN_pinyin/vocab.prepared.txt').read().split('\\n'));print('only_official',sorted(o-p),'only_prepared',sorted(p-o))"
