#!/bin/bash
# Queue GPU-count variants of repro/train.sbatch; the first to start cancels the others.
# SPREAD=1: n tasks x 1 GPU anywhere (multi-node DDP) instead of n GPUs on one node
# Env passthrough: CFG, EXTRA, TAG, LAST, SAVES_PER_SESSION (default 1), VARIANTS ("8 4"), PART/QOS/TLIMIT, GTYPE (e.g. A100_80GB), TAKEOVER
# e.g. general A100s:  PART=general QOS=normal GTYPE=A100_80GB VARIANTS=8 TAKEOVER=1 bash repro/submit_train.sh
F5=/data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS
TAG=${TAG:-main}; STATE=$F5/repro/state; mkdir -p $STATE
[ -f $STATE/$TAG.done ] && { echo "$TAG already done"; exit 0; }
[ -f $STATE/$TAG.stop ] && { echo "$TAG stopped (rm $STATE/$TAG.stop to allow)"; exit 0; }
# 40GB A100s can't hold 19200 frames/GPU; plus nodes that crashed on us
EXC="babel-y9-08,babel-y9-12,babel-y9-16"
[ -f $STATE/$TAG.exclude ] && EXC="$EXC,$(sort -u $STATE/$TAG.exclude | paste -sd,)"
export CFG=${CFG:-F5TTS_v1_Base} EXTRA="${EXTRA:-}" TAG LAST SAVES_PER_SESSION VARIANTS=${VARIANTS:-"8 4"} \
       PART=${PART:-preempt} QOS=${QOS:-preempt_qos} TLIMIT=${TLIMIT:-00:45:00} GTYPE=${GTYPE:-} TAKEOVER=${TAKEOVER:-0} SPREAD=${SPREAD:-0} CPT=${CPT:-4}
: > $STATE/$TAG.siblings
for n in $VARIANTS; do
  if [ "$SPREAD" = 1 ]; then
    # n tasks x 1 GPU, placed wherever SLURM finds them (possibly different nodes). CPT CPUs per task (default 4 =
    # 3 data workers): small per-node footprint fits more backfill gaps; ~60 GB RAM per task
    RES="--ntasks=$n --gpus-per-task=${GTYPE:+$GTYPE:}1 --cpus-per-task=$CPT --mem-per-cpu=$((60 / CPT))G"
  else
    RES="--gres=gpu:${GTYPE:+$GTYPE:}$n --cpus-per-task=$((n * 8)) --mem=$((n * 60))G"
  fi
  id=$(sbatch --parsable -J f5-train-$TAG -p $PART --qos=$QOS --time=$TLIMIT $RES --exclude=$EXC \
       --export=ALL $F5/repro/train.sbatch)
  echo $id >> $STATE/$TAG.siblings
  echo "submitted $id (${n}x${GTYPE:-any} GPU, spread=$SPREAD, $PART/$QOS) tag=$TAG"
done
