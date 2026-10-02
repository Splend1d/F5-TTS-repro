#!/bin/bash
# Queue GPU-count variants of repro/train.sbatch; the first to start cancels the others.
# Env passthrough: CFG, EXTRA, TAG, LAST, VARIANTS ("8 4"), PART/QOS/TLIMIT, GTYPE (e.g. A100_80GB), TAKEOVER
# e.g. general A100s:  PART=general QOS=normal TLIMIT=2-00:00:00 GTYPE=A100_80GB VARIANTS=8 TAKEOVER=1 bash repro/submit_train.sh
F5=/data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS
TAG=${TAG:-main}; STATE=$F5/repro/state; mkdir -p $STATE
[ -f $STATE/$TAG.done ] && { echo "$TAG already done"; exit 0; }
[ -f $STATE/$TAG.stop ] && { echo "$TAG stopped (rm $STATE/$TAG.stop to allow)"; exit 0; }
# 40GB A100s can't hold 19200 frames/GPU; plus nodes that crashed on us
EXC="babel-y9-08,babel-y9-12,babel-y9-16"
[ -f $STATE/$TAG.exclude ] && EXC="$EXC,$(sort -u $STATE/$TAG.exclude | paste -sd,)"
export CFG=${CFG:-F5TTS_v1_Base} EXTRA="${EXTRA:-}" TAG LAST VARIANTS=${VARIANTS:-"8 4"} \
       PART=${PART:-preempt} QOS=${QOS:-preempt_qos} TLIMIT=${TLIMIT:-31-00:00:00} GTYPE=${GTYPE:-} TAKEOVER=${TAKEOVER:-0}
: > $STATE/$TAG.siblings
for n in $VARIANTS; do
  set -- $n $((n * 8)) $((n * 60))G
  id=$(sbatch --parsable -J f5-train-$TAG -p $PART --qos=$QOS --time=$TLIMIT \
       --gres=gpu:${GTYPE:+$GTYPE:}$1 --cpus-per-task=$2 --mem=$3 --exclude=$EXC \
       --export=ALL $F5/repro/train.sbatch)
  echo $id >> $STATE/$TAG.siblings
  echo "submitted $id (${1}x${GTYPE:-any} GPU, $PART/$QOS) tag=$TAG"
done
