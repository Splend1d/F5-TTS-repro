#!/bin/bash
# Queue GPU-count variants of repro/train.sbatch; the first to start cancels the others.
# SPREAD=1: n tasks x 1 GPU anywhere (multi-node DDP) instead of n GPUs on one node; SPREAD=node: the same srun
# launch with all n tasks on one node
# Env passthrough: CFG, EXTRA, TAG, LAST, SAVES_PER_SESSION (default 1), VARIANTS ("8 4"), PART/QOS/TLIMIT, GTYPE (e.g. A100_80GB), TAKEOVER
# CPG / MPG: CPUs and GB of memory per GPU for one-node jobs (default 8 / 60), e.g. CPG=5 MPG=50 to fit next to
# other jobs on a 64-CPU node
# e.g. general A100s:  PART=general QOS=normal GTYPE=A100_80GB VARIANTS=8 TAKEOVER=1 bash repro/submit_train.sh
F5=/data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS-sem
TAG=${TAG:-main}; STATE=$F5/repro/state; mkdir -p $STATE
[ -f $STATE/$TAG.done ] && { echo "$TAG already done"; exit 0; }
[ -f $STATE/$TAG.stop ] && { echo "$TAG stopped (rm $STATE/$TAG.stop to allow)"; exit 0; }
# 40GB A100s can't hold 19200 frames/GPU; plus nodes that crashed on us
EXC="babel-y9-08,babel-y9-12,babel-y9-16"
[ -f $STATE/$TAG.exclude ] && EXC="$EXC,$(sort -u $STATE/$TAG.exclude | paste -sd,)"
export CFG=${CFG:-F5TTS_v1_Base} EXTRA="${EXTRA:-}" TAG LAST SAVES_PER_SESSION VARIANTS=${VARIANTS:-"8 4"} \
       PART=${PART:-preempt} QOS=${QOS:-preempt_qos} TLIMIT=${TLIMIT:-00:45:00} GTYPE=${GTYPE:-} TAKEOVER=${TAKEOVER:-0} SPREAD=${SPREAD:-0} CPT=${CPT:-4} CPG=${CPG:-8} MPG=${MPG:-60}
# DEPEND (e.g. afterany:123): dependency for this submission only; not passed on to the job, so its own
# resubmissions do not wait on a job that has long finished
DEP=${DEPEND:+--dependency=$DEPEND}; unset DEPEND
: > $STATE/$TAG.siblings
for n in $VARIANTS; do
  if [ "$SPREAD" = 1 ]; then
    # n tasks x 1 GPU, placed wherever SLURM finds them (possibly different nodes). CPT CPUs per task (default 4 =
    # 3 data workers): small per-node footprint fits more backfill gaps; ~60 GB RAM per task
    RES="--ntasks=$n --gpus-per-task=${GTYPE:+$GTYPE:}1 --cpus-per-task=$CPT --mem-per-cpu=$((60 / CPT))G"
  elif [ "$SPREAD" = node ]; then
    # n tasks x 1 GPU on ONE node, launched with srun like SPREAD=1 (its NCCL settings work on the L40S nodes, where
    # ranks started directly by the batch script died at NCCL's shared-memory setup). Asks for GPUs per node, which
    # general requires ("minimum of 1 GPU" rejects --gpus-per-task). CPT CPUs and MPG GB per task.
    RES="--nodes=1 --ntasks=$n --gres=gpu:${GTYPE:+$GTYPE:}$n --cpus-per-task=$CPT --mem=$((n * MPG))G"
  else
    RES="--gres=gpu:${GTYPE:+$GTYPE:}$n --cpus-per-task=$((n * CPG)) --mem=$((n * MPG))G"
  fi
  # drop SLURM_* vars inherited when called from inside another job (e.g. the data job's SLURM_MEM_PER_NODE
  # clashes with this job's --mem-per-cpu and makes srun fail: "SLURM_MEM_PER_* are mutually exclusive")
  id=$(env $(env | grep -o '^SLURM_[A-Za-z0-9_]*' | sed 's/^/-u /') \
       sbatch --parsable -J f5-train-$TAG -p $PART --qos=$QOS --time=$TLIMIT $RES --exclude=$EXC $DEP \
       --export=ALL $F5/repro/train.sbatch)
  echo $id >> $STATE/$TAG.siblings
  echo "submitted $id (${n}x${GTYPE:-any} GPU, spread=$SPREAD, $PART/$QOS) tag=$TAG"
done
