#!/bin/bash
# F5TTS_v1_Base on Emilia ZH+EN, preemptible + self-resubmitting.
# Submit via: bash repro/submit_train.sh   (queues an 8-GPU and a 4-GPU variant; first to start wins)
#
# Global batch is fixed at the official 8 x 38400 = 307200 frames/update:
#   19200 frames/GPU x NGPU x ACCUM, with NGPU x ACCUM = 16.
# The trainer's resume (skip update*accum batches/process) lands on the same data position
# for any such topology, so the run can hop between 4- and 8-GPU nodes across preemptions.
#SBATCH --job-name=f5-train
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --time=31-00:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@110
#SBATCH --output=/data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS/repro/logs/%x-%j.out
source /data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS/repro/env.sh
STATE=$F5/repro/state; mkdir -p $STATE
CFG=${CFG:-F5TTS_v1_Base}
EXTRA=${EXTRA:-}   # extra hydra overrides (e.g. for smoke tests)
TAG=${TAG:-main}   # separate lock/state namespace per run
LOCK=$STATE/$TAG.lock

log() { echo "[$(date '+%F %T')] [$SLURM_JOB_ID@$SLURMD_NODENAME] $*"; }

# ---- single-winner lock: cancel sibling pending jobs, bail if another copy is alive ----
if [ -f $LOCK ]; then
  owner=$(cat $LOCK)
  if [ "$owner" != "$SLURM_JOB_ID" ] && squeue -h -j "$owner" -t R 2>/dev/null | grep -q .; then
    if [ "${TAKEOVER:-0}" = 1 ]; then
      # migrate the run here: stop file blocks the old job's resubmit, TERM makes it save+exit
      log "taking over from running job $owner"
      touch $STATE/$TAG.stop
      scancel --signal=TERM --full $owner
      while squeue -h -j "$owner" 2>/dev/null | grep -q .; do sleep 10; done
      rm -f $STATE/$TAG.stop
      log "job $owner has exited; continuing from its checkpoint"
    else
      log "run owned by running job $owner; exiting"; exit 0
    fi
  fi
fi
echo $SLURM_JOB_ID > $LOCK
for j in $(squeue -h -u $USER -n $SLURM_JOB_NAME -t PD -o %i); do
  [ -f $STATE/$TAG.siblings ] && grep -qx $j $STATE/$TAG.siblings && scancel $j && log "cancelled sibling $j"
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=$(nvidia-smi -L | wc -l)
ACCUM=$((16 / NGPU))
[ $((NGPU * ACCUM)) -eq 16 ] || { log "unsupported NGPU=$NGPU"; exit 1; }
NW=$(( SLURM_CPUS_PER_TASK / NGPU - 1 )); [ $NW -gt 8 ] && NW=8
log "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c | tr '\n' ' ')"
log "NGPU=$NGPU ACCUM=$ACCUM workers/gpu=$NW CFG=$CFG TAG=$TAG EXTRA=$EXTRA"

resubmit() {  # $1 = reason; optional $2 = node to exclude
  [ -f $STATE/$TAG.stop ] && { log "stop file present, not resubmitting"; return; }
  [ -n "$2" ] && echo "$2" >> $STATE/$TAG.exclude
  log "resubmitting ($1)"
  CFG=$CFG EXTRA="$EXTRA" TAG=$TAG LAST=$LAST VARIANTS="$VARIANTS" bash $F5/repro/submit_train.sh
}

# ---- preemption: slurm sends TERM, then kills after the 2 min QOS grace time ----
# Ranks are launched directly (same env:// DDP setup `accelerate launch` produces) rather than via torchrun,
# whose agent SIGKILLs workers 30s after a SIGTERM -- too short for the ~25s checkpoint save.
# With F5_SAVE_ON_SIGTERM=1 each rank catches TERM, all ranks agree at the next micro-step,
# rank 0 writes model_last.pt, and they exit 143.
on_term() {
  log "SIGTERM (preempted?)"; resubmit preempted
  kill -TERM "${PIDS[@]}" 2>/dev/null
  # bounded wait (slurm's grace is 120s): a rank stuck in a collective must not hang the job
  for _ in $(seq 100); do
    alive=0; for p in "${PIDS[@]}"; do kill -0 $p 2>/dev/null && alive=1; done
    [ $alive = 0 ] && break; sleep 1
  done
  kill -9 "${PIDS[@]}" 2>/dev/null
  log "ranks exited after SIGTERM"; exit 0
}
trap on_term TERM

before=$(grep -h "Saved last checkpoint at update" $F5/repro/logs/f5-train-$TAG-*.out 2>/dev/null | awk '{print $NF}' | sort -n | tail -1)
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000)) WORLD_SIZE=$NGPU LOCAL_WORLD_SIZE=$NGPU
export ACCELERATE_MIXED_PRECISION=bf16 OMP_NUM_THREADS=1 F5_SAVE_ON_SIGTERM=1
PIDS=()
for r in $(seq 0 $((NGPU - 1))); do
  RANK=$r LOCAL_RANK=$r python src/f5_tts/train/train.py --config-name $CFG.yaml \
    ++datasets.batch_size_per_gpu=19200 ++datasets.max_samples=32 ++optim.grad_accumulation_steps=$ACCUM \
    ++datasets.num_workers=$NW ++ckpts.last_per_updates=${LAST:-500} \
    ++ckpts.logger=${F5_LOGGER:-tensorboard} $EXTRA &
  PIDS+=($!)
done
# wait for all ranks; if any rank fails, tear down the rest (they'd hang in NCCL otherwise)
rc=0
for _ in "${PIDS[@]}"; do
  wait -n; r=$?
  if [ $r -ne 0 ]; then
    rc=$r; log "a rank exited rc=$r; stopping the others"
    kill -TERM "${PIDS[@]}" 2>/dev/null; sleep 20; kill -9 "${PIDS[@]}" 2>/dev/null
    break
  fi
done
log "train exited rc=$rc"

if [ $rc -eq 0 ]; then
  log "TRAINING COMPLETE"; touch $STATE/$TAG.done; rm -f $LOCK; exit 0
fi

# ---- crash: requeue on a different node unless we're making no progress ----
after=$(grep -h "Saved last checkpoint at update" $F5/repro/logs/f5-train-$TAG-*.out 2>/dev/null | awk '{print $NF}' | sort -n | tail -1)
if [ "${after:-0}" != "${before:-0}" ]; then echo 0 > $STATE/$TAG.fails; fi
fails=$(( $(cat $STATE/$TAG.fails 2>/dev/null || echo 0) + 1 )); echo $fails > $STATE/$TAG.fails
if [ $fails -ge 3 ]; then
  log "3 consecutive crashes without progress -> NOT resubmitting; needs debugging"; rm -f $LOCK; exit 1
fi
# only blame the node for GPU-looking failures
bad=""; tail -c 200000 $F5/repro/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out | grep -qiE "CUDA error|NCCL|ECC|uncorrectable|Xid|device-side|CUBLAS|cuDNN error|illegal memory" && bad=$SLURMD_NODENAME
resubmit "crash rc=$rc (#$fails)" $bad
rm -f $LOCK
exit $rc
