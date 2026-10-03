#!/bin/bash
#SBATCH -p preempt --qos=preempt_qos --ntasks=4 --gpus-per-task=L40S:1 --cpus-per-task=4 --mem-per-cpu=8G --time=00:15:00
#SBATCH -J nccltest
source /data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS/repro/env.sh >/dev/null
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1) WORLD_SIZE=4 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=ib
echo "nodes $SLURM_JOB_NODELIST"
port=$((20000 + SLURM_JOB_ID % 20000))
for mode in default p2p_off p2p_shm_off; do
  echo "===== $mode"; port=$((port + 1))
  case $mode in p2p_off) export NCCL_P2P_DISABLE=1;; p2p_shm_off) export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1;; esac
  MASTER_PORT=$port timeout 120 srun -u --ntasks=4 --gpu-bind=none bash -c 'unset CUDA_VISIBLE_DEVICES; RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM python repro/diag/nccl_test.py 2>&1 | grep -E "^rank|via P2P|via SHM|NET/Socket/0|WARN|rror" | sort -u | head -8'
done
