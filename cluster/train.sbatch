#!/bin/bash
#SBATCH --job-name=hive-value
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=francesco.giordani5@unibo.it
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --chdir=/scratch.hpc/francesco.giordani5/Bees_Knees
#SBATCH --output=/scratch.hpc/francesco.giordani5/logs/train-%j.out
#
# Train the value network.
#
#   sbatch cluster/train.sbatch                                      # defaults below
#   sbatch --partition=l40 --cpus-per-task=8 --mem=32G cluster/train.sbatch
#   HIDDEN=128 EPOCHS=80 sbatch cluster/train.sbatch
#
# Measured on the cluster (2026-09-23):
#   rtx2080  8 nodes, usually all idle, 4 cores, 44 GB RAM, RTX 2080 Ti 11 GB,
#            Turing sm_75, no bf16 -> fp16 autocast.  Default here because it runs
#            immediately.
#   l40      4 nodes, usually all allocated with jobs queued behind them, 8 cores,
#            64 GB RAM, L40 48 GB, Ada sm_89 -> bf16 autocast.
# Time limit is 3 days on both. The model is small and the graphs tiny, so the GPU is
# not the bottleneck; the queue is.
set -euo pipefail

SCRATCH=/scratch.hpc/$USER
SHARDS=${SHARDS:-$SCRATCH/hive_shards}
OUT=${OUT:-$SCRATCH/models/value-$SLURM_JOB_ID}

HIDDEN=${HIDDEN:-64}
LAYERS=${LAYERS:-3}
CONV=${CONV:-GIN}
EPOCHS=${EPOCHS:-60}
PATIENCE=${PATIENCE:-10}
BATCH=${BATCH:-1024}
LR=${LR:-1e-3}
EXTRA=${EXTRA:-}

echo "node   : $(hostname)"
echo "shards : $SHARDS"
echo "out    : $OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

source "$SCRATCH/venv/bin/activate"
mkdir -p "$OUT"

python src/train_value.py \
  --shards "$SHARDS" \
  --out "$OUT" \
  --hidden-dim "$HIDDEN" \
  --num-layers "$LAYERS" \
  --conv-type "$CONV" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch-size "$BATCH" \
  --lr "$LR" \
  $EXTRA

echo
echo "weights in $OUT/best.pt"
