#!/bin/bash
#SBATCH --job-name=hive-rebuild
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=francesco.giordani5@unibo.it
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --chdir=/scratch.hpc/francesco.giordani5/Bees_Knees
#SBATCH --output=/scratch.hpc/francesco.giordani5/logs/rebuild-%j.out
#
# Rebuild the training shards from the board.txt files.
#
# This is a CPU-only job, but --gres=gpu:1 is kept because the cluster guide says the
# directive must stay as it is; both partitions are GPU partitions either way.
# rtx2080 is the default here because it has 8 idle nodes while l40's 4 nodes are
# usually allocated with a queue behind them. Use --partition=l40 --cpus-per-task=8
# if it happens to be free: the work is pure Python replay, so cores are what matter.
#
#   sbatch cluster/rebuild.sbatch
set -euo pipefail

SCRATCH=/scratch.hpc/$USER
RAW=${RAW:-$SCRATCH/hive_raw}
OUT=${OUT:-$SCRATCH/hive_shards}

echo "node    : $(hostname)"
echo "raw     : $RAW"
echo "out     : $OUT"
nproc

# rebuild_dataset.py needs numpy and nothing else, but use the venv so the Python
# version is the same one the training runs on.
source "$SCRATCH/venv/bin/activate"

python tools/rebuild_dataset.py \
  --src "$RAW" \
  --out "$OUT" \
  --collections '*tournament*' '*bots*' 'nokamute*' \
  --workers "${SLURM_CPUS_PER_TASK:-8}"

python tools/verify_dataset.py --shards "$OUT"
