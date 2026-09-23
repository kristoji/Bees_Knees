#!/bin/bash
# Run this FIRST on giano.cs.unibo.it, before anything else.
#
# It prints the facts the rest of the setup depends on and that cannot be checked from
# outside: the real partition names and time limits, the GPUs, the Python version, and
# how much room there is under /scratch.hpc. Paste the output back before pinning
# anything.
#
#   bash cluster/discover.sh
set -u
SCRATCH=${SCRATCH:-/scratch.hpc/$USER}

echo "=== host ==="
hostname; uname -a | cut -c1-100; echo

echo "=== partitions (name, time limit, nodes, state) ==="
sinfo -o "%20P %10l %6D %10t %N" 2>&1 | head -20; echo

echo "=== GPUs advertised per node ==="
sinfo -o "%20N %10c %10m %30G" 2>&1 | head -20; echo

echo "=== queue ==="
squeue -o "%.10i %.12P %.10u %.8T %.10M" 2>&1 | head -10; echo

echo "=== python ==="
for p in python3 python3.11 python3.10 python3.9; do
  command -v $p >/dev/null && echo "$p -> $($p --version 2>&1)"
done
echo "pip: $(command -v pip3 || echo 'not found')"; echo

echo "=== storage ==="
echo "home  : $HOME"
du -sh "$HOME" 2>/dev/null | tail -1
quota -s 2>/dev/null | tail -3
echo "scratch: $SCRATCH"
if [ -d "$SCRATCH" ]; then
  df -h "$SCRATCH" | tail -1
  echo "  (exists, $(du -sh "$SCRATCH" 2>/dev/null | cut -f1) used)"
else
  echo "  DOES NOT EXIST YET -- create it with: mkdir -p $SCRATCH"
fi
echo

echo "=== tools ==="
for t in git curl wget unrar unar 7z bsdtar module nvidia-smi; do
  printf "%-10s %s\n" "$t" "$(command -v $t || echo '-')"
done
echo

echo "=== modules (if any) ==="
module avail 2>&1 | head -20 || echo "no module system"
echo

echo "=== NOTE ==="
cat <<'TXT'
The official guide is ambiguous about whether /scratch.hpc is visible from the compute
nodes: one paragraph says it is only visible from giano, another says it is not visible
only from the lab machines, and the official sbatch example uses
--chdir=/scratch.hpc/name.surname. Settle it with a one-minute job:

  sbatch --partition=rtx2080 --gres=gpu:1 --time=00:01:00 \
         --chdir=/scratch.hpc/$USER --output=scratch_check.out \
         --wrap='hostname; pwd; ls -la; nvidia-smi'

If that job writes scratch_check.out under /scratch.hpc, the compute nodes see it and
everything below works as written.
TXT
