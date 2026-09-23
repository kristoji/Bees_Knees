#!/bin/bash
# Prepare the Python environment for training on giano.cs.unibo.it.
#
# The venv goes under /scratch.hpc and NOT in the home directory: the home quota is
# 400 MB and a venv with torch is several GB. For the same reason every pip call uses
# --no-cache-dir, since pip's cache also lives in the home.
#
#   bash cluster/setup_giano.sh
set -euo pipefail

SCRATCH=${SCRATCH:-/scratch.hpc/$USER}
VENV=${VENV:-$SCRATCH/venv}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu118}

echo "repo    : $REPO"
echo "scratch : $SCRATCH"
echo "venv    : $VENV"
echo "torch   : $TORCH_INDEX"
echo

mkdir -p "$SCRATCH"

if [ ! -d "$VENV" ]; then
  echo "== creating the virtualenv =="
  python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --no-cache-dir --upgrade pip

echo "== torch (CUDA 11.8 wheels, matching the cluster's driver 535) =="
grep -E '^torch==' "$REPO/cluster/requirements-giano.txt" \
  | xargs pip install --no-cache-dir --index-url "$TORCH_INDEX"

echo "== the rest =="
grep -vE '^\s*#|^\s*$|^torch==' "$REPO/cluster/requirements-giano.txt" \
  | xargs pip install --no-cache-dir

echo
echo "== check =="
python - <<'PY'
import torch, torch_geometric, numpy
print("torch          ", torch.__version__)
print("torch_geometric", torch_geometric.__version__)
print("numpy          ", numpy.__version__)
print("cuda available ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         ", torch.cuda.get_device_name(0))
    print("capability     ", torch.cuda.get_device_capability(0))
    print("bf16           ", torch.cuda.is_bf16_supported())
else:
    print("(expected on the login node: giano itself has no GPU, only the compute nodes do)")
PY

cat <<TXT

Done. Two reminders:

  - /scratch.hpc deletes files that have not been accessed for 40 days. The venv, the
    dataset and the weights all live there, so copy anything you want to keep.
  - activate with: source $VENV/bin/activate
TXT
