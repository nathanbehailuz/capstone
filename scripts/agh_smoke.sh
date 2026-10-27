#!/usr/bin/env bash
set -euo pipefail

: "${AGH_INSTANCE_ID:?Set AGH_INSTANCE_ID to the rented instance id}"

for required in pretrained datasets/pkls; do
  if [[ ! -d "$required" ]]; then
    echo "Missing required remote data: $required" >&2
    exit 2
  fi
done

run_id="agh-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="results/$run_id"
mkdir -p "$run_dir"

python - <<'PY' > "$run_dir/manifest.json"
import json
import os
import subprocess

def output(*command):
    return subprocess.check_output(command, text=True).strip()

manifest = {
    "instance_id": os.environ["AGH_INSTANCE_ID"],
    "git_revision": output("git", "rev-parse", "HEAD"),
    "gpu": output(
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
    ).splitlines(),
}
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

python -m pytest

# Keep these deliberately short. Dataset-specific effective settings are
# recorded by run_RG_explainer.py in each output directory.
python run_RG_explainer.py --dataset syn1 --policy PPO --n_epochs 1 \
  --savedir "$run_dir/node"
python run_RG_explainer.py --dataset mutag --policy PPO --n_epochs 1 \
  --savedir "$run_dir/graph"
