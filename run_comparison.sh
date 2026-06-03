#!/usr/bin/env bash
set -euo pipefail

# Run the denoiser comparison (edm_unet vs wiener vs optimal) on one dataset, all with
# the EDM Heun sampler and a shared seed so the generations are directly comparable.
#
# Usage:
#   ./run_comparison.sh [dataset] [extra generate.py overrides...]
# Examples:
#   ./run_comparison.sh afhqv2
#   ./run_comparison.sh cifar10 experiment.device=cpu sampling.num_samples=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

dataset="${1:-afhqv2}"   # afhqv2 | ffhq | cifar10
models=(edm_unet wiener optimal)

echo "Denoiser comparison on '${dataset}' (Heun sampler, shared seed)"
for m in "${models[@]}"; do
  cfg="configs/comparison/${dataset}_${m}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "Skipping missing config: $cfg" >&2
    continue
  fi
  echo "=== ${dataset} / ${m} ==="
  uv run generate.py --config "$cfg" "${@:2}"
  echo
done

echo "Done. Results under data/runs/comparison_${dataset}/"
