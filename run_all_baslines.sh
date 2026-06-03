#!/usr/bin/env bash
set -euo pipefail

# Always run relative to the repository root (the location of this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Hardcoded baselines and datasets to iterate over
baselines=(
  pca_locality
  optimal
  wiener
  nearest_dataset
)

datasets=(
  afhq
  celeba_hq
  cifar10
  fashion_mnist
  mnist
  ffhq      # HuggingFace 64x64, ships a precomputed PCA
  afhqv2    # HuggingFace 64x64, ships a precomputed PCA
  # imagenet  # HuggingFace 64x64 (~16 GB download) -- uncomment to include
)

for baseline in "${baselines[@]}"; do
  for dataset in "${datasets[@]}"; do
    config_path="configs/${baseline}/${dataset}.yaml"
    if [[ ! -f "$config_path" ]]; then
      echo "Skipping missing config: $config_path" >&2
      continue
    fi

    echo "Running baseline=${baseline} dataset=${dataset}"
    uv run generate.py --config "$config_path"
    echo
  done
done

echo "All runs completed."
