<div id="user-content-toc" display="inline">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Analytic Diffusion Studio</h1>
    </summary>
  </ul>

<p align="center">
  <i>A Unified Framework for Training-Free Diffusion Models</i>
</p>

<p align="center">
  <a href="https://github.com/analytic-diffusion/analytic-diffusion-studio/stargazers">
    <img src="https://img.shields.io/github/stars/analytic-diffusion/analytic-diffusion-studio.svg?style=social&label=Star">
  </a>
  &nbsp;
  <a href="https://github.com/analytic-diffusion/analytic-diffusion-studio/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg">
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg">
</p>

<p align="center">
  Analytic Diffusion Studio provides a modular, extensible codebase for training-free analytical diffusion models.
</p>

<p align="center">
    <a href="https://lukoianov.com">Artem Lukoianov</a>, &nbsp;
    <a href="https://chenyang.co">Chenyang Yuan</a>, &nbsp;
    <a href="https://cscarv.github.io/">Christopher Scarvelis</a>, &nbsp;
    <a href="https://scholar.google.com/citations?user=H-yl_JMAAAAJ&hl=en">Mason Kamb</a>
</p>

<p align="center">
  <img src="data/assets/main_comparison_github.png" alt="Main Comparison Results" width="100%">
</p>

> [NOTE:]
If you encounter any bugs, inconsistent behavior, or have suggestions how to improve this framework, please [open an issue](https://github.com/analytic-diffusion/analytic-diffusion-studio/issues) or better [make a pull request](https://github.com/analytic-diffusion/analytic-diffusion-studio/pulls). Your feedback is valuable!


## News

- **[2026/03/14]** Framework released as **Analytic Diffusion Studio** — a unified codebase for training-free diffusion methods.


## Supported Methods

| Method | Description | Paper |
|--------|-------------|-------|
| `pca_locality` | Analytical denoiser capturing locality from data statistics | [![arXiv](https://img.shields.io/badge/arXiv-2509.09672-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.09672) |
| `optimal` | Bayes-optimal estimator | — |
| `scfdm` | Smoothed Bayes-optimal estimator | [![arXiv](https://img.shields.io/badge/arXiv-2310.12395-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2310.12395) |
| `wiener` | Wiener filter denoiser | — |
| `nearest_dataset` | Nearest neighbor retrieval baseline | — |


## Supported Datasets

| Dataset | Config key | Auto-download | Notes |
|---------|-----------|---------------|-------|
| MNIST | `mnist` | Yes | |
| Fashion-MNIST | `fashion_mnist` | Yes | |
| CIFAR-10 | `cifar10` | Yes | |
| CelebA-HQ | `celeba_hq` | No | Download manually and place in `data/datasets/` |
| AFHQv2 | `afhq` | No | Download manually and place in `data/datasets/` |


## Environment Setup

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager.
- [Recommended] CUDA-capable GPU -- if you dont have it, make sure to change the device in the config to CPU/MPS

### Installation
No manual setup required! Just use `uv run` directly.

<details>
<summary>Alternative: Manual installation</summary>

If you prefer to set up the environment manually:

```bash
uv venv
source .venv/bin/activate  # On Linux/Mac (.venv\Scripts\activate on Windows)
uv pip install -e .
```
</details>


### Download the baseline UNET weights and the data
First we need to run this script to download the weights of the UNET models pre-trained for all of the baseline datasets.
It
You can skip this step, but then the metrics wont be available -- make sure to disable `baseline_path` in the config.

```bash
uv run download_baseline_weights.py
```

## Running Experiments

### Single Experiment

Now, run the command below to generate images with our analytical model.
UV will automatically create the virtual environment and install all dependencies (including the package in editable mode):

```bash
uv run generate.py --config configs/pca_locality/celeba_hq.yaml
```

The config path can be:
- Relative to `configs/` directory: `pca_locality/celeba_hq.yaml`
- Absolute path: `/path/to/config.yaml`

### Batch Experiments

Run all baseline-dataset combinations using the provided script:

```bash
./run_all_baselines.sh
```

This script iterates over:
- **Baselines**: `pca_locality`, `optimal`, `wiener`, `nearest_dataset`
- **Datasets**: `afhq`, `celeba_hq`, `cifar10`, `fashion_mnist`, `mnist`

It automatically skips missing config files and runs each experiment sequentially.

### Notebook

For quick experimentation, you can use the Jupyter notebook: `playground.ipynb`


## Configuration Files

Configuration files use YAML format with OmegaConf's `defaults` feature for inheritance. Each config inherits from `configs/defaults.yaml` and can override specific values.

### Configuration Structure

A typical config file (`configs/pca_locality/celeba_hq.yaml`) looks like:

```yaml
defaults:
  - /defaults.yaml

# Run metadata: name, seed, device, tags
experiment:
  run_name: pca_locality_celeba_hq  # Name of the run - overwritten in each individual config file
  tags: [baseline, pca_locality, celeba_hq]  # Tags for experiment organization
  seed: 42  # Random seed for reproducibility
  device: cuda  # Device to run on (cuda/cpu/mps)

# Dataset configuration: name, split, resolution, batch size
dataset:
  name: celeba_hq  # Dataset name (mnist, cifar10, celeba_hq, afhq, fashion_mnist)
  split: train  # Dataset split to use
  download: false  # Whether to auto-download (set false for manual downloads)
  batch_size: 256  # Batch size for dataset loading
  resolution: 64  # Image resolution (overrides default if specified)

# Model selection and hyperparameters
# Available models: pca_locality, optimal, wiener, nearest_dataset
model:
  name: pca_locality  # Model to use
  params:
    temperature: 1.0  # Temperature parameter for softmax weighting
    mask_threshold: 0.02  # Threshold for mask binarization

# Generation parameters: number of samples, inference steps
sampling:
  num_samples: 8  # Total number of samples to generate
  batch_size: 8  # Batch size for generation
  num_inference_steps: 10  # Number of diffusion steps

# Output and logging settings: WandB, file saving
metrics:
  baseline_path: "data/models/baseline_unet/celeba_hq/ckpt_epoch_200.pt"  # Path to baseline UNet checkpoint for comparison
  output:
    save_final_images: true  # Save individual sample images
    save_image_grid: true  # Save grid of all samples
    save_intermediate_images: true  # Save intermediate diffusion steps
  wandb:
    enabled: true  # Enable Weights & Biases logging
    project: locality-diffusion  # WandB project name
```

### Config Overrides via CLI

You can override any config value from the command line using dot notation:

```bash
uv run generate.py --config configs/pca_locality/celeba_hq.yaml \
    sampling.num_samples=16 \
    model.params.temperature=0.5\
    experiment.device=cpu
```

## Output Structure

Each experiment creates a run directory with the following structure:

```
data/runs/{experiment_name}/{run_name}_{optional:timestamp}/
├── config.yaml              # Saved configuration
├── grid.png                 # Grid of generated samples
├── metrics.json             # Computed metrics
├── logs/
│   └── generate.log         # Execution log
├── artifacts/
│   ├── images/              # Individual sample images
│   │   └── sample_0000.png
│   ├── intermediate_images/ # Intermediate diffusion steps
│   │   ├── x_t/            # Noisy images at each step
│   │   └── x0_pred/        # Predicted clean images at each step
│   └── comparison/          # Comparison grids (if baseline_path set)
└── code_snapshot/           # Git-tracked code snapshot
```



## Weights & Biases Integration

WandB logging is enabled by default. Using WandB is convinient for studying generation results, but can slowdown the runs. To disable or configure:

```yaml
metrics:
  wandb:
    enabled: false  # Disable WandB
    mode: offline   # Use offline mode
    project: my-project
```


## Contributing

We welcome contributions to this repository! Here are some ways you can help:

### Adding a New Method

Adding a new analytical diffusion method is straightforward:

1. **Subclass `BaseDenoiser`** in a new file under `src/local_diffusion/models/`:
   ```python
   from local_diffusion.models.base import BaseDenoiser
   from local_diffusion.models import register_model

   @register_model("your_method")
   class YourDenoiser(BaseDenoiser):
       def denoise(self, x_t, t, **kwargs):
           # Your denoising logic here
           ...
   ```
2. **Add the import** in `src/local_diffusion/models/__init__.py` so the decorator registers your model.
3. **Add config files** in `configs/your_method/` for each dataset you support (inherit from `configs/defaults.yaml`).
4. **Update the Supported Methods table** in this README and include an arXiv badge if applicable.
5. **Submit a PR** with sample outputs demonstrating your method's results.

### Reporting Issues

If you encounter bugs or have suggestions for improvements, please open an issue on GitHub. When reporting bugs, please include:
- A clear description of the problem
- Steps to reproduce the issue
- Your environment details (Python version, OS, etc.)
- Relevant error messages or logs

### Contributing Code

1. **Fork the repository** and create a new branch for your changes
2. **Follow the code style**: The project uses standard Python conventions. Ensure your code is well-documented and follows the existing patterns
3. **Add tests** if applicable (though the current codebase focuses on reproducibility of paper results)
4. **Update documentation** if you add new features or change existing behavior
5. **Submit a pull request** with a clear description of your changes


### Project Directory Structure

The project follows a structured layout:

```
analytic-diffusion-studio/
├── configs/              # Configuration files
│   ├── defaults.yaml     # Base configuration with common defaults
│   ├── pca_locality/     # Configs for the PCA locality method
│   ├── optimal/          # Optimal denoiser baseline
│   ├── wiener/           # Wiener filter baseline
│   └── nearest_dataset/  # Nearest neighbor baseline
├── src/
│   └── local_diffusion/  # Main package code
│       ├── models/       # Model implementations (pca_locality.py, etc.)
│       ├── data/         # Dataset loading utilities
│       ├── configuration.py  # Config management
│       └── metrics.py    # Evaluation metrics
├── data/                 # Data directory (created automatically)
│   ├── datasets/         # Dataset storage
│   ├── models/           # Precomputed models (Wiener filters, etc.)
│   ├── runs/             # Experiment outputs
│   └── wandb/            # Weights & Biases logs
├── generate.py           # Main entry point for experiments
├── playground.ipynb      # Interactive Jupyter notebook for experimentation
└── run_all_baselines.sh  # Batch script to run all experiments
```

## Citation

If you find this framework useful, please cite it:

```bibtex
@misc{analytic-diffusion-studio,
    title={Analytic Diffusion Studio: A Unified Framework for Training-Free Diffusion Models},
    author={Kamb, Mason and Lukoianov, Artem and Scarvelis, Christopher and Yuan, Chenyang},
    year={2025},
    url={https://github.com/analytic-diffusion/analytic-diffusion-studio},
}
```

> If you use a specific method, please also cite the corresponding paper (linked in the Supported Methods table above).
