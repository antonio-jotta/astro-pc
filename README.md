# Astro-PC paper code

This repository contains the code used for the experiments in the Astro-PC paper. Astro-PC is an astrocyte-inspired extension of predictive-coding learning in which slow layer-local controller states modulate locally computed predictive-coding plasticity.

The project is intentionally kept as a small, flat Python codebase so that the paper experiments can be reproduced directly from the command line.

## 1. Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate astro-pc
```

If you use an existing environment, make sure it includes PyTorch, torchvision, NumPy, pandas, matplotlib, and scikit-learn.

MNIST is downloaded automatically into `./data` by torchvision when needed.

## 2. Repository contents

Core files:

```text
config.py                    Global experiment configuration
controller_configs.py        Predictive-coding and astro-controller dataclasses
data.py                      MNIST, noisy XOR, and tabular data loaders
engine.py                    Epoch-level training and evaluation loops
train.py                     Single-run training entry point
pc.py                        Predictive-coding training step and Astro-PC update logic
astro_leakage.py             Gain leakage and leakage-baseline logic
astro_phagocytosis.py        Phagocytosis-like pruning mechanism
checkpointing.py             Checkpointing and resume utilities
metrics.py                   Metric saving/loading helpers
plots.py                     Basic training plots
full_model_diagnostics.py    Internal controller, plasticity, leakage, and pruning diagnostics
experiment_utils.py          Run naming and numerical-failure helpers
early_stopping.py            Optional early stopping
```

Paper experiment scripts:

```text
exp_static_corruptions.py    Static clean/corrupted MNIST evaluation
exp_fgsm.py                  FGSM adversarial evaluation
exp_context_switching.py     Multi-context corrupted-MNIST switching experiment
exp_timescale.py             Controller-timescale ablation
```

Optional utility:

```text
make_corrupted_mnist.py      Optional utility for writing corrupted MNIST datasets to disk (after having MNIST in the data folder)
```

## 3. Experiments

All commands below assume they are run from the repository root. Generated outputs are written to `results/` unless otherwise stated.

### 3.1 Static clean and corrupted MNIST

This experiment trains each model on clean MNIST and evaluates the final checkpoints on clean and corrupted MNIST test sets. The corruptions are applied deterministically at evaluation time inside the script, so pre-generating corrupted MNIST files is not required.

```bash
python exp_static_corruptions.py \
  --out_dir results/mnist_static_corrupted_5seed_v1 \
  --seeds 0,1,2,3,4 \
  --gaussian_std 0.275 \
  --motion_kernel 11 \
  --pixel_size 5 \
  --brightness_delta 0.25 \
  --batch_size 64
```

The main comparison is:

```text
Predictive coding
Passive Astro-PC null
Astro-PC
```

Expected outputs include:

```text
results/mnist_static_corrupted_5seed_v1/corrupted_mnist_raw_results.csv
results/mnist_static_corrupted_5seed_v1/corrupted_mnist_summary_long.csv
results/mnist_static_corrupted_5seed_v1/corrupted_mnist_accuracy_table.csv
results/mnist_static_corrupted_5seed_v1/corrupted_mnist_accuracy_table.tex
```

The trained checkpoints are saved under:

```text
results/mnist_static_corrupted_5seed_v1/runs/
```

These checkpoints are reused by the FGSM evaluation.

### 3.2 FGSM adversarial evaluation

Run this after the static MNIST models have been trained. It evaluates the final checkpoints under FGSM perturbations without adversarial training.

```bash
python exp_fgsm.py \
  --trained_dir results/mnist_static_corrupted_5seed_v1 \
  --out_dir results/mnist_static_corrupted_5seed_v1/fgsm_eval\
  --seeds 0,1,2,3,4 \
  --epsilons 0,0.01,0.02,0.03,0.04,0.05 \
  --batch_size 64
```

Expected outputs include:

```text
results/mnist_static_corrupted_5seed_v1/fgsm_eval_fine/fgsm_mnist_raw_results.csv
results/mnist_static_corrupted_5seed_v1/fgsm_eval_fine/fgsm_mnist_summary_long.csv
results/mnist_static_corrupted_5seed_v1/fgsm_eval_fine/fgsm_mnist_accuracy_table.csv
results/mnist_static_corrupted_5seed_v1/fgsm_eval_fine/fgsm_mnist_accuracy_table.tex
```

The paper reports adversarial cross-entropy loss as the main FGSM metric because FGSM perturbs inputs in the direction that increases the loss.

### 3.3 Multi-context corrupted-MNIST switching

This experiment trains the same model sequentially across corrupted MNIST regimes while preserving model parameters and method-specific state variables between blocks.

```bash
python exp_context_switching.py \
  --out_dir results/mnist_corruption_multi_context_5seeds \
  --context_task corruption \
  --context_sequence clean,gaussian_noise,motion_blur,pixelation,brightness \
  --gaussian_std 0.275 \
  --motion_kernel 11 \
  --pixel_size 5 \
  --seeds 0,1,2,3,4 \
  --block_epochs 2 \
  --cycles 3 \
  --batch_size 64
```

Expected outputs include:

```text
results/mnist_corruption_multi_context_5seeds/context_eval_metrics.csv
results/mnist_corruption_multi_context_5seeds/context_switching_summary.csv
results/mnist_corruption_multi_context_5seeds/context_switching_summary_per_seed.csv
results/mnist_corruption_multi_context_5seeds/final_context_accuracy_table.csv
results/mnist_corruption_multi_context_5seeds/forgetting_summary.csv
```

Per-run diagnostics are saved inside folders such as:

```text
results/mnist_corruption_multi_context_5seeds/full_astro_pc__seed0/diagnostics/
```

### 3.4 Paired seed-level statistical summary

After running the static corrupted-MNIST experiment, the FGSM evaluation, and the multi-context switching experiment, compute the paired seed-level statistics reported in the paper with:

```bash
python compute_paired_stats.py \
  --corrupted_csv results/mnist_static_corrupted_5seed_v1/corrupted_mnist_raw_results.csv \
  --fgsm_csv results/mnist_static_corrupted_5seed_v1/fgsm_eval_fine/fgsm_mnist_raw_results.csv \
  --context_eval_csv results/mnist_corruption_multi_context_5seeds/context_eval_metrics.csv \
  --fgsm_eps_subset 0.01,0.02,0.03,0.04,0.05 \
  --fgsm_eps_point 0.05 \
  --out_dir results/paired_stats
```


### 3.5 Astro-PC internal diagnostics on noisy XOR

Noisy XOR is used as a low-dimensional diagnostic task for inspecting internal controller dynamics. To reproduce this run, edit `config.py` and set:

```python
dataset = "xor"
method = "pc"
astro_controller_enabled = True
astro_controller_leak_enabled = True
phago_enabled = True
```

Then run:

```bash
python train.py
```

The run is saved under `plots/` using the automatically generated run name. Internal diagnostic plots are written to the run's `diagnostics/` directory.

Useful diagnostic plots include:

```text
astro_state_trajectories.pdf
effective_gain_trajectories.pdf
memory_drive_rms_trajectories.pdf
received_leakage_trajectories.pdf
mean_pruning_pressure_trajectories.pdf
hard_pruning_events_per_epoch.pdf
```

After running the XOR diagnostics, change `dataset` back to `"mnist"` in `config.py` before reproducing MNIST experiments.

### 3.6 Controller-timescale ablation

The paper reports a focused controller-timescale ablation for Astro-PC with:

```text
T_a in {50, 100, 250}
```

Run the preconfigured ablation script:

```bash
python exp_timescale.py \
  --out_dir results/mnist_timescale_ablation \
  --seeds 0 \
  --epochs 50 \
  --batch_size 64
```

The script trains one Astro-PC model for each controller update period and then automatically writes the summary tables and overlay plot.

Expected outputs include:

```text
results/mnist_timescale_ablation/timescale_summary.csv
results/mnist_timescale_ablation/timescale_epoch_curves.csv
results/mnist_timescale_ablation/comparison/timescale_training_accuracy_overlay.png
results/mnist_timescale_ablation/comparison/timescale_training_accuracy_overlay.pdf
```

The plot `timescale_training_accuracy_overlay.pdf` is the paper-ready overlay of test accuracy over epoch for each value of \(T_a\).

## 4. Reproducibility notes

Unless otherwise stated, experiments use:

```text
Batch size: 64
MNIST normalization: mean 0.1307, std 0.3081
FGSM epsilon: raw pixel units in [0,1]
Static corruptions:
  Gaussian noise std: 0.275
  Motion blur kernel: 11
  Brightness offset: 0.25
  Pixelation size: 5 x 5
Main seed set: 0,1,2,3,4
```

The reported Astro-PC configuration is defined in `config.py`.

## 5. Output locations

The main experiment scripts write outputs to `results/`. The single-run entry point `train.py` writes outputs to `plots/` by default.

Large generated files, model checkpoints, and downloaded datasets are not included in the repository because they can be regenerated with the commands above.
