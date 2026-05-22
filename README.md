# Hidden Confounding Diagnostics

A comprehensive research framework for detecting and diagnosing latent confounders in time series data using **Markovianity-based causal inference** methods. This project implements multiple state-of-the-art approaches (c-GC, c-GC*, PCMCI+, JPCMCIplus, FullCI, LPCMCI) and provides both command-line tools and interactive Jupyter notebooks for simulation, benchmarking, and real-world applications.

## Overview

The project addresses a fundamental challenge in causal discovery from time series: **identifying whether unobserved (latent) confounders are present** and assessing how they affect causal structure learning. By leveraging Markovianity assumptions and information-theoretic measures, we develop diagnostics that can detect violations of causal assumptions and benchmark different methods across synthetic and real datasets.

## Project Status

- ✅ **Python package**: Active and runnable from `src/markovianity_diagnostic/` with CLI commands (`markov-exp`, `markov-bootstrap`)
- ✅ **Simulation pipeline**: Comprehensive synthetic scenario generation across multiple confounding and lag configurations
- ✅ **Notebooks**: Organized method-specific notebooks for c-GC, c-GC*, PCMCI+, JPCMCIplus, FullCI, and LPCMCI
- ✅ **Benchmark outputs**: Pre-computed aggregated results and visualizations in `outputs/`
- ✅ **Visualization**: Overlay plots comparing all methods across metrics (accuracy, precision, recall, F1, FPR, balanced accuracy)

## Repository Layout

```
hidden-confounding-diagnostics/
├── src/markovianity_diagnostic/        # Main Python package
│   ├── core/                           # Core estimators & utilities
│   │   └── causalised-GC.py           # c-GC and c-GC* implementations
│   ├── experiments/                    # Experiment infrastructure
│   │   ├── scenarios.py               # Synthetic scenario generators
│   │   ├── method_adapters.py         # Method integrations (PCMCI+, FullCI, etc.)
│   │   ├── metrics.py                 # Performance metrics & statistics
│   │   ├── runner.py                  # Experiment orchestration
│   │   └── bootstrap_calibration.py   # Surrogate-null bootstrap calibration
│   └── cli/                            # Command-line interface
│       ├── markov-exp                 # Main experiment runner
│       └── markov-bootstrap           # Bootstrap calibration CLI
│
├── notebooks/                          # Interactive analysis notebooks
│   ├── c-GC/                          # c-GC method notebooks (single/variable lag, Markovian/non-Markovian)
│   ├── c-GC-star/                     # c-GC* method notebooks (single/variable lag, Markovian/non-Markovian)
│   ├── pcmciplus/                     # PCMCI+ analysis notebooks
│   ├── jpcmciplus/                    # JPCMCIplus analysis notebooks
│   ├── lpcmci/                        # LPCMCI analysis notebooks
│   ├── fullci/                        # FullCI analysis notebooks
│   ├── c-GC-star/                     # c-GC* analysis notebooks
│   ├── others/                        # Additional tutorials and analyses
│   │   ├── lpcmci_simulation.ipynb    # LPCMCI benchmark
│   │   └── tigramite_tutorial_*.ipynb # Tutorial notebooks
│   └── v2a_RSN_metrics.ipynb          # Real-world stability metrics analysis
│
├── data/                               # Datasets and utilities
│   ├── run_simulation.py              # Standalone simulation helper
│   ├── twins/                         # Twin pairs dataset (semi-synthetic)
│   ├── v2a-RSN/                       # Real-world fMRI data
│   └── STABILITY_METRICS_GUIDE.md     # Metrics documentation
│
├── outputs/                            # Generated results and figures
│   ├── c-GC_results/                  # c-GC aggregated results
│   ├── c-GC-star_results/             # c-GC* aggregated results
│   ├── pcmci_plus_results/            # PCMCI+ aggregated results
│   ├── jpcmciplus_results/            # JPCMCIplus aggregated results
│   ├── lpcmci_results/                # LPCMCI aggregated results
│   ├── fullci_results/                # FullCI aggregated results
│   └── figures/                       # Comparison plots & visualizations
│
├── pyproject.toml                      # Project configuration & dependencies
└── README.md                           # This file
```

## Installation

This project uses **uv** for fast and reliable Python package management.

### Install uv (if needed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or with Homebrew:

```bash
brew install uv
```

### Install Project Dependencies

Install the core package:

```bash
uv sync
```

Install with optional notebook extras:

```bash
uv sync --extra notebooks
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

## Running the Notebooks

The project provides extensive Jupyter notebooks for interactive analysis and method comparison across different scenarios.

### Quick Start: Launch Jupyter

From the project root directory:

```bash
jupyter notebook
```

Or use JupyterLab for a more modern interface:

```bash
jupyter lab
```

### Notebook Organization

Notebooks are organized by **method** and **scenario**. Each notebook typically includes:
- **Scenario setup**: Data generation with specific confounding/lag configurations
- **Method execution**: Running the respective algorithm
- **Results aggregation**: Computing performance metrics
- **Visualization**: Plotting results across different `n_past` values

#### Available Notebooks

**Method-Specific Simulation Notebooks** (organized by algorithm):

| Method | Single-Lag | Variable-Lag |
|--------|-----------|--------------|
| **c-GC** | `notebooks/c-GC/singleLag-Markovian.ipynb` | `notebooks/c-GC/varLags-Markovian.ipynb` |
| | `notebooks/c-GC/singleLag-NonMarkovian.ipynb` | `notebooks/c-GC/varLags-NonMarkovian.ipynb` |
| **c-GC*** | `notebooks/c-GC-star/singleLag-Markovian.ipynb` | `notebooks/c-GC-star/varLags-Markovian.ipynb` |
| | `notebooks/c-GC-star/singleLag-NonMarkovian.ipynb` | `notebooks/c-GC-star/varLags-NonMarkovian.ipynb` |
| **PCMCI+** | `notebooks/pcmciplus/pcmci_plus_singleLag-Markovian.ipynb` | `notebooks/pcmciplus/pcmci_plus_varLags-Markovian.ipynb` |
| | `notebooks/pcmciplus/pcmci_plus_singleLag-NonMarkovian.ipynb` | `notebooks/pcmciplus/pcmci_plus_varLags-NonMarkovian.ipynb` |
| **JPCMCIplus** | `notebooks/jpcmciplus/jpcmciplus_singleLag-Markovian.ipynb` | `notebooks/jpcmciplus/jpcmciplus_varLags-Markovian.ipynb` |
| | `notebooks/jpcmciplus/jpcmciplus_singleLag-NonMarkovian.ipynb` | `notebooks/jpcmciplus/jpcmciplus_varLags-NonMarkovian.ipynb` |
| **LPCMCI** | `notebooks/lpcmci/lpcmci_singleLag-Markovian.ipynb` | `notebooks/lpcmci/lpcmci_varLags-Markovian.ipynb` |
| | `notebooks/lpcmci/lpcmci_singleLag-NonMarkovian.ipynb` | `notebooks/lpcmci/lpcmci_varLags-NonMarkovian.ipynb` |
| **FullCI** | `notebooks/fullci/fullci_singleLag-Markovian.ipynb` | `notebooks/fullci/fullci_varLags-Markovian.ipynb` |
| | `notebooks/fullci/fullci_singleLag-NonMarkovian.ipynb` | `notebooks/fullci/fullci_varLags-NonMarkovian.ipynb` |

**Scenario Explanation**:
- **Single-Lag**: Fixed lag structure across all causal relationships
- **Variable-Lag**: Different lags for different relationships
- **Markovian**: Data generated from Markovian processes (no hidden confounders)
- **Non-Markovian**: Data generated with latent confounders violating Markovianity

**Additional Analysis Notebooks**:
- `notebooks/others/lpcmci_simulation.ipynb`: LPCMCI method benchmark
- `notebooks/others/tigramite_tutorial_pcmci_fullci.ipynb`: FullCI and PCMCI+ tutorials
- `notebooks/others/tigramite_tutorial_pcmciplus.ipynb`: PCMCI+ deep dive
- `notebooks/v2a_RSN_metrics.ipynb`: Real-world fMRI stability metrics analysis
- `data/twins/twins_semisynthetic_markovianity.ipynb`: Twin pairs semi-synthetic analysis

### How to Run a Notebook

1. **Open a notebook**: Click on any `.ipynb` file in Jupyter to open it
2. **Execute sequentially**: 
   - Press `Shift + Enter` to run the current cell
   - Or use `Cell → Run All` from the menu to execute all cells
3. **Monitor progress**: Most notebooks print status messages showing experiment progress
4. **Output location**: Results are automatically saved to `outputs/{method}_results/{scenario}/`

### Expected Outputs

Each notebook generates:
- **Aggregated JSON**: `{method}_aggregated.json` containing mean/std metrics across seeds
- **Results CSV**: Detailed per-seed results with ground-truth accuracy
- **Summary JSON**: High-level statistics and metadata

### Viewing Results

Pre-computed overlay comparison plots are available in `outputs/figures/`:

```
outputs/figures/
├── singleLag-Markovian.png              # Overlay across available methods
├── singleLag-Markovian-all_methods.png  # Legacy all-methods overlay
├── singleLag-NonMarkovian.png
├── singleLag-NonMarkovian-all_methods.png
├── varLags-Markovian.png
├── varLags-Markovian-all_methods.png
├── varLags-NonMarkovian.png
└── varLags-NonMarkovian-all_methods.png
```

Generate or regenerate comparison plots with:

```bash
python src/create_overlay_plots.py
```

### Performance Tips

- **Reduce runtime**: Modify notebook variables like `repeats`, `T`, or `d` to run smaller experiments
- **Parallel notebooks**: Run different method notebooks simultaneously (they output to separate directories)
- **Monitor resources**: Check RAM/CPU usage; adjust batch sizes if memory-constrained

## CLI quick start

Run a compact experiment sweep:

```bash
markov-exp \
  --scenario order1_unconfounded \
  --method gcstar_cgc \
  --repeats 3 \
  --T 300 \
  --d 6 \
  --p-values 1 2 3 \
  --output-dir outputs/order1_cgc
```

Run surrogate-null bootstrap calibration:

```bash
markov-bootstrap \
  --scenario latent_common_driver \
  --method gcstar_fcgc \
  --T 300 \
  --d 6 \
  --p-values 1 2 3 \
  --p0 1 \
  --B 20 \
  --output-dir outputs/bootstrap_fcgc
```

## Available scenarios and methods

**Scenarios**: `order1_unconfounded`, `order3_unconfounded`, `latent_common_driver`, `variable_lag_unconfounded`, `hidden_nodes`

**Built-in methods**: `baseline_lstsq`, `gcstar_cgc`, `gcstar_fcgc`
You can also pass `--user-method module:function` to plug in external analyzers.

## Notebook map

- `notebooks/pcmciplus/`: PCMCI+ analyses for single-lag/variable-lag and Markovian/non-Markovian settings.
- `notebooks/jpcmciplus/`: JPCMCIplus analyses for single-lag/variable-lag and Markovian/non-Markovian settings.
- `notebooks/lpcmci/`: LPCMCI analyses for single-lag/variable-lag and Markovian/non-Markovian settings.
- `notebooks/c-GC/` and `notebooks/c-GC-star/`: c-GC and c-GC\* simulation notebooks across lag/confounding settings.
- `notebooks/v2a_RSN_metrics.ipynb`: ground-truth-free stability metrics over v2a-RSN adjacency outputs.
