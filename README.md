# Hidden Confounding Diagnostics

Codebase for Markovianity-based hidden-confounding diagnostics with a packaged simulation/CLI pipeline and notebook-based baselines (c-GC/c-GC\*, PCMCI+, LPCMCI).

## Project status

- The Python package and CLI workflow are active and runnable from `src/markovianity_diagnostic/`.
- Notebook workflows have been reorganized into method-specific folders (`notebooks/c-GC`, `notebooks/c-GC-star`, `notebooks/pcmci`, `notebooks/others`).
- The legacy `notebooks/run_notebook_simulation.ipynb` workflow is now represented by the method-specific simulation notebooks.
- Additional analysis notebooks are present for LPCMCI (`notebooks/lpcmci_simulation.ipynb`) and v2a-RSN stability metrics (`notebooks/v2a_RSN_metrics.ipynb`).

## Repository layout

- `src/markovianity_diagnostic/core/`: estimator implementation (`causalised-GC.py`) and utilities.
- `src/markovianity_diagnostic/experiments/`: synthetic scenarios, method adapters, metrics, experiment runner, and bootstrap calibration.
- `src/markovianity_diagnostic/cli/`: command entry points (`markov-exp`, `markov-bootstrap`).
- `notebooks/`: exploratory and benchmark notebooks (c-GC/c-GC\*, PCMCI+, LPCMCI, tutorials, domain metrics).
- `data/`: datasets, simulation helpers, and generated outputs.

## Installation

```bash
pip install -e .
```

Optional notebook extras:

```bash
pip install -e ".[notebooks]"
```

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

- `notebooks/pcmci/`: PCMCI+ analyses for single-lag/variable-lag and Markovian/non-Markovian settings.
- `notebooks/lpcmci_simulation.ipynb`: LPCMCI tau-sweep simulation analysis.
- `notebooks/c-GC/` and `notebooks/c-GC-star/`: c-GC and c-GC\* simulation notebooks across lag/confounding settings.
- `notebooks/v2a_RSN_metrics.ipynb`: ground-truth-free stability metrics over v2a-RSN adjacency outputs.
