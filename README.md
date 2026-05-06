# Markovianity Diagnostic

GitHub-ready code package for the c-GC/fcGC Markovianity-diagnostic workflow used in the manuscript.

## Layout

- `src/markovianity_diagnostic/core/`
  Core estimator implementation.
- `src/markovianity_diagnostic/experiments/`
  Synthetic scenarios, adapters, metrics, bootstrap calibration, and plotting helpers.
- `src/markovianity_diagnostic/cli/`
  Command-line entry points for experiment and bootstrap runs.
- `notebooks/`
  Interactive analysis notebook aligned with the packaged code.

## Quick start

Create an environment and install the package in editable mode:

```bash
pip install -e .
```

Run a small experiment sweep:

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

Run bootstrap calibration:

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

## Notes

- The estimator path is intentionally lightweight and defaults to adjacency-level outputs so it integrates directly with the experiment harness.
- `cdt` is optional and only required if structural metrics such as SHD/SID are explicitly requested.
- The bundled notebook uses small defaults for interactive smoke tests. Scale `T`, `REPEATS`, and `n_perm` upward for manuscript-quality runs.
