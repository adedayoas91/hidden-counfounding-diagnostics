# Implementation Plan for Strengthening the Markovianity Diagnostic Work

This document turns the suggested improvements into an implementation roadmap. It is written for a developer or researcher who already has this repository and wants to make the work stronger for a Nature Methods-style submission.

The core goal is to move the project from descriptive graph-instability plots toward a calibrated, reproducible, method-comparison framework for diagnosing whether temporal causal discovery outputs are stable to conditioning depth.

## Current Starting Point

The implementation already has useful foundations:

- Python package: `hidden-confounding-diagnostics/src/markovianity_diagnostic/`
- Experiment harness: `experiments/runner.py`
- Bootstrap prototype: `experiments/bootstrap.py`, `experiments/bootstrap_runner.py`
- Graph metrics: `experiments/graph_metrics.py`
- Synthetic scenarios: `experiments/simulations.py`
- Method adapters: `experiments/adapters.py`
- CLI entry points: `markov-exp`, `markov-bootstrap`
- Existing simulation notebooks under `hidden-confounding-diagnostics/notebooks/simulations/`
- Existing v2a-RSN notebooks under `hidden-confounding-diagnostics/notebooks/v2a-RSNs/`
- Existing outputs under `hidden-confounding-diagnostics/outputs/`

The plan below assumes that new work should extend this package-first structure. Notebooks should call reusable package functions, not contain the only implementation of an analysis.

## Implementation Principles

1. Keep reusable logic in `src/markovianity_diagnostic/`.
2. Use notebooks for inspection, figures, and narrative checks.
3. Do not put unrelated analyses into one large notebook.
4. Group analyses only when they share the same data, method family, and interpretive question.
5. Existing notebooks may be amended when they already own that analysis path.
6. New notebooks should write standardized outputs to `outputs/`.
7. Every major claim should have a machine-readable artifact: JSON, CSV, figure, or log.
8. Every new metric should have at least one small synthetic smoke test.

## Proposed Directory Additions

```text
hidden-confounding-diagnostics/
  src/markovianity_diagnostic/
    experiments/
      calibration.py
      depth_selection.py
      edge_localization.py
      controls.py
      power.py
      nonlinear_tests.py
      real_data.py
      parallel.py
    methods/
      tigramite_adapters.py
      lpcmci_adapter.py
      svarfci_adapter.py
    reporting/
      tables.py
      figure_specs.py
      manuscript_exports.py
  tests/
    test_graph_metrics.py
    test_bootstrap_calibration.py
    test_depth_selection.py
    test_edge_localization.py
    test_simulation_scenarios.py
    test_adapters_smoke.py
  notebooks/
    calibration/
    simulations/
    real_data/
    comparisons/
    controls/
    power/
```

Do not move all existing notebooks immediately. First add reusable package code and create new notebooks that consume it. Existing notebooks can then be amended gradually.

## Notebook Strategy

Do not fit every improvement into one notebook. The analyses are too different. Separate notebooks by scientific question and computational pattern.

### Notebooks to Amend

Amend these current notebooks instead of replacing them:

```text
hidden-confounding-diagnostics/notebooks/v2a-RSNs/c-GC/c-GC-v2a_RSN.ipynb
hidden-confounding-diagnostics/notebooks/v2a-RSNs/c-GC-star/c-GC-star-v2a_RSN.ipynb
```

Required amendments:

- Export adjacency matrices for every recording and conditioning depth in a standardized format.
- Export `edge_counts`, `D_p`, `D_minus`, `D_plus`, `T_obs`, and cumulative instability as JSON/CSV.
- Add a short final cell that writes a manifest with input files, method settings, depth grid, and output paths.
- Do not add bootstrap calibration, power analysis, nonlinear tests, or method comparisons to these notebooks. Those belong in separate notebooks.

Amend existing simulation notebooks only where the change is local:

```text
hidden-confounding-diagnostics/notebooks/simulations/c-GC/*.ipynb
hidden-confounding-diagnostics/notebooks/simulations/c-GC-star/*.ipynb
hidden-confounding-diagnostics/notebooks/simulations/pcmciplus/*.ipynb
hidden-confounding-diagnostics/notebooks/simulations/jpcmciplus/*.ipynb
hidden-confounding-diagnostics/notebooks/simulations/lpcmci/*.ipynb
hidden-confounding-diagnostics/notebooks/simulations/fullci/*.ipynb
```

Required amendments:

- Standardize output schema.
- Add config manifests.
- Save compact ground truth and inferred adjacency dictionaries.
- Keep each method-specific notebook focused on that method and scenario family.

### New Notebooks to Create

#### 1. Calibration Notebooks

```text
hidden-confounding-diagnostics/notebooks/calibration/bootstrap_null_synthetic.ipynb
hidden-confounding-diagnostics/notebooks/calibration/bootstrap_null_v2a.ipynb
```

Purpose:

- `bootstrap_null_synthetic.ipynb`: validate surrogate-null calibration on synthetic Markovian and hidden-memory data.
- `bootstrap_null_v2a.ipynb`: apply calibrated null bands to v2a-RSN graphs.

Why separate:

- Synthetic calibration has ground truth and repeated simulations.
- v2a calibration uses real data, no ground truth, and biological interpretation.

Outputs:

```text
outputs/calibration/synthetic/bootstrap_summary.csv
outputs/calibration/synthetic/bootstrap_results.json
outputs/calibration/synthetic/bootstrap_T_obs.png
outputs/calibration/v2a/bootstrap_summary.csv
outputs/calibration/v2a/bootstrap_results.json
outputs/calibration/v2a/bootstrap_T_obs.png
outputs/calibration/v2a/depth_bands.png
```

#### 2. Depth Selection Notebook

```text
hidden-confounding-diagnostics/notebooks/real_data/v2a_depth_selection.ipynb
```

Purpose:

- Apply automatic depth-selection rules to v2a-RSN recordings.
- Report recommended `p_star` per fish and method.

Why separate:

- Depth selection is a decision-rule analysis, not a bootstrap-method validation.

Outputs:

```text
outputs/v2a-RSNs/depth_selection/depth_selection_summary.csv
outputs/v2a-RSNs/depth_selection/depth_selection.json
outputs/v2a-RSNs/depth_selection/depth_selection_plot.png
```

#### 3. Mechanism Ablation Notebooks

```text
hidden-confounding-diagnostics/notebooks/simulations/mechanism_ablations_hidden_state.ipynb
hidden-confounding-diagnostics/notebooks/simulations/mechanism_ablations_observation_artifacts.ipynb
hidden-confounding-diagnostics/notebooks/simulations/mechanism_ablations_nonstationarity.ipynb
```

Purpose:

- Separate causes of graph instability:
  - latent common driver
  - hidden nodes
  - omitted lag order
  - undersampling
  - measurement noise
  - nonstationarity
  - nonlinear dynamics

Grouping rule:

- Hidden common driver, hidden nodes, and omitted state can be grouped in `mechanism_ablations_hidden_state.ipynb`.
- Measurement noise and undersampling can be grouped in `mechanism_ablations_observation_artifacts.ipynb`.
- Nonstationarity should be separate because it needs time-varying parameters and stationarity diagnostics.

Outputs:

```text
outputs/simulations/mechanism_ablations/hidden_state/results.csv
outputs/simulations/mechanism_ablations/observation_artifacts/results.csv
outputs/simulations/mechanism_ablations/nonstationarity/results.csv
outputs/simulations/mechanism_ablations/*/summary.json
outputs/simulations/mechanism_ablations/*/figures/
```

#### 4. Edgewise Localization Notebook

```text
hidden-confounding-diagnostics/notebooks/real_data/v2a_edgewise_localization.ipynb
```

Purpose:

- Identify which edges repeatedly appear, disappear, or flip across conditioning depths.
- Apply FDR correction when surrogate frequencies are available.

Why separate:

- Edge localization produces large edge-level tables and heatmaps. It should not be mixed with global calibration or depth selection.

Outputs:

```text
outputs/v2a-RSNs/edge_localization/edge_instability.csv
outputs/v2a-RSNs/edge_localization/edge_fdr.csv
outputs/v2a-RSNs/edge_localization/deletion_heatmap.png
outputs/v2a-RSNs/edge_localization/addition_heatmap.png
```

#### 5. Positive and Negative Controls Notebook

```text
hidden-confounding-diagnostics/notebooks/controls/positive_negative_controls.ipynb
```

Purpose:

- Run known Markovian null controls.
- Run known hidden-memory positive controls.
- Run shuffled, block-shuffled, and phase-randomized controls where appropriate.

Why one notebook:

- These controls answer one question: does the diagnostic behave sensibly under known null and known violation regimes?

Outputs:

```text
outputs/controls/control_summary.csv
outputs/controls/control_results.json
outputs/controls/control_curves.png
```

#### 6. Latent-Variable Method Comparison Notebook

```text
hidden-confounding-diagnostics/notebooks/comparisons/lpcmci_svarfci_comparison.ipynb
```

Purpose:

- Compare the conditioning-depth diagnostic with methods that explicitly allow latent confounding.
- Start with LPCMCI because Tigramite is already a dependency.
- Add SVAR-FCI only if implementation availability is stable.

Why separate:

- This is a method-comparison analysis with different outputs from the c-GC family. It should not be mixed with calibration.

Outputs:

```text
outputs/comparisons/lpcmci_svarfci/summary.csv
outputs/comparisons/lpcmci_svarfci/pag_edge_marks.csv
outputs/comparisons/lpcmci_svarfci/comparison_figures/
```

#### 7. Nonlinear Dependence Tests Notebook

```text
hidden-confounding-diagnostics/notebooks/comparisons/nonlinear_ci_tests.ipynb
```

Purpose:

- Compare ParCorr with nonlinear CI tests such as GPDC or CMIknn where feasible.
- Determine whether depth instability is robust to the CI test family.

Why separate:

- Nonlinear tests have different runtime and sample-size behavior. They should not be folded into the standard method-comparison notebook.

Outputs:

```text
outputs/comparisons/nonlinear_ci_tests/results.csv
outputs/comparisons/nonlinear_ci_tests/runtime.csv
outputs/comparisons/nonlinear_ci_tests/figures/
```

#### 8. Power Analysis Notebook

```text
hidden-confounding-diagnostics/notebooks/power/sample_size_power_analysis.ipynb
```

Purpose:

- Quantify how sample size, node count, conditioning depth, edge density, and confounder strength affect detection.

Why separate:

- This is a large Monte Carlo grid and should be isolated from biological-data analysis.

Outputs:

```text
outputs/power/sample_size_power/results.csv
outputs/power/sample_size_power/power_curves.png
outputs/power/sample_size_power/runtime_summary.csv
```

#### 9. Runtime and Scaling Notebook

```text
hidden-confounding-diagnostics/notebooks/power/runtime_scaling.ipynb
```

Purpose:

- Measure runtime as a function of node count, time length, method, conditioning-depth grid, bootstrap replicates, and parallelism.

Why separate:

- Runtime scaling is an engineering analysis. It should not be mixed with statistical power.

Outputs:

```text
outputs/power/runtime_scaling/runtime_grid.csv
outputs/power/runtime_scaling/runtime_plot.png
```

#### 10. Manuscript Figure Assembly Notebook

```text
hidden-confounding-diagnostics/notebooks/reporting/nature_methods_figures.ipynb
```

Purpose:

- Load final outputs from the analyses above.
- Produce manuscript-ready figures and tables.

Why separate:

- Figure assembly should not recompute analyses. It should only read frozen outputs and export final assets.

Outputs:

```text
nature_methods/figures/
nature_methods/tables/
outputs/reporting/nature_methods_figure_manifest.json
```

## Phase 1: Harden Surrogate-Null Bootstrap Calibration

### Goal

Turn the existing bootstrap prototype into a robust calibration module for `T_obs = max_{p > p0} D_p`.

### Existing Files to Use

```text
hidden-confounding-diagnostics/src/markovianity_diagnostic/experiments/bootstrap.py
hidden-confounding-diagnostics/src/markovianity_diagnostic/experiments/bootstrap_runner.py
hidden-confounding-diagnostics/src/markovianity_diagnostic/cli/bootstrap.py
hidden-confounding-diagnostics/src/markovianity_diagnostic/experiments/plotting.py
```

### New or Refactored Files

```text
hidden-confounding-diagnostics/src/markovianity_diagnostic/experiments/calibration.py
hidden-confounding-diagnostics/tests/test_bootstrap_calibration.py
```

### Implementation Tasks

1. Move general calibration functions from `bootstrap.py` into `calibration.py`.
2. Keep `bootstrap.py` as a compatibility wrapper or remove it after updating imports.
3. Add null model classes:
   - `VARNullModel`
   - `ResidualBootstrapNull`
   - `MovingBlockBootstrapNull`
   - optional `StationaryBootstrapNull`
4. Define a common interface:

```python
class NullModel:
    def fit(self, X: np.ndarray, p0: int) -> "NullModel": ...
    def sample(self, T: int, seed: int) -> np.ndarray: ...
```

5. Add calibration output schema:

```json
{
  "config": {},
  "observed": {
    "T_obs": 0.0,
    "D_obs": {},
    "D_parts_obs": {},
    "edge_counts": {}
  },
  "null": {
    "T_boot": [],
    "D_boot": {},
    "critical_90": 0.0,
    "critical_95": 0.0,
    "critical_99": 0.0,
    "p_value": 0.0
  },
  "diagnosis": {
    "reject_global_95": false,
    "first_exceedance_depth": null
  }
}
```

6. Add CLI options:

```text
--null-model var
--bootstrap-kind residual|moving-block|stationary
--critical-levels 0.90 0.95 0.99
--save-surrogate-summaries
--n-jobs
```

7. Add calibration plots:
   - histogram of `T_boot` with `T_obs`
   - `D_p` with pointwise null bands
   - edge count trajectory with surrogate bands

### Acceptance Criteria

- `markov-bootstrap` runs on `order1_unconfounded` with `B=10` in under a few minutes.
- Output JSON contains observed and surrogate summaries.
- The global p-value is reproducible with fixed seed.
- Tests verify that `p_value` is in `[0, 1]`, critical values are ordered, and output keys are stable.

## Phase 2: Automatic Conditioning-Depth Selection

### Goal

Give users an actionable recommended depth `p_star`, not only a plot.

### New Files

```text
hidden-confounding-diagnostics/src/markovianity_diagnostic/experiments/depth_selection.py
hidden-confounding-diagnostics/tests/test_depth_selection.py
```

### Rules to Implement

Implement at least three depth-selection rules:

1. Absolute threshold:

```text
p_star = first p such that D_p <= epsilon for k consecutive transitions
```

2. Relative threshold:

```text
p_star = first p such that D_p <= fraction * max(D_p)
```

3. Bootstrap-band rule:

```text
p_star = first p such that D_p is inside the pointwise null band for all later depths
```

### Output Schema

```json
{
  "method": "c-GC",
  "recording": "fish-1",
  "p_values": [1, 2, 3, 4, 5, 6, 7],
  "selected": {
    "absolute": 4,
    "relative": 3,
    "bootstrap_band": 4
  },
  "warnings": [
    "largest instability occurs at first transition",
    "late-depth additions persist"
  ]
}
```

### Notebook

Create:

```text
hidden-confounding-diagnostics/notebooks/real_data/v2a_depth_selection.ipynb
```

### Acceptance Criteria

- Produces one row per fish and method.
- Handles missing depths.
- Reports when no stable depth is found.

## Phase 3: Mechanism Ablation Simulations

### Goal

Separate latent confounding from other causes of depth instability.

### Existing File to Extend

```text
hidden-confounding-diagnostics/src/markovianity_diagnostic/experiments/simulations.py
```

### New Scenarios

Add these scenario functions:

```python
scenario_omitted_lag_order(...)
scenario_undersampled_markov(...)
scenario_measurement_noise(...)
scenario_time_varying_coefficients(...)
scenario_regime_shift(...)
scenario_nonlinear_markov(...)
scenario_nonlinear_hidden_driver(...)
scenario_observation_filtering(...)
scenario_partial_observation_hidden_nodes_strong(...)
```

### Required Metadata

Every scenario must return metadata:

```json
{
  "scenario": "...",
  "latent": true,
  "true_order": 1,
  "observed_order": null,
  "mechanism": "latent_common_driver|omitted_lag|undersampling|measurement_noise|nonstationarity|nonlinear",
  "expected_signature": "early_deletion|late_instability|flat|power_loss"
}
```

### Notebooks

Create:

```text
hidden-confounding-diagnostics/notebooks/simulations/mechanism_ablations_hidden_state.ipynb
hidden-confounding-diagnostics/notebooks/simulations/mechanism_ablations_observation_artifacts.ipynb
hidden-confounding-diagnostics/notebooks/simulations/mechanism_ablations_nonstationarity.ipynb
```

### Acceptance Criteria

- Each scenario runs through `markov-exp`.
- Each scenario has ground truth when possible.
- Outputs include recovery metrics and graph instability metrics.
- Summary figure compares mechanisms side by side.

## Phase 4: Edgewise Instability Localization

### Goal

Identify which edges drive global graph instability.

### New File

```text
hidden-confounding-diagnostics/src/markovianity_diagnostic/experiments/edge_localization.py
```

### Metrics to Implement

For every edge `(i, j)`:

- number of appearances across depths
- number of deletions
- number of additions
- first depth where edge appears
- first depth where edge disappears
- edgewise instability frequency
- bootstrap null frequency when available
- FDR-adjusted q-value

### Output Columns

```text
recording
method
source
target
n_depths_present
n_deletions
n_additions
first_appearance_p
first_deletion_p
instability_frequency
null_frequency
p_value
q_value
status
```

### Notebook

Create:

```text
hidden-confounding-diagnostics/notebooks/real_data/v2a_edgewise_localization.ipynb
```

### Acceptance Criteria

- Produces edge-level CSV for each fish and method.
- Heatmaps render for additions and deletions.
- If bootstrap outputs are missing, notebook still runs descriptive localization.

## Phase 5: Positive and Negative Controls

### Goal

Show that the diagnostic behaves sensibly in known null and known violation cases.

### New File

```text
hidden-confounding-diagnostics/src/markovianity_diagnostic/experiments/controls.py
```

### Controls to Implement

Synthetic controls:

- order-1 Markov null
- order-3 Markov null
- latent common driver positive control
- hidden nodes positive control

Real-data controls:

- time-shuffled traces
- block-shuffled traces
- phase-randomized traces, if appropriate
- circularly shifted source-target controls

### Notebook

Create:

```text
hidden-confounding-diagnostics/notebooks/controls/positive_negative_controls.ipynb
```

### Acceptance Criteria

- Markovian null controls show low instability after true order.
- Positive controls show stronger instability.
- Real-data shuffled controls do not reproduce the same biological edge structure.

## Phase 6: LPCMCI and SVAR-FCI Comparisons

### Goal

Compare the stability diagnostic with methods that explicitly handle latent confounding.

### New Files

```text
hidden-confounding-diagnostics/src/markovianity_diagnostic/methods/lpcmci_adapter.py
hidden-confounding-diagnostics/src/markovianity_diagnostic/methods/svarfci_adapter.py
hidden-confounding-diagnostics/tests/test_adapters_smoke.py
```

### Implementation Notes

Start with LPCMCI because Tigramite is already installed. The adapter should normalize output to:

```python
{
    "adjacency": np.ndarray,
    "edge_marks": dict,
    "raw_graph": object,
    "metadata": dict,
}
```

For SVAR-FCI, first verify availability in current dependencies. If no reliable implementation exists, document it as a planned extension and do not block the LPCMCI comparison.

### Notebook

Create:

```text
hidden-confounding-diagnostics/notebooks/comparisons/lpcmci_svarfci_comparison.ipynb
```

### Acceptance Criteria

- LPCMCI runs on at least small synthetic scenarios.
- Outputs distinguish directed, bidirected, and uncertain marks where available.
- Comparison table reports whether instability edges overlap with latent-confounding marks.

## Phase 7: Real-Data Confidence Intervals

### Goal

Add uncertainty to real-data instability summaries.

### New File

```text
hidden-confounding-diagnostics/src/markovianity_diagnostic/experiments/real_data.py
```

### Methods

Implement block bootstrap over time:

- contiguous blocks
- moving blocks
- optional stationary bootstrap

For each replicate:

1. resample blocks;
2. rerun learner across depths;
3. compute edge counts, `D_p`, `D_minus`, `D_plus`, `T_obs`;
4. summarize percentile intervals.

### Notebook

This can be grouped with calibration only if it uses the same bootstrap outputs. Prefer a separate notebook if runtime is large:

```text
hidden-confounding-diagnostics/notebooks/real_data/v2a_uncertainty_bands.ipynb
```

### Acceptance Criteria

- Reports 90%, 95%, and 99% intervals.
- Figures show real-data curves with uncertainty bands.
- Warnings are emitted if block length is too short relative to autocorrelation.

## Phase 8: Nonlinear Conditional-Independence Tests

### Goal

Check whether depth sensitivity is robust to dependence-test family.

### New File

```text
hidden-confounding-diagnostics/src/markovianity_diagnostic/experiments/nonlinear_tests.py
```

### Implementation Tasks

1. Add Tigramite adapters for:
   - ParCorr
   - GPDC, if available
   - CMIknn, if available
2. Add graceful skip behavior if optional tests are unavailable.
3. Log runtime and failures.
4. Compare graph instability curves across CI tests.

### Notebook

Create:

```text
hidden-confounding-diagnostics/notebooks/comparisons/nonlinear_ci_tests.ipynb
```

### Acceptance Criteria

- ParCorr baseline runs.
- At least one nonlinear test runs on reduced-size synthetic data.
- Notebook clearly reports runtime constraints.

## Phase 9: Power Analysis

### Goal

Quantify when the diagnostic has enough data to work.

### New File

```text
hidden-confounding-diagnostics/src/markovianity_diagnostic/experiments/power.py
```

### Grid

Vary:

- `T`: 250, 500, 1000, 2000, 5000
- `d`: 5, 10, 20, 50
- edge density
- confounder strength
- latent autocorrelation
- noise scale
- conditioning depth grid
- method

### Metrics

- true positive rate for detecting instability
- false positive rate under Markovian null
- selected `p_star`
- runtime
- memory usage if easy to collect

### Notebook

Create:

```text
hidden-confounding-diagnostics/notebooks/power/sample_size_power_analysis.ipynb
```

### Acceptance Criteria

- Produces power curves.
- Identifies minimal sample sizes for stable behavior under each scenario.
- Reports where deeper conditioning mainly causes power collapse.

## Phase 10: Runtime Scalability

### Goal

Make the diagnostic practical.

### New File

```text
hidden-confounding-diagnostics/src/markovianity_diagnostic/experiments/parallel.py
```

### Implementation Tasks

1. Add joblib or multiprocessing-based parallel execution.
2. Parallelize over:
   - simulation repeats
   - bootstrap replicates
   - fish
   - method
   - conditioning depth where safe
3. Add adjacency caching keyed by:

```text
dataset_hash
method
method_params
p_value
seed
```

4. Add `--cache-dir`, `--resume`, and `--force` flags to CLIs.

### Notebook

Create:

```text
hidden-confounding-diagnostics/notebooks/power/runtime_scaling.ipynb
```

### Acceptance Criteria

- Interrupted runs can resume from cached adjacencies.
- Runtime summary is written to CSV.
- Results are identical with `n_jobs=1` and `n_jobs>1` for fixed seeds.

## Phase 11: Reporting and Manuscript Exports

### Goal

Make the manuscript reproducible from frozen outputs.

### New Files

```text
hidden-confounding-diagnostics/src/markovianity_diagnostic/reporting/tables.py
hidden-confounding-diagnostics/src/markovianity_diagnostic/reporting/figure_specs.py
hidden-confounding-diagnostics/src/markovianity_diagnostic/reporting/manuscript_exports.py
```

### Notebook

Create:

```text
hidden-confounding-diagnostics/notebooks/reporting/nature_methods_figures.ipynb
```

### Outputs

```text
nature_methods/figures/
nature_methods/tables/
outputs/reporting/nature_methods_figure_manifest.json
```

### Acceptance Criteria

- Manuscript figures are regenerated from output files, not from ad hoc notebook state.
- Every figure has a manifest entry listing source data and code path.
- Tables are exported as CSV and LaTeX.

## Suggested Execution Order

### Milestone 1: Calibration and Depth Selection

Implement:

1. Phase 1: hardened bootstrap calibration
2. Phase 2: automatic depth selection
3. `bootstrap_null_synthetic.ipynb`
4. `bootstrap_null_v2a.ipynb`
5. `v2a_depth_selection.ipynb`

Why first:

- This directly strengthens the main Nature Methods claim.

### Milestone 2: Specificity and Failure Modes

Implement:

1. Phase 3: mechanism ablations
2. Phase 5: positive and negative controls
3. Phase 9: power analysis

Why second:

- This addresses the biggest reviewer criticism: instability is not specific to latent confounding.

### Milestone 3: Localization and Method Comparisons

Implement:

1. Phase 4: edgewise localization
2. Phase 6: LPCMCI comparison
3. Phase 8: nonlinear CI tests

Why third:

- This makes the method more interpretable and compares it with expected baselines.

### Milestone 4: Engineering and Reproducibility

Implement:

1. Phase 10: runtime scalability
2. Phase 11: manuscript export pipeline
3. notebook cleanup and standardized manifests

Why last:

- These changes make the project robust once the core science is settled.

## Testing Plan

Add `pytest` as a development dependency if not already available.

Minimum tests:

```text
tests/test_graph_metrics.py
tests/test_bootstrap_calibration.py
tests/test_depth_selection.py
tests/test_edge_localization.py
tests/test_simulation_scenarios.py
tests/test_adapters_smoke.py
```

Test requirements:

- Metrics ignore diagonal edges.
- Instability decomposition satisfies `D_p = D_minus + D_plus`.
- Bootstrap p-values are reproducible with fixed seeds.
- Depth-selection rules handle stable, unstable, and missing-depth cases.
- Every scenario returns `ScenarioResult` with valid metadata.
- Adapters return square binary adjacency matrices with zero diagonals.

## CLI Plan

Extend existing CLIs rather than creating many new ones.

Existing:

```text
markov-exp
markov-bootstrap
```

Add optional subcommands or flags:

```text
markov-exp --scenario mechanism_grid ...
markov-bootstrap --null-model var --bootstrap-kind moving-block ...
markov-depth-select --input adjacency_dir --output-dir ...
markov-edge-localize --input adjacency_dir --bootstrap-json ...
markov-controls --control block-shuffle ...
```

If keeping scripts minimal, implement `markov-depth-select`, `markov-edge-localize`, and `markov-controls` only after the package functions are stable.

## Output Schema Standards

Every analysis output directory should contain:

```text
config.json
manifest.json
results.json
summary.csv
figures/
```

`manifest.json` should include:

```json
{
  "created_at": "ISO timestamp",
  "git_commit": "optional",
  "analysis": "name",
  "input_paths": [],
  "output_paths": [],
  "method": "name",
  "method_params": {},
  "p_values": [],
  "random_seed": 0,
  "software_versions": {}
}
```

## Manuscript Payoff

After this plan is implemented, the manuscript can make stronger claims:

- The diagnostic is calibrated against a specified Markovian null.
- The selected conditioning depth is generated by a reproducible rule.
- Known failure modes are empirically separated.
- Edge-level instability can be localized.
- Real-data results have uncertainty intervals.
- Method comparisons include latent-variable-aware alternatives.
- Figures and tables are reproducible from package outputs.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Bootstrap calibration is slow | Add caching, `n_jobs`, and reduced smoke-test modes |
| VAR null is misspecified | Report it explicitly; add nonlinear or residual bootstrap alternatives |
| Nonlinear CI tests are too slow | Run reduced-size experiments and report runtime limits |
| LPCMCI output is hard to compare with binary adjacency | Preserve raw edge marks and separately collapse to adjacency |
| Real data lacks ground truth | Keep real-data claims descriptive/calibrated, not recovery claims |
| Too many notebooks become unmaintainable | Keep notebooks thin and package-driven; require manifests |

## Definition of Done

The improvement program is complete when:

1. `markov-bootstrap` produces calibrated global and pointwise null outputs.
2. `markov-depth-select` or equivalent package function returns `p_star`.
3. Mechanism ablation notebooks show which mechanisms mimic latent confounding.
4. v2a-RSN real-data outputs include uncertainty bands and depth-selection summaries.
5. Edgewise localization exports edge-level CSVs and heatmaps.
6. LPCMCI comparison runs on at least synthetic data and one reduced v2a recording.
7. Power analysis reports sample-size and node-count regimes.
8. Runtime scaling reports practical limits and cache/resume behavior.
9. Manuscript figures are generated from frozen outputs with manifests.
10. All core package functions have smoke tests.
