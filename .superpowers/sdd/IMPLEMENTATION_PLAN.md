# Nature Methods Improvements: Complete Implementation Plan

## Global Constraints & Principles

### Core Requirements
- **Target venue**: Nature Methods
- **Approach**: All 11 phases must achieve full suite of improvements
- **Code structure**: Python package-first (src/markovianity_diagnostic/)
- **Test methodology**: TDD (test-first, watch fail, minimal code to pass)
- **Output schema**: Standardized JSON/CSV with manifests on all analyses
- **Reproducibility**: All configs logged, seeds fixed, versions recorded

### Testing Standards
- Every new module has pytest tests
- Tests verify behavior, not implementation
- Fixtures for synthetic data in conftest.py
- Smoke tests for all scenarios
- Edge cases covered (empty input, missing depths, None values)

### Notebook Standards
- Notebooks call reusable package functions
- Do not duplicate analysis logic in notebooks
- Each notebook answers one scientific question
- Standardized output: config.json, manifest.json, results.json, summary.csv, figures/
- Every major claim has machine-readable artifact

### Manifest Schema (All Outputs)
```json
{
  "created_at": "ISO timestamp",
  "git_commit": "short hash",
  "analysis": "phase name",
  "input_paths": [],
  "output_paths": [],
  "method": "name",
  "method_params": {},
  "p_values": [],
  "random_seed": 42,
  "software_versions": {
    "python": "3.x",
    "tigramite": "x.x",
    "numpy": "x.x"
  }
}
```

---

## Milestone 1: Calibration and Depth Selection (Phases 1-2)

### Phase 1: Hardened Surrogate-Null Bootstrap Calibration

**Goal**: Turn bootstrap prototype into robust calibration module for T_obs = max_{p > p0} D_p

**Implementation Tasks**:

#### Task 1.1: Create calibration.py Module with Base Classes
- File: `src/markovianity_diagnostic/experiments/calibration.py`
- Define `NullModel` abstract base class:
  - Method: `fit(X: np.ndarray, p0: int) -> NullModel`
  - Method: `sample(T: int, seed: int) -> np.ndarray`
  - Property: `metadata: dict`
- Create test file: `tests/test_bootstrap_calibration.py`
- Tests:
  - NullModel.fit() returns self
  - NullModel.sample() returns correct shape (T, d)
  - Reproducibility with fixed seed

#### Task 1.2: Implement Null Model Classes
- Classes to implement in `calibration.py`:
  - `VARNullModel(NullModel)`: Fit VAR(p0), sample from it
  - `ResidualBootstrapNull(NullModel)`: Fit VAR, resample residuals with replacement
  - `MovingBlockBootstrapNull(NullModel)`: Fit VAR, resample blocks of residuals
  - Optional: `StationaryBootstrapNull(NullModel)`
- Tests for each:
  - fit() accepts X shape (T, d) with T >= p0
  - sample() produces stationary-looking output
  - Metadata reports parameters correctly

#### Task 1.3: Add Calibration Output Schema
- Class: `CalibrationResult` dataclass
  - observed: {T_obs, D_obs, D_parts_obs, edge_counts}
  - null: {T_boot, D_boot, critical_90, critical_95, critical_99, p_value}
  - diagnosis: {reject_global_95, first_exceedance_depth}
- Method: `to_json(path: str)` and `from_json(path: str)`
- Tests:
  - Schema serializes to valid JSON
  - Deserialize round-trip is lossless
  - All critical values are ordered (90 < 95 < 99)
  - p_value ∈ [0, 1]

#### Task 1.4: Refactor bootstrap.py and Update CLI
- Move general functions from `bootstrap.py` → `calibration.py`
- Keep `bootstrap.py` as compatibility wrapper (or remove, update imports)
- Add CLI options to `cli/bootstrap.py`:
  - `--null-model {var, residual, moving-block, stationary}`
  - `--critical-levels 0.90 0.95 0.99`
  - `--save-surrogate-summaries`
  - `--n-jobs` (default 1, optional parallelization)
- Tests: CLI produces valid output JSON with all required keys

#### Task 1.5: Add Calibration Plotting Functions
- Function: `plot_T_boot_histogram(T_boot, T_obs, critical_values, output_path)`
- Function: `plot_D_p_with_bands(D_p, D_boot_pointwise, p_values, output_path)`
- Function: `plot_edge_count_trajectory(edge_counts, edge_counts_boot, output_path)`
- Tests: Output files exist and are valid PNG

#### Task 1.6: Create Bootstrap Null Synthetic Notebook
- File: `notebooks/calibration/bootstrap_null_synthetic.ipynb`
- Scenarios:
  - order1_unconfounded: Markovian null
  - order1_with_latent_confounder: Positive control
  - order3_unconfounded: Misspecified null
- For each:
  - Run calibration with B=50 bootstrap replicates
  - Export: bootstrap_summary.csv, bootstrap_results.json, bootstrap_T_obs.png
  - Verify: p-value ∈ [0, 1], critical values ordered
- Output directory: `outputs/calibration/synthetic/`

#### Task 1.7: Create Bootstrap Null V2A Notebook
- File: `notebooks/calibration/bootstrap_null_v2a.ipynb`
- Load v2a-RSN c-GC outputs
- Run block bootstrap over time (block length 50)
- For each fish and method:
  - Compute null T_boot distribution
  - Compute pointwise null bands
  - Export results and depth bands plot
- Output directory: `outputs/calibration/v2a/`
- Acceptance: Figures render, CSV has correct rows per fish

**Acceptance Criteria**:
- `markov-bootstrap` runs on `order1_unconfounded` with B=10 in <5 minutes
- Output JSON contains all required keys
- p_value reproducible with fixed seed
- Tests verify output schema stability

---

### Phase 2: Automatic Conditioning-Depth Selection

**Goal**: Provide actionable recommended depth p_star

**Implementation Tasks**:

#### Task 2.1: Create depth_selection.py Module
- File: `src/markovianity_diagnostic/experiments/depth_selection.py`
- Class: `DepthSelector` with three selection rules
  - Rule 1 (absolute): `select_absolute(D_p, epsilon=0.1, k_stable=2) -> int`
  - Rule 2 (relative): `select_relative(D_p, fraction=0.1) -> int`
  - Rule 3 (bootstrap-band): `select_bootstrap_band(D_p, D_boot_pointwise, confidence=0.95) -> int`
- Output class: `DepthSelectionResult`
  - p_values: [1, 2, 3, ...]
  - selected: {absolute: 4, relative: 3, bootstrap_band: 4}
  - warnings: ["largest instability at first transition", ...]
- Tests:
  - Each rule returns valid depth or None
  - Handles missing depths gracefully
  - Warns when no stable depth found
  - Warnings are informative

#### Task 2.2: Create v2a_depth_selection Notebook
- File: `notebooks/real_data/v2a_depth_selection.ipynb`
- Load v2a-RSN c-GC and c-GC* outputs (all recordings)
- For each fish, method combination:
  - Apply all three selection rules
  - Store results
- Outputs:
  - depth_selection_summary.csv (one row per fish+method)
  - depth_selection.json (detailed results)
  - depth_selection_plot.png (depth curves with rules highlighted)
- Output directory: `outputs/v2a-RSNs/depth_selection/`
- Acceptance: CSV has correct row count (n_fish × 2)

**Acceptance Criteria**:
- Produces one row per fish and method
- Handles missing depths without error
- Reports when no stable depth found
- Three rules produce sensible outputs

---

## Milestone 2: Specificity and Failure Modes (Phases 3, 5, 9)

### Phase 3: Mechanism Ablation Simulations

**Goal**: Separate latent confounding from other causes of depth instability

**Implementation Tasks**:

#### Task 3.1: Extend simulations.py with Mechanism Scenarios
- File: `src/markovianity_diagnostic/experiments/simulations.py` (extend)
- New scenario functions to add:
  - `scenario_omitted_lag_order(d=5, T=1000, true_order=1, observed_order=0)`: VAR(1) observed as VAR(0)
  - `scenario_undersampled_markov(d=5, T=1000, true_dt=10, observed_dt=100)`: Sample every Nth time step
  - `scenario_measurement_noise(d=5, T=1000, noise_scale=0.5)`: Add Gaussian noise
  - `scenario_time_varying_coefficients(d=5, T=1000)`: Coefficients vary with time
  - `scenario_regime_shift(d=5, T=1000, shift_point=500)`: Coefficients change mid-series
  - `scenario_nonlinear_markov(d=5, T=1000)`: Nonlinear state transitions
  - `scenario_nonlinear_hidden_driver(d=5, T=1000)`: Nonlinear latent confounder
  - `scenario_observation_filtering(d=5, T=1000)`: Low-pass filter on observations
  - `scenario_partial_observation_hidden_nodes_strong(d=5, T=1000, hidden_d=2, strength=1.0)`: Strong hidden nodes

- Every scenario returns: `ScenarioResult`
  - X: (T, d) observed data
  - ground_truth_adj: (d, d) true adjacency matrix (where available)
  - metadata: dict with scenario, latent (bool), true_order, observed_order, mechanism, expected_signature

- Tests for each scenario:
  - Returns correct shape
  - Metadata is valid
  - Data is not all zeros/NaNs
  - Deterministic with fixed seed

#### Task 3.2: Create Mechanism Ablation Notebooks
- File 1: `notebooks/simulations/mechanism_ablations_hidden_state.ipynb`
  - Scenarios: latent_confounder, hidden_nodes, omitted_lag_order
  - Run c-GC, c-GC* across depths
  - Compute: recovery metrics, D_p trajectories, graph metrics
  - Output: results.csv, summary.json, figures/

- File 2: `notebooks/simulations/mechanism_ablations_observation_artifacts.ipynb`
  - Scenarios: measurement_noise, undersampling
  - Same analysis as above
  - Output: results.csv, summary.json, figures/

- File 3: `notebooks/simulations/mechanism_ablations_nonstationarity.ipynb`
  - Scenarios: time_varying_coefficients, regime_shift
  - Stationarity diagnostics included
  - Output: results.csv, summary.json, figures/

- Output directories: `outputs/simulations/mechanism_ablations/{hidden_state, observation_artifacts, nonstationarity}/`
- Acceptance: Summary figures show mechanism comparison

**Acceptance Criteria**:
- Each scenario runs through `markov-exp` without error
- Recovery metrics computed where applicable
- Graph instability metrics match expected signatures
- Summary figures compare mechanisms side by side

---

### Phase 5: Positive and Negative Controls

**Goal**: Show diagnostic behaves sensibly under known null and violation cases

**Implementation Tasks**:

#### Task 5.1: Create controls.py Module
- File: `src/markovianity_diagnostic/experiments/controls.py`
- Classes and functions:
  - `SyntheticMarkovianNull(p=1)`: order-p Markov with known ground truth
  - `SyntheticLatentControlPositive()`: latent confounder scenario
  - `SyntheticHiddenNodesPositive()`: hidden nodes scenario
  - `TimeShuffledControl(X)`: Shuffle time dimension, break temporal structure
  - `BlockShuffledControl(X, block_size=50)`: Shuffle blocks of time
  - `PhaseRandomizedControl(X)`: Phase randomization for time series
  - `CircularlyShiftedControl(X, lag=1)`: Circular shift for control edges

- Tests:
  - Null controls return valid data
  - Shuffled controls preserve marginal distributions but break structure
  - All return same shape as input

#### Task 5.2: Create positive_negative_controls Notebook
- File: `notebooks/controls/positive_negative_controls.ipynb`
- Synthetic controls (N=10 repeats each):
  - Markov order-1 null
  - Markov order-3 null
  - Latent confounder positive control
  - Hidden nodes positive control
- Real-data controls (v2a-RSN c-GC):
  - Time-shuffled traces
  - Block-shuffled traces
  - Phase-randomized traces
  - Circularly shifted source-target controls
- For each control:
  - Run learner across depths
  - Compute D_p, edge counts, recovery metrics
  - Compare with biological baseline
- Outputs: control_summary.csv, control_results.json, control_curves.png
- Output directory: `outputs/controls/`
- Acceptance: Markovian nulls show low instability; positive controls show higher instability

**Acceptance Criteria**:
- Null controls show low instability after true order
- Positive controls show stronger instability
- Shuffled controls do not reproduce biological edge structure

---

### Phase 9: Power Analysis

**Goal**: Quantify when diagnostic has enough data

**Implementation Tasks**:

#### Task 9.1: Create power.py Module
- File: `src/markovianity_diagnostic/experiments/power.py`
- Grid parameters:
  - T: [250, 500, 1000, 2000, 5000]
  - d: [5, 10, 20, 50]
  - edge_density: [0.2, 0.5, 0.8]
  - confounder_strength: [0.1, 0.5, 1.0]
  - latent_autocorr: [0.5, 0.9]
  - noise_scale: [0.1, 1.0]
  - p_grid_max: [5, 10]
  - method: ['c-GC', 'c-GC-star']
  - n_repeats: 20

- Metrics to compute per grid cell:
  - True positive rate (TPR) for detecting instability
  - False positive rate (FPR) under Markovian null
  - Selected p_star across runs
  - Runtime per configuration
  - Memory usage (if feasible)

- Tests:
  - Grid produces valid results
  - TPR and FPR in [0, 1]
  - p_star is reasonable

#### Task 9.2: Create sample_size_power_analysis Notebook
- File: `notebooks/power/sample_size_power_analysis.ipynb`
- Run reduced grid (subset of full power.py grid) for feasibility
- Grid: T in [500, 1000, 2000], d in [5, 10, 20], 10 repeats each
- Outputs: power_curves.png, results.csv, summary statistics
- Output directory: `outputs/power/sample_size_power/`
- Acceptance: Produces power curves; identifies minimal sample sizes

**Acceptance Criteria**:
- Power curves show expected trends (higher T, higher power)
- Identifies regimes where depth conditioning breaks power
- Runtime reasonable (can run in parallel)

---

## Milestone 3: Localization and Method Comparisons (Phases 4, 6, 8)

### Phase 4: Edgewise Instability Localization

**Goal**: Identify which edges drive global graph instability

**Implementation Tasks**:

#### Task 4.1: Create edge_localization.py Module
- File: `src/markovianity_diagnostic/experiments/edge_localization.py`
- Function: `compute_edge_instability_metrics(adj_dict: Dict[int, ndarray], bootstrap_adj_dict=None) -> pd.DataFrame`
- For every edge (i, j):
  - n_depths_present: count appearances across p_values
  - n_deletions: count times edge disappears
  - n_additions: count times edge appears
  - first_appearance_p: first depth where (i,j) appears
  - first_deletion_p: first depth where (i,j) disappears
  - instability_frequency: n_additions + n_deletions
  - null_frequency: frequency in bootstrap samples (if available)
  - p_value: empirical p-value
  - q_value: FDR-adjusted (if null_frequency available)
  - status: categorical {persistent, unstable, late_addition}

- Output columns: recording, method, source, target, n_depths_present, n_deletions, n_additions, first_appearance_p, first_deletion_p, instability_frequency, null_frequency, p_value, q_value, status

- Tests:
  - Output has correct columns
  - Diagonal edges excluded
  - Metrics are non-negative
  - FDR q_values are monotone ordered

#### Task 4.2: Create v2a_edgewise_localization Notebook
- File: `notebooks/real_data/v2a_edgewise_localization.ipynb`
- Load v2a-RSN c-GC and c-GC* outputs
- For each recording and method:
  - Compute edge-level instability metrics
  - Create heatmaps: additions (n_additions per edge per depth), deletions (n_deletions per edge per depth)
  - Export edge_instability.csv, edge_fdr.csv (if bootstrap available)
- Outputs: edge_instability.csv, edge_fdr.csv, deletion_heatmap.png, addition_heatmap.png
- Output directory: `outputs/v2a-RSNs/edge_localization/`
- Acceptance: Heatmaps render; CSV rows match expected counts

**Acceptance Criteria**:
- Edge-level CSV for each fish and method
- Heatmaps render correctly
- Runs even if bootstrap outputs missing (descriptive localization only)

---

### Phase 6: LPCMCI and SVAR-FCI Comparisons

**Goal**: Compare stability diagnostic with methods that explicitly handle latent confounding

**Implementation Tasks**:

#### Task 6.1: Create lpcmci_adapter.py
- File: `src/markovianity_diagnostic/methods/lpcmci_adapter.py`
- Class: `LPCMCIAdapter`
  - `__init__(tau_min=1, tau_max=5, pc_alpha=0.05)`
  - `fit(X, p_values_to_test: List[int])` → runs LPCMCI at each p, records PAG
  - Output: Dict[int, {adjacency, edge_marks, raw_pag, metadata}]
  
- Normalize output to standard schema:
  ```python
  {
    "adjacency": np.ndarray (binary, directed edges),
    "edge_marks": dict (preserves bidirected, uncertain marks),
    "raw_graph": object (PAG for inspection),
    "metadata": {algorithm, params, p_value, runtime}
  }
  ```

- Tests:
  - fit() accepts valid X
  - Output schema matches standard
  - Adjacency is square binary matrix
  - Diagonal is zero

#### Task 6.2: Create svarfci_adapter.py Placeholder
- File: `src/markovianity_diagnostic/methods/svarfci_adapter.py`
- Document: Implementation is deferred. Placeholder class with error on use.
- Notes: Availability and stability of SVAR-FCI needs verification; LPCMCI sufficient for first submission.
- No tests required (placeholder only)

#### Task 6.3: Create lpcmci_svarfci_comparison Notebook
- File: `notebooks/comparisons/lpcmci_svarfci_comparison.ipynb`
- Compare on:
  - Small synthetic data (order1_unconfounded, order1_with_latent_confounder, latent_nodes)
  - One reduced v2a-RSN recording (if runtime permits)
- For each scenario/recording:
  - Run c-GC (reference)
  - Run c-GC-star (reference)
  - Run LPCMCI
  - Compare edge structure
  - Table: edges (overlap with instability, overlap with latent marks, unique to method)
- Outputs: summary.csv, pag_edge_marks.csv, comparison_figures/
- Output directory: `outputs/comparisons/lpcmci_svarfci/`
- Acceptance: LPCMCI runs; output table compares methods

**Acceptance Criteria**:
- LPCMCI runs on at least small synthetic scenarios
- Output distinguishes directed vs. bidirected vs. uncertain marks
- Comparison table reports instability-latent mark overlap

---

### Phase 8: Nonlinear Conditional-Independence Tests

**Goal**: Check if depth sensitivity is robust to CI test family

**Implementation Tasks**:

#### Task 8.1: Create nonlinear_tests.py Module
- File: `src/markovianity_diagnostic/experiments/nonlinear_tests.py`
- Adapters for Tigramite tests:
  - `ParCorrAdapter`: ParCorr (baseline linear)
  - `GPDCAdapter`: GPDC if available
  - `CMIknnAdapter`: CMIknn if available
  - Fallback: Log warning, skip unavailable tests

- Function: `run_ci_test_comparison(X, ci_test_names, p_values, method='c-GC') -> Dict[test, results]`
- Tests:
  - At least ParCorr baseline runs
  - Graceful skip for unavailable tests
  - Runtime per test logged

#### Task 8.2: Create nonlinear_ci_tests Notebook
- File: `notebooks/comparisons/nonlinear_ci_tests.ipynb`
- Run on reduced-size synthetic data (d=5, T=500) due to runtime constraints
- CI tests: ParCorr, GPDC (if available)
- Scenarios: order1_unconfounded, order1_with_latent_confounder
- Outputs: D_p curves per CI test, runtime.csv
- Output directory: `outputs/comparisons/nonlinear_ci_tests/`
- Acceptance: ParCorr baseline runs; at least one nonlinear test runs; runtime documented

**Acceptance Criteria**:
- ParCorr baseline runs without error
- At least one nonlinear test runs on reduced data
- Notebook clearly reports runtime constraints

---

## Milestone 4: Engineering and Reproducibility (Phases 10-11)

### Phase 10: Runtime Scalability

**Goal**: Make diagnostic practical

**Implementation Tasks**:

#### Task 10.1: Create parallel.py Module
- File: `src/markovianity_diagnostic/experiments/parallel.py`
- Functions:
  - `parallel_simulation_grid(scenarios, methods, p_values, n_repeats, n_jobs=4) -> results`
  - `parallel_bootstrap(X, null_model, B, n_jobs=4) -> T_boot`
  - `parallel_over_fish(fish_list, analysis_func, n_jobs=4) -> results`

- Caching layer:
  - Cache key: `hash(dataset) + method + params + p_value + seed`
  - Store: adjacency matrices, edge counts, D_p values
  - Flag: `--cache-dir`, `--resume`, `--force` on CLIs

- Tests:
  - n_jobs=1 and n_jobs>1 produce identical results (fixed seed)
  - Cache is created and loaded correctly
  - Resume skips cached tasks

#### Task 10.2: Create runtime_scaling Notebook
- File: `notebooks/power/runtime_scaling.ipynb`
- Measure runtime for:
  - Varying T: [500, 1000, 2000, 5000]
  - Varying d: [5, 10, 20, 50]
  - Methods: c-GC, c-GC-star
  - p_grid_max: [5, 10]
  - n_jobs: [1, 4]

- Outputs: runtime_grid.csv (columns: T, d, method, p_grid_max, n_jobs, runtime_seconds)
  - runtime_plot.png: Runtime vs. T and d
  - Report practical limits and optimal n_jobs

- Output directory: `outputs/power/runtime_scaling/`
- Acceptance: Runtime grid complete; plots render

**Acceptance Criteria**:
- Interrupted runs can resume from cache
- Runtime summary written to CSV
- Results identical with n_jobs=1 and n_jobs>1 (fixed seeds)

---

### Phase 11: Reporting and Manuscript Exports

**Goal**: Make manuscript reproducible from frozen outputs

**Implementation Tasks**:

#### Task 11.1: Create reporting/tables.py and reporting/figure_specs.py
- File: `src/markovianity_diagnostic/reporting/tables.py`
  - Function: `export_depth_selection_table(depth_results) -> pd.DataFrame`
  - Function: `export_calibration_table(calibration_results) -> pd.DataFrame`
  - Function: `export_comparison_table(lpcmci_results) -> pd.DataFrame`

- File: `src/markovianity_diagnostic/reporting/figure_specs.py`
  - Dict: `FIGURE_SPECS` with figure names, input paths, output paths, caption templates

#### Task 11.2: Create reporting/manuscript_exports.py
- File: `src/markovianity_diagnostic/reporting/manuscript_exports.py`
- Functions:
  - `generate_all_figures(output_dir) -> manifest`
  - `generate_all_tables(output_dir) -> manifest`
  - `create_manuscript_manifest(figures_manifest, tables_manifest) -> manifest.json`

- Manifest format:
  ```json
  {
    "figures": [
      {
        "name": "Figure1_Calibration",
        "source_data": ["outputs/calibration/v2a/bootstrap_T_obs.png"],
        "code_path": "reporting/figure_specs.py",
        "caption": "..."
      }
    ],
    "tables": [...]
  }
  ```

#### Task 11.3: Create nature_methods_figures Notebook
- File: `notebooks/reporting/nature_methods_figures.ipynb`
- Load frozen outputs from all earlier phases
- Regenerate all manuscript figures from output files (not from ad hoc state)
- Export as high-resolution PNG/PDF
- Generate LaTeX table code
- Create nature_methods_figure_manifest.json
- Output directories:
  - nature_methods/figures/
  - nature_methods/tables/
  - outputs/reporting/

- Acceptance: All figures regenerated from output files; manifest complete

**Acceptance Criteria**:
- Manuscript figures regenerated from output files, not ad hoc computation
- Every figure has manifest entry with source data and code path
- Tables exported as CSV and LaTeX

---

## Setup and Integration Tasks

### Setup Task: Create tests/ Directory Structure
- Create: `tests/` directory
- Create: `tests/__init__.py`
- Create: `tests/conftest.py` with:
  - Fixture: `synthetic_markov_data()` - Returns order-1 Markov (T=200, d=5)
  - Fixture: `synthetic_data_with_confounder()` - Returns data with latent confounder
  - Fixture: `tmp_outputs()` - Returns temporary directory for test outputs
  - Fixture: `seed_fix(seed=42)` - Sets all seeds for reproducibility

- Test files to create (will be populated by implementers):
  - `tests/test_graph_metrics.py`
  - `tests/test_bootstrap_calibration.py`
  - `tests/test_depth_selection.py`
  - `tests/test_edge_localization.py`
  - `tests/test_simulation_scenarios.py`
  - `tests/test_adapters_smoke.py`

### Setup Task: Verify Dependencies
- Check: pytest installed
- Check: Tigramite available
- Check: numpy, scipy, pandas, matplotlib present
- Add pytest to dev dependencies if needed

### Integration Task: Update v2a-RSN Notebooks
- Notebooks: `notebooks/v2a-RSNs/c-GC/c-GC-v2a_RSN.ipynb`, `c-GC-star/c-GC-star-v2a_RSN.ipynb`
- Required changes:
  - Export adjacency matrices (one CSV per recording-depth pair)
  - Export metrics: edge_counts, D_p, D_minus, D_plus, T_obs, cumulative instability
  - Add final cell writing manifest.json with:
    - input_files, method_settings, depth_grid, output_paths
  - Do NOT add bootstrap, power, nonlinear tests, or method comparisons

### Integration Task: Update Simulation Notebooks
- Notebooks: `notebooks/simulations/{c-GC,c-GC-star,pcmciplus,jpcmciplus,lpcmci,fullci}/*.ipynb`
- Required changes:
  - Standardize output schema (config.json, manifest.json, results.json, summary.csv)
  - Add compact ground truth and inferred adjacency dicts
  - Keep each notebook focused on its method and scenario family
  - Add final manifest cell

---

## Testing Strategy

### Minimum Pytest Coverage

```
tests/
├── conftest.py
├── test_graph_metrics.py
│   ├── test_D_p_nonnegative
│   ├── test_D_p_diagonal_ignored
│   ├── test_instability_decomposition
│   └── test_edge_counts_nonzero_only
├── test_bootstrap_calibration.py
│   ├── test_null_model_interface
│   ├── test_bootstrap_reproducibility_seed
│   ├── test_critical_values_ordered
│   └── test_p_value_in_range
├── test_depth_selection.py
│   ├── test_absolute_rule_returns_valid_depth
│   ├── test_relative_rule_handles_missing_depths
│   ├── test_bootstrap_band_rule_warnings
│   └── test_no_stable_depth_case
├── test_edge_localization.py
│   ├── test_edge_metrics_shape
│   ├── test_diagonal_excluded
│   └── test_fdr_monotonicity
├── test_simulation_scenarios.py
│   ├── test_omitted_lag_order_shape
│   ├── test_latent_confounder_has_metadata
│   ├── test_nonlinear_deterministic_seed
│   └── test_all_scenarios_no_nans
└── test_adapters_smoke.py
    ├── test_lpcmci_adapter_output_schema
    ├── test_c_gc_adapter_adjacency_binary
    └── test_adapters_diagonal_zero
```

### Test Execution
- Run before each merge: `pytest tests/ -v`
- Coverage target: >80% for core experiments modules
- Smoke tests: All scenarios return valid ScenarioResult

---

## Output Directory Structure

```
outputs/
├── calibration/
│   ├── synthetic/
│   │   ├── bootstrap_summary.csv
│   │   ├── bootstrap_results.json
│   │   ├── bootstrap_T_obs.png
│   │   ├── config.json
│   │   └── manifest.json
│   └── v2a/
│       ├── bootstrap_summary.csv
│       ├── bootstrap_results.json
│       ├── bootstrap_T_obs.png
│       ├── depth_bands.png
│       ├── config.json
│       └── manifest.json
├── v2a-RSNs/
│   ├── depth_selection/
│   │   ├── depth_selection_summary.csv
│   │   ├── depth_selection.json
│   │   ├── depth_selection_plot.png
│   │   └── manifest.json
│   └── edge_localization/
│       ├── edge_instability.csv
│       ├── edge_fdr.csv
│       ├── deletion_heatmap.png
│       ├── addition_heatmap.png
│       └── manifest.json
├── simulations/
│   └── mechanism_ablations/
│       ├── hidden_state/
│       │   ├── results.csv
│       │   ├── summary.json
│       │   ├── figures/
│       │   └── manifest.json
│       ├── observation_artifacts/
│       └── nonstationarity/
├── controls/
│   ├── control_summary.csv
│   ├── control_results.json
│   ├── control_curves.png
│   └── manifest.json
├── comparisons/
│   ├── lpcmci_svarfci/
│   │   ├── summary.csv
│   │   ├── pag_edge_marks.csv
│   │   ├── comparison_figures/
│   │   └── manifest.json
│   └── nonlinear_ci_tests/
│       ├── results.csv
│       ├── runtime.csv
│       ├── figures/
│       └── manifest.json
├── power/
│   ├── sample_size_power/
│   │   ├── results.csv
│   │   ├── power_curves.png
│   │   └── manifest.json
│   └── runtime_scaling/
│       ├── runtime_grid.csv
│       ├── runtime_plot.png
│       └── manifest.json
└── reporting/
    └── nature_methods_figure_manifest.json

nature_methods/
├── figures/
│   ├── Figure1_Calibration.png
│   ├── Figure2_DepthSelection.png
│   └── ...
└── tables/
    ├── Table1_MethodComparison.csv
    └── ...
```

---

## Execution Order and Dependencies

### Milestone 1 (Phases 1-2)
1. Phase 1.1-1.4: Calibration module (can parallelize 1.1, 1.2 after 1.1, then 1.3, 1.4)
2. Phase 1.5-1.7: Calibration notebooks (depend on Phase 1.1-1.4 complete)
3. Phase 2.1-2.2: Depth selection (depend on Phase 1 complete for bootstrap outputs)

### Milestone 2 (Phases 3, 5, 9)
- Phase 3 (mechanism ablations): Can start in parallel with Milestone 1
- Phase 5 (controls): Can start in parallel, depends only on existing infrastructure
- Phase 9 (power analysis): Can start in parallel, depends only on existing infrastructure

### Milestone 3 (Phases 4, 6, 8)
- Phase 4 (edgewise localization): Depends on Phase 1 complete (for bootstrap frequency)
- Phase 6 (method comparisons): Can start in parallel with Phase 4
- Phase 8 (nonlinear tests): Can start in parallel with Phases 4 and 6

### Milestone 4 (Phases 10-11)
- Phase 10 (parallelization): Can be implemented in parallel with earlier phases
- Phase 11 (reporting): Final phase, depends on all others complete for outputs

---

## Definition of Done

1. ✅ `markov-bootstrap` produces calibrated global and pointwise null outputs
2. ✅ `markov-depth-select` (or package function) returns p_star
3. ✅ Mechanism ablation notebooks show which mechanisms mimic latent confounding
4. ✅ v2a-RSN outputs include uncertainty bands and depth-selection summaries
5. ✅ Edgewise localization exports edge-level CSVs and heatmaps
6. ✅ LPCMCI comparison runs on at least synthetic data
7. ✅ Power analysis reports sample-size and node-count regimes
8. ✅ Runtime scaling reports practical limits and cache/resume
9. ✅ Manuscript figures generated from frozen outputs with manifests
10. ✅ All core package functions have smoke tests

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Bootstrap calibration is slow | Add caching, `n_jobs`, reduced smoke-test modes |
| VAR null is misspecified | Report explicitly; add residual bootstrap alternatives |
| Nonlinear CI tests too slow | Run reduced-size data, document runtime limits |
| LPCMCI hard to compare | Preserve raw edge marks, separately collapse to adjacency |
| Real data lacks ground truth | Keep claims descriptive/calibrated, not recovery claims |
| Too many notebooks unmaintainable | Keep thin, package-driven, require manifests |
| Context compaction loses progress | Use progress ledger in .superpowers/sdd/ |
| Parallel tasks conflict | Separate by namespace, use file locks for caching |

