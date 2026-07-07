# Execution Order for the New Experiments

## Verdict

This is the recommended order to run the new experiments. The previous ordering was close, but it was missing the baseline real-data adjacency export step, used legacy paths for the new real-data notebooks, and omitted the nonlinear conditional-independence sensitivity notebook.

Run commands from the `hidden-confounding-diagnostics/` repository root unless a notebook states otherwise.

## Preflight

Before launching the full experiment stack, run the test suite or at least the relevant smoke tests:

```bash
uv run pytest -q
```

This checks that the shared loaders, synthetic generators, diagnostics, and notebook support modules are importable before long-running notebooks start. If this fails, fix the code first; otherwise downstream notebook failures will be harder to interpret.

Also check whether existing outputs are final or scaffold outputs from earlier review work. If an output manifest says `pending_full_rerun` or was generated with smoke-test settings, rerun the corresponding notebook before using it in manuscript figures.

## Phase 0: Baseline Real-Data Inputs

These notebooks produce the real v2a adjacency dictionaries and summaries that later real-data analyses consume. The old aggregate/template notebook names `notebooks/v2a-RSNs/c-GC/c-GC-v2a_RSN.ipynb` and `notebooks/v2a-RSNs/c-GC-star/c-GC-star-v2a_RSN.ipynb` are not present in the current repository. They were superseded by one notebook per recording so long runs can be stopped, resumed, or parallelized independently.

Run all four recordings with the common `n25-e9-r16` analysis profile. Each recording uses a deterministic recording-specific random seed to sample 9 emitters and 16 receivers without replacement. This is the closest integer allocation to the requested 35%/65% split for 25 traces (36%/64%). The original cell indices, seed, and realized fractions are written to `run_metadata.json`. All connectivity analyses use conditioning depths `p=1,...,7`.

| Step | Notebook | Depends on | What it does | Why it is required / result contribution |
| --- | --- | --- | --- | --- |
| 0.1 | `notebooks/v2a-RSNs/c-GC/220119_F2_run11.ipynb`<br>`notebooks/v2a-RSNs/c-GC/220127_F4_run2.ipynb`<br>`notebooks/v2a-RSNs/c-GC/220210_F1_run6.ipynb`<br>`notebooks/v2a-RSNs/c-GC/220210_F2_run5.ipynb` | Raw v2a RSN data and c-GC code | Runs c-GC separately for all four recordings using the same seeded 9-emitter/16-receiver sample and `p=1,...,7`. Each notebook checkpoints after every completed depth and records selection provenance. | Provides the observed real-data c-GC baseline for depth selection, calibration, and edge localization. |
| 0.2 | `notebooks/v2a-RSNs/c-GC-star/220119_F2_run11.ipynb`<br>`notebooks/v2a-RSNs/c-GC-star/220127_F4_run2.ipynb`<br>`notebooks/v2a-RSNs/c-GC-star/220210_F1_run6.ipynb`<br>`notebooks/v2a-RSNs/c-GC-star/220210_F2_run5.ipynb` | Raw v2a RSN data and c-GC-star code | Runs c-GC-star separately for all four recordings with the identical selected traces and depth grid. | Supplies the matched c-GC-star outputs used for method comparison and downstream analyses. |
Expected outputs are isolated under `outputs/v2a-RSNs/n25-e9-r16/c-GC/` and `outputs/v2a-RSNs/n25-e9-r16/c-GC-star/`. This prevents the restart logic from reusing full-recording pickles or prior 50-trace profile outputs. Each method directory contains `{recording}.pkl`, per-recording `run_metadata.json`, and method-level summary/transition tables.

## Phase 1: Calibration Foundation

Calibration should run before interpreting real-data depth or edgewise findings as evidence against a calibrated null.

| Step | Notebook | Depends on | What it does | Why it is required / result contribution |
| --- | --- | --- | --- | --- |
| 1.1 (parallel) | `notebooks/calibration/bootstrap_null_synthetic_c-GC.ipynb`<br>`notebooks/calibration/bootstrap_null_synthetic_c-GC-star.ipynb` | Synthetic data generators and the corresponding c-GC variant | Builds checkpointed bootstrap/null reference distributions for each eligible method over `p=1,...,7`. The notebooks use disjoint method result and checkpoint directories and may run concurrently. `B` is a cumulative target: increasing it reuses compatible replicates and computes only the missing indices. | Validates that the calibration procedure behaves sensibly before applying it to empirical data. It contributes the null-calibration evidence needed for method credibility. |
| 1.2 (parallel) | `notebooks/calibration/bootstrap_null_v2a_220119_F2_run11.ipynb`<br>`notebooks/calibration/bootstrap_null_v2a_220127_F4_run2.ipynb`<br>`notebooks/calibration/bootstrap_null_v2a_220210_F1_run6.ipynb`<br>`notebooks/calibration/bootstrap_null_v2a_220210_F2_run5.ipynb` | Complete Phase 0 adjacency pickles and original filtered traces | Each notebook loads one recording's observed adjacency grids for c-GC and c-GC-star directly from `{recording}.pkl`, fits the moving-block null to the raw traces, and reruns only that learner on checkpointed surrogate series for `p=1,...,7`. The four notebooks have disjoint recording-level output and checkpoint directories, so they may run concurrently. | Converts observed graph instability into recording- and method-specific global thresholds and pointwise depth bands. Per-replicate checkpoints make each long surrogate inference resumable and provide adjacency replicates for later edge-localization analysis. |

Expected outputs include calibration artifacts under `outputs/calibration/synthetic/c-GC-family/` and `outputs/calibration/v2a/n25-e9-r16/`.

The method-specific `notebooks/simulations/*/extension_metrics.ipynb`
notebooks are not prerequisites for synthetic bootstrap calibration. The two
bootstrap notebooks generate their own controlled samples and observed graph
grids. Extension-metrics notebooks instead preserve repeated trial data,
ground truth and per-depth adjacencies for later depth-selection,
edge-localization, power and reporting analyses. Run them only when those
frozen per-trial artifacts are needed; running them first does not shorten or
initialize Phase 1.1.

The v2a calibration notebooks deliberately do not consume
`transitions.csv`. Those tables use one row per `(recording, P)`, whereas the
calibration API requires adjacency matrices. The authoritative observed inputs
are the completed pickle dictionaries, with conditioning depths represented by
their integer keys. Their outputs are isolated under the corresponding
`outputs/calibration/v2a/n25-e9-r16/{recording}/{method}/` directories.

## Phase 2: Real-Data Depth Selection

| Step | Notebook | Depends on | What it does | Why it is required / result contribution |
| --- | --- | --- | --- | --- |
| 2.1 | `notebooks/real_data/v2a_depth_selection.ipynb` | Phase 0 outputs; Phase 1 outputs for calibrated interpretation | Estimates how the diagnostic changes as conditioning depth increases on v2a RSN data. | Produces the core empirical depth-selection result: the depth at which the observed process appears most stable or where instability persists. This directly supports the manuscript's real-data Markovianity diagnostic claims. |

Use `notebooks/real_data/v2a_depth_selection.ipynb` as the canonical new notebook. Any older duplicate under `notebooks/v2a-RSNs/` should be treated as a legacy mirror unless it has been intentionally synchronized.

Expected outputs include `outputs/v2a-RSNs/n25-e9-r16/depth_selection/depth_selection.json`, summary tables, plots, and a manifest.

## Phase 3: Real-Data Edge Localization

| Step | Notebook | Depends on | What it does | Why it is required / result contribution |
| --- | --- | --- | --- | --- |
| 3.1 | `notebooks/real_data/v2a_edgewise_localization.ipynb` | Phase 0 outputs; Phase 1 outputs for calibrated edge FDR | Localizes which directed edges or node pairs contribute most to depth-dependent instability. | Moves the real-data result from a global diagnostic to interpretable network-level evidence. With calibration outputs, it can support edgewise FDR or instability-threshold claims; without calibration it should be reported as descriptive. |

Use `notebooks/real_data/v2a_edgewise_localization.ipynb` as the canonical new notebook. Any older duplicate under `notebooks/v2a-RSNs/` should be treated as a legacy mirror unless it has been intentionally synchronized.

Expected outputs include `outputs/v2a-RSNs/n25-e9-r16/edge_localization/edge_instability.csv`, `edge_fdr.csv`, heatmaps, and a manifest.

## Phase 4: Mechanism Ablations

These three notebooks can run in parallel after the shared code passes preflight. They do not need the real-data calibration outputs, but their final interpretation should be compared with the calibrated real-data findings.

| Step | Notebook | Depends on | What it does | Why it is required / result contribution |
| --- | --- | --- | --- | --- |
| 4.1 | `notebooks/simulations/mechanism_ablations_hidden_state.ipynb` | Synthetic generators and diagnostics | Simulates systems with latent hidden-state structure. | Tests whether the diagnostic responds to genuine latent-state or hidden-confounder mechanisms. This provides positive mechanistic support for the method. |
| 4.2 | `notebooks/simulations/mechanism_ablations_observation_artifacts.ipynb` | Synthetic generators and diagnostics | Simulates measurement or observation artifacts that can mimic dependence changes. | Separates latent-structure signals from artifacts of observation. This guards against over-interpreting empirical instability as hidden confounding. |
| 4.3 | `notebooks/simulations/mechanism_ablations_nonstationarity.ipynb` | Synthetic generators and diagnostics | Simulates nonstationary processes and regime changes. | Tests whether depth instability could be explained by nonstationarity rather than latent confounding. This is needed for a balanced limitations and specificity analysis. |

Expected outputs include mechanism-specific summaries, plots, and manifests under `outputs/simulations/`.

## Phase 5: Positive and Negative Controls

| Step | Notebook | Depends on | What it does | Why it is required / result contribution |
| --- | --- | --- | --- | --- |
| 5.1 | `notebooks/controls/positive_negative_controls.ipynb` | Synthetic generators and diagnostics | Runs cases where the expected answer is known: positive controls should trigger the diagnostic and negative controls should not. | Provides falsification and sanity checks. These results are important for showing that the diagnostic is not merely detecting any finite-sample noise or any arbitrary dependence structure. |

This can run in parallel with Phase 4, Phase 6, Phase 7, Phase 8, and Phase 9 after preflight.

Expected outputs include control summaries and plots under `outputs/controls/`.

## Phase 6: Method Comparisons

| Step | Notebook | Depends on | What it does | Why it is required / result contribution |
| --- | --- | --- | --- | --- |
| 6.1 | `notebooks/comparisons/lpcmci_svarfci_comparison.ipynb` | Synthetic benchmark outputs and installed comparison-method dependencies | Compares the proposed diagnostic against latent-aware causal discovery baselines such as LPCMCI and SVAR-FCI where available. | Positions the method relative to established approaches. This contributes the comparative evidence expected in a Nature Methods-style evaluation. |

This can run in parallel with controls and ablations once the comparison dependencies are available. If optional baseline dependencies are missing, record that explicitly in the output manifest instead of silently dropping the comparison.

Expected outputs include comparison tables, method-level metrics, and manifests under `outputs/comparisons/lpcmci_svarfci/`.

## Phase 7: Nonlinear Conditional-Independence Sensitivity

| Step | Notebook | Depends on | What it does | Why it is required / result contribution |
| --- | --- | --- | --- | --- |
| 7.1 | `notebooks/comparisons/nonlinear_ci_tests.ipynb` | Synthetic benchmarks and CI-test implementations | Repeats diagnostic checks with nonlinear or alternative conditional-independence tests where supported. | Tests whether the conclusions depend on a single linear/Gaussian dependence test. This contributes robustness evidence and clarifies the assumptions behind reported failures or successes. |

This notebook was missing from the earlier order even though the critical path referenced a Phase 8. It should be included before manuscript figure assembly.

Expected outputs include sensitivity summaries under `outputs/comparisons/nonlinear_ci_tests/`.

## Phase 8: Sample-Size and Power Analysis

| Step | Notebook | Depends on | What it does | Why it is required / result contribution |
| --- | --- | --- | --- | --- |
| 8.1 | `notebooks/power/sample_size_power_analysis.ipynb` | Synthetic generators and diagnostic code | Sweeps sample size, effect size, and related simulation settings. | Defines when the diagnostic has enough power to detect the target mechanism. This contributes operating-regime guidance and helps interpret null results. |

This can run in parallel with Phases 4-7 after preflight.

Expected outputs include power curves, summary tables, and manifests under `outputs/power/sample_size_power/`.

## Phase 9: Runtime Scaling

| Step | Notebook | Depends on | What it does | Why it is required / result contribution |
| --- | --- | --- | --- | --- |
| 9.1 | `notebooks/power/runtime_scaling.ipynb` | Diagnostic code and representative benchmark settings | Measures runtime as the number of variables, conditioning depth, sample size, or bootstrap count changes. | Provides computational feasibility evidence and helps justify practical recommendations for running the method. |

This can run in parallel with Phases 4-8 after preflight.

Expected outputs include runtime tables, scaling plots, and manifests under `outputs/power/runtime_scaling/`.

## Phase 10: Reporting and Manuscript Figure Assembly

| Step | Notebook | Depends on | What it does | Why it is required / result contribution |
| --- | --- | --- | --- | --- |
| 10.1 | `notebooks/reporting/nature_methods_figures.ipynb` | All final outputs from Phases 0-9 | Reads frozen experiment outputs and generates the final manuscript figures, figure panels, and summary tables. | This is the final assembly step. It should not perform new scientific analyses; it should only standardize and export results that have already been generated and checked. |

Run this last. If any upstream notebook is rerun after reporting, rerun this notebook as well so the manuscript artifacts reflect the latest frozen outputs.

Expected outputs include final figures/tables under the reporting or `nature_methods/` artifact directories, depending on the notebook configuration.

## Critical Path

The dependency-critical path is:

```text
Preflight -> Phase 0 baseline real-data inputs -> Phase 1 calibration -> Phase 2 depth selection -> Phase 10 reporting
```

If calibrated edge-level claims are part of the results, include Phase 3 before reporting:

```text
Preflight -> Phase 0 baseline real-data inputs -> Phase 1 calibration -> Phase 3 edge localization -> Phase 10 reporting
```

The following can run in parallel after preflight, and after Phase 0 where they need real-data inputs:

```text
Phase 4 mechanism ablations
Phase 5 controls
Phase 6 method comparisons
Phase 7 nonlinear CI sensitivity
Phase 8 sample-size and power analysis
Phase 9 runtime scaling
```

## Recommended Full Run Order

1. Run preflight tests.
2. Run all eight full Phase 0 notebooks if `outputs/v2a-RSNs/n25-e9-r16/c-GC/` or `outputs/v2a-RSNs/n25-e9-r16/c-GC-star/` are incomplete. Existing outputs outside this profile do not satisfy the dependency.
3. Run Phase 1 calibration.
4. Run Phase 2 depth selection.
5. Run Phase 3 edge localization.
6. Run Phases 4-9 in parallel where compute resources allow.
7. Inspect manifests and summaries for failed, skipped, or smoke-test-only runs.
8. Run Phase 10 reporting after all selected upstream outputs are frozen.

## Runtime Notes

The earlier estimate of 1-2 hours is only realistic for small smoke-test settings. Full runs depend on bootstrap count, simulation repeats, sample-size grids, comparison-method availability, and `n_jobs`. Calibration, mechanism ablations, power analysis, runtime scaling, and LPCMCI/SVAR-FCI comparisons are expected to dominate wall-clock time.
