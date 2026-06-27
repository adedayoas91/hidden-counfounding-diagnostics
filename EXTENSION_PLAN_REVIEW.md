# Extension-plan implementation review

Review date: 2026-06-27

## Scope

The repository does not contain a file named `extension_plan.md`. This review
uses `others/agent_artifacts/.superpowers/sdd/IMPLEMENTATION_PLAN.md`, the
repository's complete 11-phase extension plan, as the requirements source.

The review covered package code, CLIs, tests, notebooks, frozen outputs,
manifests, lint, compilation, and package build behavior.

## Disposition

| Area | Status | Review result |
|---|---|---|
| Calibration models and CLI | Reworked | Null-model and `n_jobs` CLI options were previously metadata-only. The CLI now executes VAR, residual, moving-block, or stationary bootstrap models, writes pointwise bands, supports surrogate summaries, and uses actual software versions. |
| Calibration plots | Reworked | Added the three required reusable plotting functions with input validation and headless rendering. |
| Depth selection | Pass | Three rules, warnings, missing-depth behavior, and result schema are implemented and tested. |
| Mechanism scenarios | Pass | All nine planned scenarios exist, return the common schema, carry mechanism metadata, and have deterministic tests. |
| Controls | Reworked | Phase randomization previously broke real-signal Fourier symmetry and did not preserve the spectrum. Circular shifting moved all channels together and therefore preserved cross-channel timing. Both controls now implement the stated null transformations. |
| Edge localization | Pass with limitation | Descriptive metrics and replicate-level empirical FDR are implemented. Existing v2a outputs are descriptive because no valid real-data bootstrap replicate set has been generated. |
| LPCMCI/SVAR-FCI | Pass with limitation | LPCMCI uses Tigramite and preserves edge marks. SVAR-FCI is an explicit deferred placeholder, as permitted by the plan. |
| Nonlinear CI comparison | Reworked | The module was absent and the notebook only wrote `status: planned`. Added PCMCI+ CI-test adapters, graceful optional-test skipping, a dependency-light nonlinear residual test, and an executable comparison notebook. |
| Power analysis | Reworked | TPR/FPR previously measured edge recovery on only a latent scenario, contrary to the plan. They now represent instability detection on positive and Markovian-null runs. Seed derivation is stable across Python processes. |
| Parallel/cache layer | Reworked | Cache hits previously returned `None`, dropping completed simulation/fish results. Cache keys also ignored depth grids and fish content. Cached values are now returned, keys cover relevant inputs, and thread-backed parallelism works in restricted environments. |
| Reporting tables/manifests | Reworked | Calibration table parsing did not support `CalibrationResult`, used the wrong v2a path, and wrote the wrong manifest filename. These paths and schemas are corrected. |
| Package quality | Reworked | Static-analysis violations were removed, the missing nonlinear module is packaged, and the package builds successfully. |

## Items that still require experimental reworking

These are not safe to mark complete by code review alone:

1. `notebooks/calibration/bootstrap_null_v2a.ipynb` is scientifically invalid.
   It treats rows of exported edge-count summaries as a time series and passes
   a nested `{"D_p": ...}` object where the bootstrap API requires
   `dict[depth, adjacency]`. It must be rewritten to load the original
   recording traces, rerun the learner for every surrogate and depth, and then
   compute graph instability.
2. The checked-in calibration, controls, and runtime-scaling artifacts are
   explicit placeholders (`pending_full_rerun`) with empty bootstrap samples or
   zero-sized runtime rows. They are not empirical results and must not be cited.
3. All extension notebooks currently have zero executed code cells. The
   notebook code and output artifacts therefore lack execution evidence.
4. `generate_all_figures` still copies an existing PNG when available; it does
   not reconstruct every figure from machine-readable source data. Phase 11
   remains incomplete until each figure type has a data-to-figure renderer.
5. GPDC and CMIknn are skipped in the current environment because their
   optional `dcor` and `numba` dependencies are unavailable. The new
   dependency-light nonlinear residual test provides one nonlinear comparison,
   satisfying the minimum runnable path without adding dependencies.
6. Full real-data bootstrap, complete power grid, runtime grid, and notebook
   reruns were not executed during this review because they are long-running
   scientific experiments, not unit verification.

## Verification requirements before manuscript use

- Execute every extension notebook from a clean environment.
- Replace every `pending_full_rerun`/`planned` artifact with outputs produced by
  that execution.
- Validate manifests against actual input and output paths.
- Confirm the v2a bootstrap uses original traces and full learner reruns.
- Regenerate manuscript figures from machine-readable artifacts.
- Run the complete test, lint, compilation, and build checks after reruns.
