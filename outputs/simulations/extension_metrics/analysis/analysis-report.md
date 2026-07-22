# Extension-metric simulation analysis

## Scope and quality control

The completed output contains 560 method-depth rows: two methods, four
scenarios, ten paired repeats, and seven conditioning depths. Saved input
arrays and ground-truth graphs match across methods for every
scenario-repeat pair.

Inference is restricted to the order-1 unconfounded and latent-common-driver
families. Every trial in those families was finite and below the conservative
numerical explosion threshold `max(abs(X)) < 1e+06`. The
order-3-unconfounded and variable-lag-unconfounded families were excluded
wholesale: each contained three explosive repeats, including non-finite values
and finite magnitudes as high as approximately `1e308`. Their graph changes
cannot be interpreted as causal-diagnostic evidence.

## Primary result

The order-1 null was perfectly stable over depths 1--7 in all ten repeats for
both methods (`T_obs = 0` and cumulative instability `= 0`). With an AR(1)
latent common driver, mean `T_obs` was 0.0333 (SD
0.0157, 95% bootstrap CI
[0.0244, 0.0433]) for
c-GC and 0.0189 (SD
0.0139, 95% CI
[0.0111, 0.0267]) for
c-GC*. Exact paired sign-flip tests against the matched order-1 runs gave
Holm-adjusted `p = 0.0039` for c-GC and
`p = 0.0078` for c-GC*. The latent-driver run had
positive instability in 10/10 c-GC pairs and 8/10 c-GC* pairs.

Cumulative instability through depth seven was 0.0867
(SD 0.0428) for c-GC and
0.0444 (SD 0.0335) for c-GC*.
This secondary endpoint was also zero for every order-1 null run.

## Interpretation

These results support the paper's diagnostic claim for one hidden-memory
mechanism: the same fixed-horizon learner is stable for a clean order-1 process
and changes when an autocorrelated latent common driver is added. The target is
hidden memory or observed-state inadequacy, not unique attribution to latent
confounding. The two higher-order families were intended to test additional
omitted-history mechanisms, but their numerical quality-control failure
prevents a claim about the breadth of detection across mechanisms. The results
also do not imply graph-recovery accuracy; stability and correctness are
different properties. No surrogate-null calibration was performed here, so
the exact paired randomization tests compare simulated scenario families and
are not bootstrap-calibrated tests for an individual dataset.

## Reproducibility

- Inputs: `../fast_gcstar_cgc/extension_metrics.csv`,
  `../fast_gcstar_fcgc/extension_metrics.csv`, and saved trial arrays.
- Methods: optimized vectorized `FastGcStar` implementations of c-GC and
  c-GC*, with a fixed one-lag graph horizon.
- Repeats: 10 paired seeds per scenario.
- Depths: `p = 1, ..., 7`.
- Data dimensions: `T = 2000`, `d = 10`.
- Confidence intervals: percentile bootstrap over paired simulation repeats,
  20,000 resamples, seed 20260705.
- Primary tests: exact two-sided sign-flip tests on paired `T_obs`
  differences; Holm correction across the two method-specific tests.
