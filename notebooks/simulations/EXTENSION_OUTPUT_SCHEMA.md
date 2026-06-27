# Extension-compatible simulation outputs

The original method notebooks export recovery summaries but generally do not
retain per-trial graphs. Those summaries cannot be used to reconstruct graph
instability or edge-level transitions.

Run the `extension_metrics.ipynb` notebook in each method directory to create
the compatible artifacts under:

```text
outputs/simulations/extension_metrics/<method>/
```

Supported methods:

- `fast_gcstar_cgc`
- `fast_gcstar_fcgc`
- `pcmciplus`
- `jpcmciplus`
- `lpcmci`
- `fullci`

## Computed directly

`extension_metrics.csv` contains, per scenario, repeat, and depth:

- edge count
- graph instability `D_p`
- deleted-edge component `D_minus`
- added-edge component `D_plus`
- cumulative instability
- global statistic `T_obs`
- absolute and relative selected depths
- recovery accuracy, precision, recall, FPR, TP, FP, TN, and FN

`extension_results.json` contains the same summaries and warnings.

## Frozen inputs for downstream extensions

Each trial directory contains:

- `data.npy`: original simulated observed series
- `ground_truth.npy`: compact ground-truth adjacency
- `adjacencies.npz`: inferred adjacency at every depth

These files make the run directly consumable by:

- surrogate-null calibration and bootstrap-band depth selection
- edgewise instability localization and empirical FDR
- method-comparison tables
- power and runtime aggregation
- manuscript figure and table exports

Bootstrap critical values, empirical p-values, q-values, and power curves are
not properties of one fitted simulation run. They require additional surrogate
or repeated-grid analyses and therefore remain in their dedicated calibration,
localization, and power workflows.
