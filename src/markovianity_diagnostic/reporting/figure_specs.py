"""Figure specifications and metadata for Nature Methods manuscript."""

from __future__ import annotations


FIGURE_SPECS: dict[str, dict] = {
    "Figure1_DepthSelection": {
        "name": "Figure1_DepthSelection",
        "source_data_paths": [
            "outputs/v2a-RSNs/n50-e18-r32/depth_selection/depth_selection.json",
            "outputs/v2a-RSNs/n50-e18-r32/depth_selection/depth_selection_plot.png",
        ],
        "output_path": "nature_methods/figures/Figure1_DepthSelection.png",
        "caption": """
**Conditioning-depth selection across v2a-RGC recordings.**
For c-GC and c-GC* analyses with the reported graph horizon fixed at N_LAGS = 1,
we sweep n_pasts = 1–7. The p = 1 graph is the minimal one-lag baseline, and
diagnostic extra-history conditioning begins at p = 2. The instability metric
D_p measures the normalized graph change from A_{p-1} to A_p, so D_2 quantifies
the effect of adding the first past step beyond the one-lag baseline and D_3–D_7
track subsequent stabilization or persistent depth sensitivity.
""",
        "fig_type": "curve_plot",
    },
    "Figure2_EdgeInstability": {
        "name": "Figure2_EdgeInstability",
        "source_data_paths": [
            "outputs/v2a-RSNs/n50-e18-r32/edge_localization/manifest.json",
        ],
        "output_path": "nature_methods/figures/Figure2_EdgeInstability.png",
        "caption": """
**Edge-level instability heatmap across conditioning depths.**
For each inferred edge in the v2a-RGC network, heatmap shows the fraction of bootstrap resamples
(out of 100) where that edge was added or deleted when increasing conditioning depth from p to p+1.
Red indicates high instability (present/absent inconsistently across resamples).
Rows correspond to directed edges; columns to conditioning depth transitions.
""",
        "fig_type": "heatmap",
    },
    "Figure3_Comparison": {
        "name": "Figure3_Comparison",
        "source_data_paths": [
            "outputs/comparisons/lpcmci_svarfci/comparison_results.json",
        ],
        "output_path": "nature_methods/figures/Figure3_Comparison.png",
        "caption": """
**Comparison of LPCMCI and c-GC on synthetic time series.**
Across five simulated scenarios with varying causal structures and Markovianity assumptions,
we plot edge overlap between the two methods (gray bars) and the fraction of edges marked unstable
by c-GC (blue bars).
Higher blue bars indicate scenarios where Markovianity violations are prevalent.
LPCMCI edges generally align with stable c-GC edges, validating consistency when Markovianity holds.
""",
        "fig_type": "comparison_bars",
    },
    "Figure4_MethodSummary": {
        "name": "Figure4_MethodSummary",
        "source_data_paths": [
            "outputs/simulations/c-GC_results/varLags-Markovian/cgc_aggregated.json",
            "outputs/simulations/c-GC_results/varLags-NonMarkovian/cgc_aggregated.json",
            "outputs/simulations/c-GC_results/singleLag-Markovian/cgc_aggregated.json",
            "outputs/simulations/c-GC_results/singleLag-NonMarkovian/cgc_aggregated.json",
            "outputs/simulations/c-GC-star_results/varLags-Markovian/cgcstar_aggregated.json",
            "outputs/simulations/c-GC-star_results/varLags-NonMarkovian/cgcstar_aggregated.json",
            "outputs/simulations/c-GC-star_results/singleLag-Markovian/cgcstar_aggregated.json",
            "outputs/simulations/c-GC-star_results/singleLag-NonMarkovian/cgcstar_aggregated.json",
        ],
        "output_path": "nature_methods/figures/Figure4_MethodSummary.png",
        "caption": """
**Matched conditioning-depth performance across simulation scenarios.**
Bar plots show mean accuracy (±std), precision, recall, and false positive rate (FPR)
for c-GC and c-GC* across four simulation scenarios:
(A) VAR with Markovian structure (order 1),
(B) VAR with non-Markovian memory dependence,
(C) Single-lag DAG with Markovian dynamics,
(D) Single-lag DAG with hidden confounding.
Only methods implementing the same fixed-lag, requested-depth intervention are
included in this primary comparison.
""",
        "fig_type": "summary_table",
    },
    "Figure5_ControlsSummary": {
        "name": "Figure5_ControlsSummary",
        "source_data_paths": [
            "outputs/controls/",
        ],
        "output_path": "nature_methods/figures/Figure5_ControlsSummary.png",
        "caption": """
**Control experiments validating Markovianity diagnostics.**
Instability curves D_p across conditioning depths for negative controls
(where true Markovianity holds) and positive controls
(where hidden confounding is present).
Negative controls show low D_p throughout (true causal structure is Markovian).
Positive controls show high D_p at low conditioning depths, which decrease as hidden confounding is controlled.
""",
        "fig_type": "curve_plot",
    },
}


__all__ = ["FIGURE_SPECS"]
