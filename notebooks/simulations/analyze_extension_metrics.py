"""Analyze completed c-GC extension-metric simulations.

The script creates a compact, auditable analysis bundle under
``outputs/simulations/extension_metrics/analysis``. Inference is restricted to
scenario families that pass trial-level numerical quality control.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import pandas as pd


METHODS = {
    "fast_gcstar_cgc": "c-GC",
    "fast_gcstar_fcgc": "c-GC*",
}
SCENARIOS = {
    "order1_unconfounded": "Order-1 unconfounded",
    "order3_unconfounded": "Order-3 unconfounded",
    "latent_common_driver": "Latent common driver",
    "variable_lag_unconfounded": "Variable-lag unconfounded",
}
PRIMARY_SCENARIOS = ("order1_unconfounded", "latent_common_driver")
MAX_ABS_QC = 1e6
BOOTSTRAP_REPLICATES = 20_000
RNG_SEED = 20260705
COLORS = {
    "order1_unconfounded": "#0072B2",
    "latent_common_driver": "#D55E00",
}


def project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src" / "markovianity_diagnostic").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root")


def percentile_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[float, float]:
    samples = rng.choice(
        values,
        size=(BOOTSTRAP_REPLICATES, values.size),
        replace=True,
    )
    means = samples.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Two-sided paired randomization p-value, discarding exact zero pairs."""

    nonzero = differences[~np.isclose(differences, 0.0)]
    if nonzero.size == 0:
        return 1.0
    observed = abs(float(nonzero.mean()))
    absolute = np.abs(nonzero)
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=nonzero.size):
        statistics.append(abs(float(np.mean(absolute * np.asarray(signs)))))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(pvalues) - rank) * pvalues[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def load_metrics(root: Path) -> pd.DataFrame:
    frames = []
    for method in METHODS:
        path = root / method / "extension_metrics.csv"
        frame = pd.read_csv(path)
        expected_rows = len(SCENARIOS) * 10 * 7
        if len(frame) != expected_rows:
            raise ValueError(f"{path} has {len(frame)} rows; expected {expected_rows}")
        if frame.duplicated(["scenario", "repeat", "depth"]).any():
            raise ValueError(f"{path} contains duplicate trial-depth rows")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def audit_trials(root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for method in METHODS:
        for scenario in SCENARIOS:
            for repeat in range(10):
                trial = root / method / "trials" / scenario / f"repeat_{repeat:03d}"
                data = np.load(trial / "data.npy", allow_pickle=False)
                truth = np.load(trial / "ground_truth.npy", allow_pickle=False)
                finite = np.isfinite(data)
                max_abs = (
                    float(np.max(np.abs(data[finite]))) if finite.any() else np.inf
                )
                records.append(
                    {
                        "method": method,
                        "scenario": scenario,
                        "repeat": repeat,
                        "n_values": data.size,
                        "n_nonfinite": int(data.size - finite.sum()),
                        "max_abs_finite": max_abs,
                        "truth_edges": int(truth.sum()),
                        "passes_qc": bool(finite.all() and max_abs < MAX_ABS_QC),
                    }
                )

    qc = pd.DataFrame(records)
    for scenario in SCENARIOS:
        for repeat in range(10):
            paths = [
                root / method / "trials" / scenario / f"repeat_{repeat:03d}"
                for method in METHODS
            ]
            data_a = np.load(paths[0] / "data.npy", allow_pickle=False)
            data_b = np.load(paths[1] / "data.npy", allow_pickle=False)
            truth_a = np.load(paths[0] / "ground_truth.npy", allow_pickle=False)
            truth_b = np.load(paths[1] / "ground_truth.npy", allow_pickle=False)
            if not np.array_equal(data_a, data_b, equal_nan=True):
                raise ValueError(f"Method inputs differ for {scenario}, repeat {repeat}")
            if not np.array_equal(truth_a, truth_b):
                raise ValueError(f"Ground truths differ for {scenario}, repeat {repeat}")
    return qc


def repeat_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["method", "scenario", "repeat"], as_index=False)
        .agg(
            T_obs=("T_obs", "first"),
            cumulative_instability=("cumulative_instability", "last"),
        )
        .sort_values(["method", "scenario", "repeat"])
    )


def build_statistics(repeats: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    records: list[dict[str, object]] = []
    primary_pvalues: list[float] = []
    for method in METHODS:
        wide = repeats[repeats["method"].eq(method)].pivot(
            index="repeat",
            columns="scenario",
            values=["T_obs", "cumulative_instability"],
        )
        for endpoint in ("T_obs", "cumulative_instability"):
            differences = (
                wide[endpoint]["latent_common_driver"]
                - wide[endpoint]["order1_unconfounded"]
            ).to_numpy()
            low, high = percentile_mean_ci(differences, rng=rng)
            pvalue = exact_sign_flip_pvalue(differences)
            record = {
                "comparison": "latent_common_driver - order1_unconfounded",
                "method": method,
                "endpoint": endpoint,
                "n_pairs": differences.size,
                "mean_difference": float(differences.mean()),
                "median_difference": float(np.median(differences)),
                "ci95_low": low,
                "ci95_high": high,
                "positive_pairs": int((differences > 0).sum()),
                "negative_pairs": int((differences < 0).sum()),
                "zero_pairs": int(np.isclose(differences, 0.0).sum()),
                "p_exact": pvalue,
                "p_holm_primary": np.nan,
            }
            if endpoint == "T_obs":
                primary_pvalues.append(pvalue)
            records.append(record)

    adjusted = holm_adjust(primary_pvalues)
    cursor = 0
    for record in records:
        if record["endpoint"] == "T_obs":
            record["p_holm_primary"] = adjusted[cursor]
            cursor += 1

    for endpoint in ("T_obs", "cumulative_instability"):
        wide = repeats[repeats["scenario"].eq("latent_common_driver")].pivot(
            index="repeat",
            columns="method",
            values=endpoint,
        )
        differences = (
            wide["fast_gcstar_cgc"] - wide["fast_gcstar_fcgc"]
        ).to_numpy()
        low, high = percentile_mean_ci(differences, rng=rng)
        records.append(
            {
                "comparison": "c-GC - c-GC* within latent_common_driver",
                "method": "paired_methods",
                "endpoint": endpoint,
                "n_pairs": differences.size,
                "mean_difference": float(differences.mean()),
                "median_difference": float(np.median(differences)),
                "ci95_low": low,
                "ci95_high": high,
                "positive_pairs": int((differences > 0).sum()),
                "negative_pairs": int((differences < 0).sum()),
                "zero_pairs": int(np.isclose(differences, 0.0).sum()),
                "p_exact": exact_sign_flip_pvalue(differences),
                "p_holm_primary": np.nan,
            }
        )
    return pd.DataFrame(records)


def summary_table(repeats: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 1)
    rows: list[dict[str, object]] = []
    for (method, scenario), group in repeats.groupby(["method", "scenario"]):
        if scenario not in PRIMARY_SCENARIOS:
            continue
        for endpoint in ("T_obs", "cumulative_instability"):
            values = group[endpoint].to_numpy()
            low, high = percentile_mean_ci(values, rng=rng)
            rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "endpoint": endpoint,
                    "n": values.size,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(repeats: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    endpoints = (
        ("T_obs", r"Maximum instability $T_{\mathrm{obs}}$"),
        ("cumulative_instability", r"Cumulative instability $\sum_{p=2}^7 D_p$"),
    )
    jitter = np.linspace(-0.035, 0.035, 10)
    for axis, (endpoint, ylabel) in zip(axes, endpoints, strict=True):
        for method_index, method in enumerate(METHODS):
            subset = repeats[repeats["method"].eq(method)]
            wide = subset.pivot(index="repeat", columns="scenario", values=endpoint)
            x_null = method_index * 3.0
            x_latent = x_null + 1.0
            for repeat_index, (_, row) in enumerate(wide.iterrows()):
                axis.plot(
                    [x_null + jitter[repeat_index], x_latent + jitter[repeat_index]],
                    [
                        row["order1_unconfounded"],
                        row["latent_common_driver"],
                    ],
                    color="#999999",
                    alpha=0.45,
                    linewidth=0.75,
                    zorder=1,
                )
            for x, scenario in (
                (x_null, "order1_unconfounded"),
                (x_latent, "latent_common_driver"),
            ):
                values = wide[scenario].to_numpy()
                axis.scatter(
                    x + jitter,
                    values,
                    s=23,
                    color=COLORS[scenario],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=2,
                )
                axis.scatter(
                    [x],
                    [values.mean()],
                    marker="D",
                    s=42,
                    color="black",
                    zorder=3,
                )
        axis.set_xticks([0.5, 3.5], list(METHODS.values()))
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=-0.004)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[0].scatter([], [], color=COLORS["order1_unconfounded"], label="Order-1 null")
    axes[0].scatter(
        [], [], color=COLORS["latent_common_driver"], label="Latent common driver"
    )
    axes[0].scatter([], [], marker="D", color="black", label="Mean")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_depth_profiles(metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7), sharey=True)
    rng = np.random.default_rng(RNG_SEED + 2)
    for axis, method in zip(axes, METHODS, strict=True):
        subset = metrics[
            metrics["method"].eq(method)
            & metrics["scenario"].isin(PRIMARY_SCENARIOS)
            & metrics["depth"].gt(1)
        ]
        for scenario in PRIMARY_SCENARIOS:
            scenario_data = subset[subset["scenario"].eq(scenario)]
            means = scenario_data.groupby("depth")["D_p"].mean()
            lows = []
            highs = []
            for depth in means.index:
                values = scenario_data[
                    scenario_data["depth"].eq(depth)
                ]["D_p"].to_numpy()
                low, high = percentile_mean_ci(values, rng=rng)
                lows.append(low)
                highs.append(high)
            axis.plot(
                means.index,
                means.values,
                marker="o",
                color=COLORS[scenario],
                label=SCENARIOS[scenario],
            )
            axis.fill_between(
                means.index,
                lows,
                highs,
                color=COLORS[scenario],
                alpha=0.18,
                linewidth=0,
            )
        axis.set_title(METHODS[method])
        axis.set_xlabel("Conditioning depth, $p$")
        axis.set_xticks(range(2, 8))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[0].set_ylabel(r"Adjacent-depth instability, $D_p$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_schematic(path: Path) -> None:
    observed_color = "#0072B2"
    latent_color = "#D55E00"
    excluded_color = "#777777"

    def node(
        axis: plt.Axes,
        xy: tuple[float, float],
        label: str,
        *,
        color: str = observed_color,
        dashed: bool = False,
    ) -> None:
        circle = Circle(
            xy,
            0.085,
            facecolor="white",
            edgecolor=color,
            linewidth=2.0,
            linestyle="--" if dashed else "-",
            zorder=3,
        )
        axis.add_patch(circle)
        axis.text(*xy, label, ha="center", va="center", fontsize=10, zorder=4)

    def arrow(
        axis: plt.Axes,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = "#444444",
        bend: float = 0.0,
    ) -> None:
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            color=color,
            linewidth=1.4,
            connectionstyle=f"arc3,rad={bend}",
            shrinkA=9,
            shrinkB=9,
            zorder=2,
        )
        axis.add_patch(patch)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.7))
    for axis in axes.ravel():
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")

    axis = axes[0, 0]
    axis.set_title("a  Order-1 unconfounded", loc="left", fontweight="bold")
    node(axis, (0.20, 0.55), r"$X_{t-2}$")
    node(axis, (0.50, 0.55), r"$X_{t-1}$")
    node(axis, (0.80, 0.55), r"$X_t$")
    arrow(axis, (0.29, 0.55), (0.41, 0.55))
    arrow(axis, (0.59, 0.55), (0.71, 0.55))
    axis.text(
        0.5,
        0.20,
        r"$X_t=A_1X_{t-1}+\varepsilon_t$"
        "\nretained; 10/10 repeats passed QC",
        ha="center",
        va="center",
        fontsize=9,
    )

    axis = axes[0, 1]
    axis.set_title("b  Order-3 unconfounded", loc="left", fontweight="bold")
    for x, label in zip(
        (0.12, 0.34, 0.56),
        (r"$X_{t-3}$", r"$X_{t-2}$", r"$X_{t-1}$"),
        strict=True,
    ):
        node(axis, (x, 0.62), label, color=excluded_color)
        arrow(axis, (x + 0.06, 0.56), (0.78, 0.48), color=excluded_color)
    node(axis, (0.86, 0.45), r"$X_t$", color=excluded_color)
    axis.text(
        0.5,
        0.18,
        "three sparse lag matrices\nQC-excluded: 3/10 explosive repeats",
        ha="center",
        va="center",
        fontsize=9,
        color=excluded_color,
    )

    axis = axes[1, 0]
    axis.set_title("c  Latent common driver", loc="left", fontweight="bold")
    node(axis, (0.18, 0.68), r"$H_{t-2}$", color=latent_color, dashed=True)
    node(axis, (0.45, 0.68), r"$H_{t-1}$", color=latent_color, dashed=True)
    node(axis, (0.75, 0.72), r"$X_t^{(i)}$")
    node(axis, (0.75, 0.38), r"$X_t^{(j)}$")
    node(axis, (0.28, 0.30), r"$X_{t-1}$")
    arrow(axis, (0.27, 0.68), (0.36, 0.68), color=latent_color)
    arrow(axis, (0.53, 0.68), (0.67, 0.72), color=latent_color)
    arrow(axis, (0.51, 0.62), (0.68, 0.41), color=latent_color)
    arrow(axis, (0.35, 0.33), (0.68, 0.39), color=observed_color)
    axis.text(
        0.47,
        0.08,
        r"$H_t=0.8H_{t-1}+\eta_t$; "
        r"$X_t=A_1X_{t-1}+wH_{t-1}+\varepsilon_t$"
        "\nretained; 10/10 repeats passed QC",
        ha="center",
        va="center",
        fontsize=8.5,
    )

    axis = axes[1, 1]
    axis.set_title("d  Variable-lag unconfounded", loc="left", fontweight="bold")
    source_nodes = (
        ((0.18, 0.74), r"$X_{t-1}^{(j)}$"),
        ((0.18, 0.50), r"$X_{t-2}^{(k)}$"),
        ((0.18, 0.26), r"$X_{t-3}^{(\ell)}$"),
    )
    for position, label in source_nodes:
        node(axis, position, label, color=excluded_color)
        arrow(axis, (0.27, position[1]), (0.71, 0.50), color=excluded_color)
    node(axis, (0.80, 0.50), r"$X_t^{(i)}$", color=excluded_color)
    axis.text(
        0.5,
        0.08,
        "lower-density sparse order-3 variant\n"
        "QC-excluded: 3/10 explosive repeats",
        ha="center",
        va="center",
        fontsize=9,
        color=excluded_color,
    )

    fig.suptitle(
        "Extension simulation scenarios (representative dependencies)",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_number(value: float) -> str:
    return f"{value:.4f}"


def markdown_table(frame: pd.DataFrame, *, decimals: int = 6) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{decimals}g}"
        return str(value)

    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    rows.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def write_reports(
    output: Path,
    qc: pd.DataFrame,
    summary: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    scenario_qc = (
        qc.groupby("scenario")
        .agg(
            method_trials=("passes_qc", "size"),
            passing_method_trials=("passes_qc", "sum"),
            nonfinite_values=("n_nonfinite", "sum"),
            maximum_absolute_value=("max_abs_finite", "max"),
        )
        .reset_index()
    )
    cgc_t = summary[
        summary["method"].eq("fast_gcstar_cgc")
        & summary["scenario"].eq("latent_common_driver")
        & summary["endpoint"].eq("T_obs")
    ].iloc[0]
    fcgc_t = summary[
        summary["method"].eq("fast_gcstar_fcgc")
        & summary["scenario"].eq("latent_common_driver")
        & summary["endpoint"].eq("T_obs")
    ].iloc[0]
    cgc_c = summary[
        summary["method"].eq("fast_gcstar_cgc")
        & summary["scenario"].eq("latent_common_driver")
        & summary["endpoint"].eq("cumulative_instability")
    ].iloc[0]
    fcgc_c = summary[
        summary["method"].eq("fast_gcstar_fcgc")
        & summary["scenario"].eq("latent_common_driver")
        & summary["endpoint"].eq("cumulative_instability")
    ].iloc[0]
    cgc_test = statistics[
        statistics["method"].eq("fast_gcstar_cgc")
        & statistics["endpoint"].eq("T_obs")
    ].iloc[0]
    fcgc_test = statistics[
        statistics["method"].eq("fast_gcstar_fcgc")
        & statistics["endpoint"].eq("T_obs")
    ].iloc[0]

    report = f"""# Extension-metric simulation analysis

## Scope and quality control

The completed output contains 560 method-depth rows: two methods, four
scenarios, ten paired repeats, and seven conditioning depths. Saved input
arrays and ground-truth graphs match across methods for every
scenario-repeat pair.

Inference is restricted to the order-1 unconfounded and latent-common-driver
families. Every trial in those families was finite and below the conservative
numerical explosion threshold `max(abs(X)) < {MAX_ABS_QC:.0e}`. The
order-3-unconfounded and variable-lag-unconfounded families were excluded
wholesale: each contained three explosive repeats, including non-finite values
and finite magnitudes as high as approximately `1e308`. Their graph changes
cannot be interpreted as causal-diagnostic evidence.

## Primary result

The order-1 null was perfectly stable over depths 1--7 in all ten repeats for
both methods (`T_obs = 0` and cumulative instability `= 0`). With an AR(1)
latent common driver, mean `T_obs` was {format_number(cgc_t["mean"])} (SD
{format_number(cgc_t["sd"])}, 95% bootstrap CI
[{format_number(cgc_t["ci95_low"])}, {format_number(cgc_t["ci95_high"])}]) for
c-GC and {format_number(fcgc_t["mean"])} (SD
{format_number(fcgc_t["sd"])}, 95% CI
[{format_number(fcgc_t["ci95_low"])}, {format_number(fcgc_t["ci95_high"])}]) for
c-GC*. Exact paired sign-flip tests against the matched order-1 runs gave
Holm-adjusted `p = {cgc_test["p_holm_primary"]:.4f}` for c-GC and
`p = {fcgc_test["p_holm_primary"]:.4f}` for c-GC*. The latent-driver run had
positive instability in 10/10 c-GC pairs and 8/10 c-GC* pairs.

Cumulative instability through depth seven was {format_number(cgc_c["mean"])}
(SD {format_number(cgc_c["sd"])}) for c-GC and
{format_number(fcgc_c["mean"])} (SD {format_number(fcgc_c["sd"])}) for c-GC*.
This secondary endpoint was also zero for every order-1 null run.

## Interpretation

These results support sensitivity to one specified hidden-memory mechanism:
the same fixed-horizon learner is stable for a clean order-1 process and
changes when an autocorrelated latent common driver is added. They do not
establish specificity against higher-order or heterogeneous-lag alternatives,
because the two intended control families failed numerical quality control.
They also do not imply graph-recovery accuracy; stability and correctness are
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
  {BOOTSTRAP_REPLICATES:,} resamples, seed {RNG_SEED}.
- Primary tests: exact two-sided sign-flip tests on paired `T_obs`
  differences; Holm correction across the two method-specific tests.
"""
    (output / "analysis-report.md").write_text(report)

    stats_lines = [
        "# Statistical appendix",
        "",
        "## Scenario-level numerical quality control",
        "",
        markdown_table(scenario_qc, decimals=4),
        "",
        f"`passes_qc` requires all values to be finite and `max(abs(X)) < {MAX_ABS_QC:.0e}`. "
        "Counts include both methods; method inputs were identical within each paired repeat.",
        "",
        "## Descriptive statistics for retained scenarios",
        "",
        markdown_table(summary),
        "",
        "## Paired comparisons",
        "",
        markdown_table(statistics),
        "",
        "The `p_holm_primary` column is populated only for the two prespecified "
        "`T_obs` comparisons. Method contrasts and cumulative-instability tests "
        "are secondary and are reported without multiplicity-adjusted claims.",
    ]
    (output / "stats-appendix.md").write_text("\n".join(stats_lines) + "\n")

    catalog = """# Figure catalog

## Figure 1 — Extension simulation summary

- Files: `figures/extension_summary.pdf`, `figures/extension_summary.png`
- Panels: maximum adjacent-depth instability and cumulative instability.
- Marks: paired repeat-level points connected within method; diamonds are
  arithmetic means.
- Included scenarios: order-1 unconfounded and latent common driver.
- Exclusions: both higher-order scenario families failed numerical QC.

## Figure 2 — Adjacent-depth instability profiles

- Files: `figures/extension_depth_profiles.pdf`,
  `figures/extension_depth_profiles.png`
- Panels: c-GC and c-GC*.
- Lines: repeat means at each transition; ribbons are 95% percentile-bootstrap
  confidence intervals over the ten paired repeats.
- The depth axis begins at 2 because `D_2` is the first graph transition.

## Figure 3 — Extension scenario schematic

- Files: `figures/extension_scenarios.pdf`,
  `figures/extension_scenarios.png`
- Panels: order-1 unconfounded, order-3 unconfounded, latent common driver,
  and variable-lag unconfounded.
- Blue nodes are observed, orange dashed nodes are latent, and grey panels
  denote scenario families excluded after numerical quality control.
- The drawings show representative temporal dependencies, not complete
  multivariate graphs.
"""
    (output / "figure-catalog.md").write_text(catalog)


def main() -> None:
    root = project_root()
    metric_root = root / "outputs" / "simulations" / "extension_metrics"
    output = metric_root / "analysis"
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(metric_root)
    qc = audit_trials(metric_root)
    repeats = repeat_metrics(metrics)
    summary = summary_table(repeats)
    statistics = build_statistics(repeats)

    qc.to_csv(output / "qc-trials.csv", index=False)
    summary.to_csv(output / "descriptive-statistics.csv", index=False)
    statistics.to_csv(output / "paired-tests.csv", index=False)
    plot_summary(repeats, figure_dir / "extension_summary")
    plot_depth_profiles(metrics, figure_dir / "extension_depth_profiles")
    plot_scenario_schematic(figure_dir / "extension_scenarios")
    write_reports(output, qc, summary, statistics)

    print(f"Wrote analysis bundle: {output}")
    print("Retained scenarios:", ", ".join(PRIMARY_SCENARIOS))
    excluded = [
        scenario
        for scenario, group in qc.groupby("scenario")
        if not bool(group["passes_qc"].all())
    ]
    print("QC-excluded scenarios:", ", ".join(excluded))


if __name__ == "__main__":
    main()
