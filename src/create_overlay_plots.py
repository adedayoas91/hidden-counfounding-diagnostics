#!/usr/bin/env python
"""Create the primary c-GC/c-GC* conditioning-depth plots."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT.parent / 'outputs' / 'simulations'
FIGURES_DIR = OUTPUT_DIR / 'figures'
MANUSCRIPT_FIGURES_DIR = ROOT.parent.parent / 'images'

METRIC_NAMES = ['accuracy', 'precision', 'recall', 'fpr', 'balanced_accuracy', 'f1']
METRIC_LABELS = {
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'fpr': 'False Positive Rate',
    'balanced_accuracy': 'Balanced Accuracy',
    'f1': 'F1 Score',
}

# Only these methods implement the matched fixed-horizon depth intervention.
METHODS = {
    'c-GC': {
        'color': '#1f77b4',  # blue
        'marker': 'o',
        'linestyle': '-',
        'path': OUTPUT_DIR / 'c-GC_results',
        'agg_filename': 'cgc_aggregated.json',
    },
    'c-GC*': {
        'color': '#ff7f0e',  # orange
        'marker': 's',
        'linestyle': '--',
        'path': OUTPUT_DIR / 'c-GC-star_results',
        'agg_filename': 'cgcstar_aggregated.json',
    },
}


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_path_for_log(path: Path) -> str:
    """Return a readable path for logs without assuming a specific ancestor."""
    try:
        return str(path.relative_to(ROOT.parent))
    except ValueError:
        return str(path)


def load_method_data(scenario: str) -> dict:
    """Load data for all methods for a given scenario."""
    data = {}

    for method_name, method_config in METHODS.items():
        scenario_path = method_config['path'] / scenario / method_config['agg_filename']

        if scenario_path.exists():
            agg = load_json(scenario_path)
            n_pasts = sorted([int(k) for k in agg.keys()])

            data[method_name] = {
                'n_pasts': n_pasts,
                'aggregated': agg,
                'config': method_config,
            }
        else:
            print(f"  ⚠ Missing: {format_path_for_log(scenario_path)}")

    return data


def compute_data_range(all_data: dict, metric: str) -> tuple:
    """Compute y-axis limits based on actual data range."""
    all_means = []
    all_stds = []

    for method_data in all_data.values():
        agg = method_data['aggregated']
        for n in method_data['n_pasts']:
            all_means.append(agg[str(n)][metric]['mean'])
            all_stds.append(agg[str(n)][metric]['std'])

    if not all_means:
        return 0, 1

    min_val = min(all_means) - max(all_stds) * 2
    max_val = max(all_means) + max(all_stds) * 2

    # Add 10% padding
    padding = (max_val - min_val) * 0.1
    min_val = max(0, min_val - padding)
    max_val = min(1.0, max_val + padding)

    return min_val, max_val


def plot_overlay_scenario(scenario: str) -> None:
    """Create overlay plot for a scenario."""
    print(f"\n📊 Creating overlay plot for {scenario}...")

    # Load all method data
    all_data = load_method_data(scenario)

    if len(all_data) < 2:
        print(f"  ⚠ Insufficient methods available (need ≥2, found {len(all_data)})")
        return

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    # Plot each metric
    for metric_idx, metric in enumerate(METRIC_NAMES):
        axis = axes[metric_idx]

        # Plot each method
        for method_name, method_data in all_data.items():
            n_pasts = method_data['n_pasts']
            agg = method_data['aggregated']
            config = method_data['config']

            means = [agg[str(n)][metric]['mean'] for n in n_pasts]
            stds = [agg[str(n)][metric]['std'] for n in n_pasts]

            # Plot line with markers
            axis.plot(
                n_pasts, means,
                marker=config['marker'],
                markersize=8,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=2.5,
                label=method_name,
                alpha=0.8
            )

            # Add error bars
            axis.errorbar(
                n_pasts, means,
                yerr=stds,
                fmt='none',
                color=config['color'],
                capsize=4,
                capthick=2,
                alpha=0.6,
                elinewidth=1.5
            )

        # Formatting
        axis.set_xlabel('$n_{past}$', fontsize=12, fontweight='bold')
        axis.set_ylabel('Score', fontsize=12, fontweight='bold')
        axis.set_title(METRIC_LABELS[metric], fontsize=13, fontweight='bold')
        axis.grid(True, alpha=0.3, linestyle='--')
        axis.legend(fontsize=11, loc='best', framealpha=0.95)

        # Set y limits based on data range
        if metric == 'fpr':
            axis.set_ylim([-0.002, 0.08])
        else:
            y_min, y_max = compute_data_range(all_data, metric)
            axis.set_ylim([y_min, y_max])

    plt.tight_layout()

    # Ensure figures directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Save the primary figure and the copy consumed by the LaTeX manuscript.
    output_path = FIGURES_DIR / f'{scenario}.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    manuscript_path = MANUSCRIPT_FIGURES_DIR / f'{scenario}.png'
    fig.savefig(manuscript_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Saved overlay plot → {output_path.relative_to(ROOT.parent)}")


def main():
    """Main entry point."""
    print("\n🔄 Creating overlay plots for all scenarios...\n")

    scenarios = ['singleLag-Markovian', 'singleLag-NonMarkovian', 'varLags-Markovian', 'varLags-NonMarkovian']

    for scenario in scenarios:
        plot_overlay_scenario(scenario)

    print("\n✅ All overlay plots created!\n")


if __name__ == '__main__':
    main()
