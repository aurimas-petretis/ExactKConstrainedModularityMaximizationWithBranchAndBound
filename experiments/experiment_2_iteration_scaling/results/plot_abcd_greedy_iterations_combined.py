"""
Combined plotting script for Greedy Algorithm Iteration Scaling on ABCD graphs.
Shows modularity and success rate side by side.
"""

import os
import sys
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_latest_results(results_dir):
    """Load the most recent experiment results CSV."""
    pattern = os.path.join(results_dir, "experiment_2_abcd_*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    latest_file = max(files, key=os.path.getmtime)
    print(f"Loading: {latest_file}")

    return pd.read_csv(latest_file)


def plot_combined(df, output_dir):
    """Create side-by-side plots of modularity and success rate."""
    # Filter successful runs only
    df_success = df[df['success'] == True].copy()

    if df_success.empty:
        print("No successful runs to plot!")
        return

    # Get unique iteration and xi values sorted
    iterations = sorted(df_success['iterations'].unique())
    xi_values = sorted(df_success['xi'].unique())

    # Colors and markers for different xi values
    colors = ['blue', 'orange', 'green']
    markers = ['o', 's', '^']

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ========== Left plot: Mean Modularity ==========
    all_mean_opts = {}

    for idx, xi in enumerate(xi_values):
        df_xi = df_success[df_success['xi'] == xi]

        mean_mods = []
        sems_upper = []
        sems_lower = []

        for it in iterations:
            mods = df_xi[df_xi['iterations'] == it]['modularity'].values
            gaps = df_xi[df_xi['iterations'] == it]['gap'].values
            mean_mod = np.mean(mods)
            mean_gap = np.mean(gaps)
            mean_mods.append(mean_mod)

            n = len(gaps)
            gaps_above_mean_mod = gaps[gaps < mean_gap]
            gaps_below_mean_mod = gaps[gaps > mean_gap]

            sem_upper = np.std(mean_gap - gaps_above_mean_mod) / np.sqrt(n) if len(gaps_above_mean_mod) > 0 else 0
            sem_lower = np.std(gaps_below_mean_mod - mean_gap) / np.sqrt(n) if len(gaps_below_mean_mod) > 0 else 0

            sems_upper.append(sem_upper)
            sems_lower.append(sem_lower)

        all_mean_opts[xi] = df_xi['optimal_modularity'].mean()

        ax1.errorbar(range(len(iterations)), mean_mods,
                     yerr=[sems_lower, sems_upper],
                     color=colors[idx % len(colors)],
                     marker=markers[idx % len(markers)],
                     markersize=6, linewidth=2, capsize=4,
                     label=f'ξ = {xi}')

        ax1.axhline(y=all_mean_opts[xi], color=colors[idx % len(colors)],
                    linestyle='--', linewidth=1.5, alpha=0.5)

    ax1.set_xticks(range(len(iterations)))
    ax1.set_xticklabels(iterations)
    ax1.set_xlabel('Number of Iterations', fontsize=12)
    ax1.set_ylabel('Mean Modularity', fontsize=12)
    ax1.set_title('Mean Modularity vs. Number of Iterations', fontsize=14)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, axis='y', alpha=0.3)

    # ========== Right plot: Success Rate ==========
    for idx, xi in enumerate(xi_values):
        df_xi = df_success[df_success['xi'] == xi]
        success_rates = []

        for it in iterations:
            it_data = df_xi[df_xi['iterations'] == it]
            found = it_data[it_data['found_optimal'] == True].shape[0]
            total = it_data.shape[0]
            rate = found / total * 100 if total > 0 else 0
            success_rates.append(rate)

        ax2.plot(range(len(iterations)), success_rates,
                 color=colors[idx % len(colors)],
                 marker=markers[idx % len(markers)],
                 markersize=8, linewidth=2,
                 label=f'ξ = {xi}')

    ax2.axhline(y=100, color='gray', linestyle='--',
                linewidth=1.5, alpha=0.5)

    ax2.set_xticks(range(len(iterations)))
    ax2.set_xticklabels(iterations)
    ax2.set_xlabel('Number of Iterations', fontsize=12)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Rate of Finding Optimal vs. Number of Iterations', fontsize=14)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Main title
    fig.suptitle('Greedy Algorithm Performance on ABCD Graphs', fontsize=16, y=1.02)

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "abcd_greedy_iterations_combined.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")

    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = script_dir

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        df = pd.read_csv(csv_path)
        print(f"Loading: {csv_path}")
    else:
        df = load_latest_results(results_dir)

    plot_combined(df, results_dir)


if __name__ == "__main__":
    main()