"""Plot ABCD n-scaling results."""

import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

COLORS = {
    'gurobi': 'blue', 'bnb_c': 'red', 'pyomo_cplex': 'green', 'bnb_py': 'orange', 'bnb_c_seq': 'darkred',
    'igraph': 'purple', 'bayan': 'teal'
}
MARKERS = {
    'gurobi': 'o', 'bnb_c': 's', 'pyomo_cplex': '^', 'bnb_py': 'd', 'bnb_c_seq': 'v',
    'igraph': 'P', 'bayan': 'X'
}


def main():
    # Find latest ABCD results
    files = glob.glob("experiment_1_abcd_results_*.csv")
    if not files:
        print("No ABCD results found in results/")
        return

    csv_path = max(files)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    algorithms = df['algorithm'].unique()

    # Plot 1: Median time vs n
    ax1 = axes[0]
    for alg in algorithms:
        alg_df = df[(df['algorithm'] == alg) & (df['success'] == True)]
        if len(alg_df) == 0:
            continue
        if alg in ('bnb_c_pruned', 'leiden_c_x100'):
            continue
        stats = alg_df.groupby('n')['time'].median()
        ax1.plot(stats.index, stats.values, label=alg, color=COLORS.get(alg, 'gray'),
                 marker=MARKERS.get(alg, 'o'), markersize=6)

    ax1.set_xlabel('Number of nodes (n)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('ABCD Scaling: Median Time vs Graph Size')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Success rate vs n (for bnb_c_pruned and leiden_c_x100, count only optimal solutions)
    ax2 = axes[1]
    timeout = df[df['timed_out'] == True]['time'].max() if 'timed_out' in df.columns else 10
    for alg in algorithms:
        alg_df = df[df['algorithm'] == alg]
        if alg in ('bnb_c_pruned', 'leiden_c_x100'):
            continue
            # stats = alg_df.groupby('n')['optimal'].apply(lambda x: x.eq(True).mean()) * 100
        else:
            stats = alg_df.groupby('n')['success'].mean() * 100
        ax2.plot(stats.index, stats.values, label=alg,
                 color=COLORS.get(alg, 'gray'), marker=MARKERS.get(alg, 'o'), markersize=6)

    ax2.set_xlabel('Number of nodes (n)')
    ax2.set_ylabel('Success rate (%)')
    ax2.set_title(f'ABCD Success Rate vs Graph Size (timeout={timeout:.0f}s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)

    plt.tight_layout()

    output_path = "plot_abcd_n_scaling.pdf"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()