"""Plot planted partition degree distribution scaling results."""

import glob
import pandas as pd
import matplotlib.pyplot as plt

COLORS = {
    'gurobi': 'blue', 'bnb_c': 'red'
}
MARKERS = {
    'gurobi': 'o', 'bnb_c': 's'
}


def main():
    # Find latest planted partition results
    files = glob.glob("experiment_4_planted_partition_results_*.csv")
    if not files:
        print("No planted partition results found in results/")
        return

    csv_path = max(files)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    algorithms = df['algorithm'].unique()

    # Determine timeout value
    timeout = df[df['timed_out'] == True]['time'].max() if 'timed_out' in df.columns else 10

    # Plot 1: Median time vs n (all runs, timed-out use timeout value)
    ax1 = axes[0]
    for alg in algorithms:
        alg_df = df[df['algorithm'] == alg].copy()
        if len(alg_df) == 0:
            continue
        alg_df.loc[alg_df['timed_out'] == True, 'time'] = timeout
        stats = alg_df.groupby('n')['time'].median()
        normal = stats[stats < timeout]
        at_timeout = stats[stats >= timeout]
        ax1.plot(normal.index, normal.values, label=alg, color=COLORS.get(alg, 'gray'),
                 marker=MARKERS.get(alg, 'o'), markersize=6)
        if len(at_timeout) > 0:
            ax1.plot(at_timeout.index, at_timeout.values, color=COLORS.get(alg, 'gray'),
                     marker='^', markersize=10, linestyle='none')

    ax1.axhline(y=timeout, color='black', linestyle='--', alpha=0.4, label=f'timeout={timeout:.0f}s')
    ax1.set_xlabel('Number of nodes (n)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title(f'Planted Partition Median Time vs Graph Size')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Success rate vs n
    ax2 = axes[1]
    for alg in algorithms:
        alg_df = df[df['algorithm'] == alg]
        stats = alg_df.groupby('n')['success'].mean() * 100
        ax2.plot(stats.index, stats.values, label=alg,
                 color=COLORS.get(alg, 'gray'), marker=MARKERS.get(alg, 'o'), markersize=6)

    ax2.set_xlabel('Number of nodes (n)')
    ax2.set_ylabel('Success rate (%)')
    ax2.set_title(f'Planted Partition Success Rate vs Graph Size (timeout={timeout:.0f}s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)

    plt.tight_layout()

    output_path = "plot_planted_partition_n_detailed.pdf"
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()