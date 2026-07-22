"""Plot Relaxed Caveman noise scaling results."""

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
    # Find latest Relaxed Caveman results
    files = glob.glob("experiment_6_relaxed_caveman_results_*.csv")
    if not files:
        print("No Relaxed Caveman results found in results/")
        return

    csv_path = max(files)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    algorithms = df['algorithm'].unique()

    # Determine timeout value
    timed_out_df = df[df['timed_out'] == True] if 'timed_out' in df.columns else pd.DataFrame()
    timeout = timed_out_df['time'].max() if len(timed_out_df) > 0 else None

    # Plot 1: Median time vs p (all runs, timed-out use timeout value)
    ax1 = axes[0]
    for alg in algorithms:
        alg_df = df[df['algorithm'] == alg].copy()
        if len(alg_df) == 0:
            continue
        stats = alg_df.groupby('p')['time'].median()
        if timeout is not None:
            alg_df.loc[alg_df['timed_out'] == True, 'time'] = timeout
            normal = stats[stats < timeout]
            at_timeout = stats[stats >= timeout]
        else:
            normal = stats
            at_timeout = pd.Series(dtype=float)
        ax1.plot(normal.index, normal.values, label=alg, color=COLORS.get(alg, 'gray'),
                 marker=MARKERS.get(alg, 'o'), markersize=6)
        if len(at_timeout) > 0:
            ax1.plot(at_timeout.index, at_timeout.values, color=COLORS.get(alg, 'gray'),
                     marker='^', markersize=10, linestyle='none')

    if timeout is not None:
        ax1.axhline(y=timeout, color='black', linestyle='--', alpha=0.4, label=f'timeout={timeout:.0f}s')
    ax1.set_xlabel('Rewiring probability (p)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title(f'Relaxed Caveman Median Time vs Noise')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Success rate vs p
    ax2 = axes[1]
    for alg in algorithms:
        alg_df = df[df['algorithm'] == alg]
        stats = alg_df.groupby('p')['success'].mean() * 100
        ax2.plot(stats.index, stats.values, label=alg,
                 color=COLORS.get(alg, 'gray'), marker=MARKERS.get(alg, 'o'), markersize=6)

    ax2.set_xlabel('Rewiring probability (p)')
    ax2.set_ylabel('Success rate (%)')
    timeout_str = f' (timeout={timeout:.0f}s)' if timeout is not None else ''
    ax2.set_title(f'Relaxed Caveman Success Rate vs Noise{timeout_str}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)

    plt.tight_layout()

    output_path = "plot_relaxed_caveman_noise_scaling.pdf"
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()