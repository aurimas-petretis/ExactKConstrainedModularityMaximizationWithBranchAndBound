"""Plot k scaling results on karate club graph."""

import glob
import pandas as pd
import matplotlib.pyplot as plt

COLORS = {
    'bnb_c': 'red', 'gurobi': 'blue', 'igraph': 'green', 'bayan': 'orange'
}
MARKERS = {
    'bnb_c': 's', 'gurobi': 'o', 'igraph': 'D', 'bayan': '^'
}


def main():
    files = glob.glob("experiment_5_karate_results_*.csv")
    if not files:
        print("No karate results found in results/")
        return

    csv_path = max(files)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")

    k_constrained = ['bnb_c', 'gurobi']
    unconstrained = ['igraph', 'bayan']

    # Split dataframes
    df_kc = df[df['algorithm'].isin(k_constrained)]
    df_uc = df[df['algorithm'].isin(unconstrained)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Median time vs k
    ax1 = axes[0]
    for alg in k_constrained:
        alg_df = df_kc[df_kc['algorithm'] == alg].copy()
        if len(alg_df) == 0:
            continue
        stats = alg_df.groupby('k')['time'].median()
        ax1.plot(stats.index, stats.values, label=alg, color=COLORS.get(alg, 'gray'),
                 marker=MARKERS.get(alg, 'o'), markersize=6)

    # Plot unconstrained as single points at their n_communities
    for alg in unconstrained:
        row = df_uc[(df_uc['algorithm'] == alg) & (df_uc['success'] == True)]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        k_found = row['n_communities']
        if pd.notna(k_found):
            ax1.plot(k_found, row['time'], label=f"{alg} (k*={int(k_found)})",
                     color=COLORS.get(alg, 'gray'), marker=MARKERS.get(alg, 'o'), markersize=10)

    ax1.set_xlabel('Number of communities (k)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Karate Club: Time vs k')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(2, 11))

    # Plot 2: Modularity vs k
    ax2 = axes[1]
    for alg in k_constrained:
        alg_df = df_kc[(df_kc['algorithm'] == alg) & (df_kc['success'] == True)]
        if len(alg_df) == 0:
            continue
        stats = alg_df.groupby('k')['modularity'].median()
        ax2.plot(stats.index, stats.values, label=alg, color=COLORS.get(alg, 'gray'),
                 marker=MARKERS.get(alg, 'o'), markersize=6)

    for alg in unconstrained:
        row = df_uc[(df_uc['algorithm'] == alg) & (df_uc['success'] == True)]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        k_found = row['n_communities']
        if pd.notna(k_found):
            ax2.plot(k_found, row['modularity'], label=f"{alg} (k*={int(k_found)})",
                     color=COLORS.get(alg, 'gray'), marker=MARKERS.get(alg, 'o'), markersize=10)

    ax2.set_xlabel('Number of communities (k)')
    ax2.set_ylabel('Modularity (Q)')
    ax2.set_title('Karate Club: Modularity vs k')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(2, 11))

    plt.tight_layout()

    output_path = "plot_karate_k_scaling.pdf"
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()