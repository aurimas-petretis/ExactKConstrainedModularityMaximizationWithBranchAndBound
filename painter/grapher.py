import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"


def draw_set_solution(G, communities, filename, title='Set solution'):
    colored_nodes = [-1] * len(G.nodes)
    for idx, nodes in enumerate(communities):
        for node in nodes:
            colored_nodes[node] = idx

    print(colored_nodes)

    plt.figure(figsize=(8, 6))

    pos = nx.spring_layout(G, seed=8)

    cmap = plt.cm.Set3.copy()
    cmap.set_under("white")

    # Draw the full graph normally
    nx.draw_networkx_nodes(G, pos, node_color=colored_nodes, cmap=cmap, edgecolors="black", vmin=0, vmax=max(len(communities) - 1, 1), node_size=300, margins=0)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray')

    plt.title(title)
    plt.axis('off')
    if(filename):
        location = IMAGES_DIR / f"{filename}.pdf"
        plt.savefig(location, bbox_inches="tight", pad_inches=0)
        print(f"Plot saved to {location}")
    plt.show()


def draw_white_graph(G, filename, title='Set solution'):
    plt.figure(figsize=(8, 6))

    pos = nx.spring_layout(G, seed=8)

    # Draw the full graph normally
    nx.draw_networkx_nodes(G, pos, node_color="#ffffff", edgecolors="black", linewidths=1.0, cmap=plt.cm.Set3, node_size=300)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray')

    plt.title(title)
    plt.axis('off')
    if(filename):
        location = IMAGES_DIR / f"{filename}.pdf"
        plt.savefig(location)
        print(f"Plot saved to {location}")
    plt.show()


def plot_bounds_evolution(upper_bound_history, lower_bound_history, filename=None, title=f'Upper and Lower Bound Evolution'):
    """Plot the evolution of upper and lower bounds during the algorithm execution."""
    if not upper_bound_history or not lower_bound_history:
        print("Failed to plot bound history: No bound history available (bestfs does not have upper bound history implemented)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract data
    iterations_upper = [x[0] for x in upper_bound_history]
    upper_bounds = [x[1] for x in upper_bound_history]
    iterations_lower = [x[0] for x in lower_bound_history]
    lower_bounds = [x[1] for x in lower_bound_history]

    # Plot bounds
    ax.plot(iterations_upper, upper_bounds, label='Upper Bound', color='red', alpha=0.7, linewidth=1.5)
    ax.plot(iterations_lower, lower_bounds, label='Lower Bound', color='blue', alpha=0.7, linewidth=1.5)

    # Fill the gap between bounds
    ax.fill_between(iterations_upper, upper_bounds,
                    np.interp(iterations_upper, iterations_lower, lower_bounds),
                    alpha=0.2, color='gray', label='Search Space')

    ax.set_xlabel('Recursive calls', fontsize=12)
    ax.set_ylabel('Modularity Bound', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add final values as text
    final_upper = upper_bounds[-1] if upper_bounds else 'N/A'
    final_lower = lower_bounds[-1] if lower_bounds else 'N/A'
    ax.text(0.02, 0.98, f'Final Upper Bound: {final_upper:.4f}\nFinal Lower Bound: {final_lower:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if filename:
        location = IMAGES_DIR / f"{filename}.pdf"
        plt.savefig(location, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {location}")

    plt.show()
