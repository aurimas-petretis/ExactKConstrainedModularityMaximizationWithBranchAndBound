"""Python wrapper for the C Leiden k-constrained modularity maximization solver."""

import subprocess
import json
from pathlib import Path
import sys


def solve_bnb_k_modularity_cleiden(
    graph,
    k: int,
    iterations: int = 100
) -> tuple[list[set[int]], float]:
    """
    Solve k-constrained modularity maximization using C Leiden algorithm only.

    This runs the modified Leiden algorithm used for initial lower bound
    calculation in the branch-and-bound solver, without the full BnB search.

    Args:
        graph: NetworkX graph
        k: Number of communities
        iterations: Number of Leiden restarts (default 100)

    Returns:
        Tuple of (partition as list of sets, modularity)
    """
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Build adjacency matrix
    adj_matrix = [[0] * n for _ in range(n)]
    for u, v in graph.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj_matrix[i][j] = 1
        adj_matrix[j][i] = 1

    # Format input
    lines = [str(n)]
    for row in adj_matrix:
        lines.append(' '.join(map(str, row)))
    input_data = '\n'.join(lines)

    # Find executable
    script_dir = Path(__file__).parent
    is_windows = sys.platform.startswith('win')
    exec_name = 'leiden_solver.exe' if is_windows else 'leiden_solver'
    executable = script_dir / exec_name
    if not executable.exists():
        raise FileNotFoundError(
            f"Leiden executable not found at {executable}. "
            "Please run 'make leiden' in the bnb_c/ directory first."
        )

    # Run solver
    cmd = [str(executable), str(k), str(iterations)]

    result = subprocess.run(
        cmd,
        input=input_data,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Leiden solver failed: {result.stderr}")

    # Parse output
    output = json.loads(result.stdout)

    # Convert partition indices back to original node labels
    partition = [
        {nodes[idx] for idx in community}
        for community in output['partition']
    ]

    return partition, output['modularity']