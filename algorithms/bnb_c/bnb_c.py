"""Python wrapper for the C branch-and-bound k-constrained modularity maximization solver."""

import subprocess
import json
import os
from pathlib import Path
from typing import Optional
import sys


def solve_bnb_k_modularity_c(
    graph,
    k: int,
    leiden_iterations: int = 100,
    num_threads: int = 0
) -> tuple[list[set[int]], float, float]:
    """
    Solve k-constrained modularity maximization using C branch-and-bound.

    Args:
        graph: NetworkX graph
        k: Number of communities
        leiden_iterations: Number of Leiden restarts for initial bound
        num_threads: Number of threads for parallel execution (0 = sequential)

    Returns:
        Tuple of (partition as list of sets, modularity, initial_lower_bound)
        initial_lower_bound is the modularity from Leiden initialization before BnB search
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
    if num_threads > 0:
        exec_name = 'bnb_solver_parallel.exe' if is_windows else 'bnb_solver_parallel'
        executable = script_dir / exec_name
        if not executable.exists():
            raise FileNotFoundError(
                f"Parallel C executable not found at {executable}. "
                "Please run 'make parallel' in the bnb_c/ directory first."
            )
    else:
        exec_name = 'bnb_solver.exe' if is_windows else 'bnb_solver'
        executable = script_dir / exec_name
        if not executable.exists():
            raise FileNotFoundError(
                f"C executable not found at {executable}. "
                "Please run 'make' in the bnb_c/ directory first."
            )

    # Run solver
    cmd = [str(executable), str(k), str(leiden_iterations)]
    if num_threads > 0:
        cmd.append(str(num_threads))

    result = subprocess.run(
        cmd,
        input=input_data,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"C solver failed: {result.stderr}")

    # Parse output
    output = json.loads(result.stdout)

    # Convert partition indices back to original node labels
    partition = [
        {nodes[idx] for idx in community}
        for community in output['partition']
    ]

    return partition, output['modularity'], output['initial_lower_bound']