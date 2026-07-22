"""
Experiment: Static Graph Comparison - BnB C vs Gurobi

Objective: Compare solve_bnb_k_modularity_c and gurobi_k_mod_max on static/reproducible
benchmark graphs where both algorithms are challenged.
"""

import time
import os
import signal
import multiprocessing as mp
from datetime import datetime

import networkx as nx
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from algorithms.gurobi_solution import gurobi_k_mod_max
from algorithms.bnb_c.bnb_c import solve_bnb_k_modularity_c


# Configuration
TIMEOUT = 120  # seconds - longer to find true limits
NUM_THREADS = 8  # for C implementation

# Test cases: (graph_name, graph_generator_func, k_values)
TEST_CASES = []


def add_static_graphs():
    """Add static test graphs to TEST_CASES."""

    # Real-world static graphs
    TEST_CASES.append(('florentine_families', lambda: nx.florentine_families_graph(), [2, 3, 4, 5]))
    TEST_CASES.append(('davis_southern_women', lambda: nx.davis_southern_women_graph(), [2, 3, 4, 5]))
    TEST_CASES.append(('karate_club', lambda: nx.karate_club_graph(), [2, 3, 4, 5, 6, 7, 8, 9, 10]))
    TEST_CASES.append(('les_miserables', lambda: nx.les_miserables_graph(), [2, 3, 4, 5]))

    # Planted partition - weak communities (challenges BnB)
    TEST_CASES.append((
        'planted_partition_80_weak',
        lambda: nx.planted_partition_graph(2, 40, 0.25, 0.08, seed=42),
        [2, 3, 4]
    ))
    TEST_CASES.append((
        'planted_partition_100_medium',
        lambda: nx.planted_partition_graph(2, 50, 0.30, 0.06, seed=42),
        [2, 3]
    ))
    TEST_CASES.append((
        'planted_partition_120_hard',
        lambda: nx.planted_partition_graph(2, 60, 0.30, 0.05, seed=42),
        [2]
    ))


    TEST_CASES.append((
        'planted_partition_120_strong_001',
        lambda: nx.planted_partition_graph(2, 60, 0.2, 0.01, seed=42),
        [2]
    ))
    TEST_CASES.append((
        'planted_partition_120_strong_002',
        lambda: nx.planted_partition_graph(2, 60, 0.2, 0.02, seed=42),
        [2]
    ))
    TEST_CASES.append((
        'planted_partition_120_strong_003',
        lambda: nx.planted_partition_graph(2, 60, 0.2, 0.03, seed=42),
        [2]
    ))
    TEST_CASES.append((
        'planted_partition_120_strong_004',
        lambda: nx.planted_partition_graph(2, 60, 0.2, 0.04, seed=42),
        [2]
    ))
    TEST_CASES.append((
        'planted_partition_120_strong_005',
        lambda: nx.planted_partition_graph(2, 60, 0.2, 0.05, seed=42),
        [2]
    ))

    TEST_CASES.append((
        'connected_caveman_1000',
        lambda: nx.connected_caveman_graph(2, 1000),
        [2]
    ))
    TEST_CASES.append((
        'connected_caveman_2000',
        lambda: nx.connected_caveman_graph(2, 2000),
        [2]
    ))

    TEST_CASES.append((
        'relaxed_connected_caveman_100_001',
        lambda: nx.relaxed_caveman_graph(2, 100, 0.01),
        [2]
    ))

    TEST_CASES.append((
        'relaxed_connected_caveman_100_002',
        lambda: nx.relaxed_caveman_graph(2, 100, 0.02),
        [2]
    ))

    TEST_CASES.append((
        'relaxed_connected_caveman_100_004',
        lambda: nx.relaxed_caveman_graph(2, 100, 0.04),
        [2]
    ))

    TEST_CASES.append((
        'relaxed_connected_caveman_100_006',
        lambda: nx.relaxed_caveman_graph(2, 100, 0.06),
        [2]
    ))

    TEST_CASES.append((
        'relaxed_connected_caveman_100_008',
        lambda: nx.relaxed_caveman_graph(2, 100, 0.08),
        [2]
    ))

    TEST_CASES.append((
        'relaxed_connected_caveman_100_010',
        lambda: nx.relaxed_caveman_graph(2, 100, 0.1),
        [2]
    ))

    TEST_CASES.append((
        'relaxed_connected_caveman_100_012',
        lambda: nx.relaxed_caveman_graph(2, 100, 0.12),
        [2]
    ))

    TEST_CASES.append((
        'relaxed_connected_caveman_100_014',
        lambda: nx.relaxed_caveman_graph(2, 100, 0.14),
        [2]
    ))

    TEST_CASES.append((
        'relaxed_connected_caveman_100_016',
        lambda: nx.relaxed_caveman_graph(2, 100, 0.16),
        [2]
    ))

    TEST_CASES.append((
        'relaxed_connected_caveman_100_018',
        lambda: nx.relaxed_caveman_graph(2, 100, 0.18),
        [2]
    ))

    TEST_CASES.append((
        'relaxed_connected_caveman_100_020',
        lambda: nx.relaxed_caveman_graph(2, 100, 0.2),
        [2]
    ))

    # Stochastic Block Model - flexible community structure
    # 3 communities of sizes 30, 35, 35 with varying inter/intra connection probabilities
    TEST_CASES.append((
        'sbm_100_3comm',
        lambda: nx.stochastic_block_model(
            sizes=[30, 35, 35],
            p=[[0.3, 0.05, 0.05],
               [0.05, 0.3, 0.05],
               [0.05, 0.05, 0.3]],
            seed=42
        ),
        [2, 3, 4]
    ))
    # SBM with weaker community structure (more challenging)
    TEST_CASES.append((
        'sbm_80_weak',
        lambda: nx.stochastic_block_model(
            sizes=[40, 40],
            p=[[0.2, 0.1],
               [0.1, 0.2]],
            seed=42
        ),
        [2, 3]
    ))


def run_algorithm_worker(args):
    """Worker function to run algorithm in separate process."""
    algorithm_name, G_data, k, num_threads = args

    G = nx.Graph()
    G.add_nodes_from(range(G_data['num_nodes']))
    G.add_edges_from(G_data['edges'])

    start_time = time.time()

    try:
        if algorithm_name == 'gurobi':
            solution, modularity = gurobi_k_mod_max(G, k)
            optimal = True
        elif algorithm_name == 'bnb_c':
            solution, modularity, initial_lb = solve_bnb_k_modularity_c(G, k, num_threads=num_threads)
            optimal = True
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        elapsed = time.time() - start_time
        return {
            'success': True,
            'time': elapsed,
            'modularity': modularity,
            'optimal': optimal,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'success': False,
            'time': elapsed,
            'error': str(e)
        }


def _run_worker_in_group(worker_func, args, result_queue):
    """Run worker in a new process group so entire tree can be killed on timeout."""
    os.setpgrp()
    try:
        result = worker_func(args)
        result_queue.put(result)
    except Exception as e:
        result_queue.put({'success': False, 'error': str(e)})


def run_with_timeout(algorithm_name, G, k, timeout, num_threads=8):
    """Run algorithm with timeout using multiprocessing."""
    G_data = {
        'num_nodes': G.number_of_nodes(),
        'edges': list(G.edges())
    }

    args = (algorithm_name, G_data, k, num_threads)

    result_queue = mp.Queue()
    process = mp.Process(
        target=_run_worker_in_group,
        args=(run_algorithm_worker, args, result_queue)
    )
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
        return {
            'success': False,
            'time': timeout,
            'error': 'timeout',
            'timed_out': True
        }

    if not result_queue.empty():
        return result_queue.get()
    return {'success': False, 'error': 'worker crashed'}


def run_experiment():
    """Run the static graph comparison experiment."""
    add_static_graphs()

    results = []
    algorithms = ['gurobi', 'bnb_c']

    total_runs = sum(len(k_values) for _, _, k_values in TEST_CASES) * len(algorithms)
    current_run = 0

    print(f"Starting Experiment: Static Graph Comparison (BnB C vs Gurobi)")
    print(f"Total runs: {total_runs}")
    print(f"Timeout: {TIMEOUT}s, Threads: {NUM_THREADS}")
    print("=" * 70)

    for graph_name, graph_generator, k_values in TEST_CASES:
        print(f"\n--- Graph: {graph_name} ---")

        G = graph_generator()
        n = G.number_of_nodes()
        num_edges = G.number_of_edges()
        print(f"    |V|={n}, |E|={num_edges}")

        for k in k_values:
            print(f"\n  k = {k}:")

            for alg in algorithms:
                current_run += 1
                print(f"    [{current_run}/{total_runs}] {alg}...", end=" ", flush=True)

                result = run_with_timeout(alg, G, k, TIMEOUT, NUM_THREADS)

                record = {
                    'graph_name': graph_name,
                    'n': n,
                    'num_edges': num_edges,
                    'k': k,
                    'algorithm': alg,
                    'time': result.get('time'),
                    'success': result.get('success', False),
                    'timed_out': result.get('timed_out', False),
                    'modularity': result.get('modularity'),
                    'optimal': result.get('optimal'),
                    'error': result.get('error'),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(record)

                if result.get('timed_out'):
                    print(f"TIMEOUT ({TIMEOUT}s)")
                elif result.get('success'):
                    print(f"{result['time']:.3f}s, Q={result['modularity']:.6f}")
                else:
                    print(f"FAILED: {result.get('error', 'unknown')}")

    return results


def save_results(results, output_dir):
    """Save results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"static_comparison_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return csv_path


def print_summary(results):
    """Print a summary comparison table."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)

    # Group by graph and k
    for graph_name in df['graph_name'].unique():
        graph_df = df[df['graph_name'] == graph_name]
        print(f"\n{graph_name}:")
        print("-" * 50)

        for k in sorted(graph_df['k'].unique()):
            k_df = graph_df[graph_df['k'] == k]
            print(f"  k={k}:")

            for _, row in k_df.iterrows():
                alg = row['algorithm']
                if row['timed_out']:
                    status = f"TIMEOUT ({TIMEOUT}s)"
                elif row['success']:
                    status = f"{row['time']:.3f}s, Q={row['modularity']:.6f}"
                else:
                    status = f"FAILED"
                print(f"    {alg:12s}: {status}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results")

    results = run_experiment()
    save_results(results, output_dir)
    print_summary(results)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()