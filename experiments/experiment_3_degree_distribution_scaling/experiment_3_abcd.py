"""
Experiment 3: Degree Distribution Scaling Analysis on ABCD Benchmark

Objective: Determine how each algorithm scales with the degree distribution exponent gamma on ABCD graphs.
Fixed n=140, varying gamma.
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
N = 140  # Fixed node count
GAMMA_VALUES = [2, 2.2, 2.4, 2.6, 2.8, 3]
K = 2
NUM_SEEDS = 50
TIMEOUT = 30  # seconds
NUM_THREADS = 8  # for C implementation

# ABCD parameters
XI = 0.1


def generate_abcd_graph(n, gamma):
    """Generate ABCD benchmark graph with given size, gamma, and seed."""
    from abcd_graph import ABCDGraph, ABCDParams

    min_degree = 2
    max_degree = n // 3
    min_community = n // 3
    max_community = 2 * n // 3

    params = ABCDParams(
        gamma=gamma,
        vcount=n,
        xi=XI,
        min_degree=min_degree,
        max_degree=max_degree,
        min_community_size=min_community,
        max_community_size=max_community
    )
    graph = ABCDGraph(params).build()
    G = graph.exporter.to_networkx()

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)

    return G


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
    """Run the full degree distribution scaling experiment."""
    results = []
    algorithms = ['gurobi', 'bnb_c']

    total_runs = len(GAMMA_VALUES) * NUM_SEEDS * len(algorithms)
    current_run = 0

    print(f"Starting Experiment: ABCD Degree Distribution Scaling")
    print(f"Total runs: {total_runs}")
    print(f"n = {N}, gamma values: {GAMMA_VALUES}")
    print(f"k = {K}, seeds = {NUM_SEEDS}, timeout = {TIMEOUT}s")
    print(f"ABCD xi = {XI}")
    print("=" * 60)

    for gamma in GAMMA_VALUES:
        print(f"\n--- Testing gamma = {gamma} ---")

        for seed in range(NUM_SEEDS):
            G = generate_abcd_graph(N, gamma)
            actual_n = G.number_of_nodes()
            num_edges = G.number_of_edges()

            print(f"  Seed {seed}: |V|={actual_n}, |E|={num_edges}")

            for alg in algorithms:
                current_run += 1
                print(f"    [{current_run}/{total_runs}] Running {alg}...", end=" ", flush=True)

                result = run_with_timeout(alg, G, K, TIMEOUT, NUM_THREADS)

                record = {
                    'n': N,
                    'actual_n': actual_n,
                    'gamma': gamma,
                    'k': K,
                    'seed': seed,
                    'num_edges': num_edges,
                    'algorithm': alg,
                    'time': result.get('time'),
                    'success': result.get('success', False),
                    'timed_out': result.get('timed_out', False),
                    'modularity': result.get('modularity'),
                    'optimal': result.get('optimal'),
                    'error': result.get('error'),
                    'graph_type': 'abcd',
                    'xi': XI,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(record)

                if result.get('timed_out'):
                    print(f"TIMEOUT")
                elif result.get('success'):
                    print(f"{result['time']:.2f}s, Q={result['modularity']:.4f}")
                else:
                    print(f"FAILED: {result.get('error', 'unknown')}")

    return results


def save_results(results, output_dir):
    """Save results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"experiment_3_abcd_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return csv_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results")

    results = run_experiment()
    save_results(results, output_dir)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()