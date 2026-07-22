"""
Experiment 3: Degree Distribution Scaling Analysis on LFR Benchmark

Objective: Determine how each algorithm scales with the degree distribution exponent tau1 on LFR graphs.
Fixed n=140, varying tau1.
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
TAU1_VALUES = [2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
K = 2
NUM_SEEDS = 50
TIMEOUT = 30  # seconds
NUM_THREADS = 8  # for C implementation

# LFR parameters
TAU2 = 1.25
MU = 0.1


def generate_lfr_graph(n, tau1, seed):
    """Generate LFR benchmark graph with given size, tau1, and seed."""
    G = nx.LFR_benchmark_graph(
        n=n,
        tau1=tau1,
        tau2=TAU2,
        mu=MU,
        min_degree=2,
        max_degree=n // 3,
        min_community=n // 3,
        max_community=2 * n // 3,
        seed=seed
    )
    for node in G.nodes():
        if 'community' in G.nodes[node]:
            del G.nodes[node]['community']
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

    total_runs = len(TAU1_VALUES) * NUM_SEEDS * len(algorithms)
    current_run = 0

    print(f"Starting Experiment: LFR Degree Distribution Scaling")
    print(f"Total runs: {total_runs}")
    print(f"n = {N}, tau1 values: {TAU1_VALUES}")
    print(f"k = {K}, seeds = {NUM_SEEDS}, timeout = {TIMEOUT}s")
    print("=" * 60)

    for tau1 in TAU1_VALUES:
        print(f"\n--- Testing tau1 = {tau1} ---")

        for seed in range(NUM_SEEDS):
            G = generate_lfr_graph(N, tau1, seed)
            num_edges = G.number_of_edges()

            print(f"  Seed {seed}: |V|={N}, |E|={num_edges}")

            for alg in algorithms:
                current_run += 1
                print(f"    [{current_run}/{total_runs}] Running {alg}...", end=" ", flush=True)

                result = run_with_timeout(alg, G, K, TIMEOUT, NUM_THREADS)

                record = {
                    'n': N,
                    'tau1': tau1,
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
                    'graph_type': 'lfr',
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
    csv_path = os.path.join(output_dir, f"experiment_3_lfr_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results")

    results = run_experiment()
    save_results(results, output_dir)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()