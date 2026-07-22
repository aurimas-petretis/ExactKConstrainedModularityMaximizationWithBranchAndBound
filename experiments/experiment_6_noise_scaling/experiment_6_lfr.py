"""
Experiment 4: Size Scaling Analysis on LFR Benchmark

Objective: Determine how each algorithm scales with graph size n on LFR graphs.
"""


import time
import os
import signal
import multiprocessing as mp
from datetime import datetime

import csv

import networkx as nx

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from algorithms.gurobi_solution import gurobi_k_mod_max
from algorithms.bnb_c.bnb_c import solve_bnb_k_modularity_c


# Configuration
N = 80
K = 2
NUM_SEEDS = 50
TIMEOUT = 120  # seconds
NUM_THREADS = 8  # for C implementation

# LFR parameters
TAU1 = 3
TAU2 = 1.25
MU_VALUES = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5
]


def generate_lfr_graph(mu, seed):
    """Generate LFR benchmark graph with given size and seed."""
    for attempt in range(10):
        try:
            G = nx.LFR_benchmark_graph(
                n=N,
                tau1=TAU1,
                tau2=TAU2,
                mu=mu,
                min_degree=2,
                max_degree=N // 5,
                min_community=N // 4,
                max_community=N // 2,
                seed=seed + attempt * 1000
            )
            for node in G.nodes():
                if 'community' in G.nodes[node]:
                    del G.nodes[node]['community']
            return G
        except nx.ExceededMaxIterations:
            continue
    raise RuntimeError(f"Failed to generate LFR graph for n={n}, seed={seed} after 10 attempts")


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
            solution, modularity, initial_lb = solve_bnb_k_modularity_c(G, k, leiden_iterations=100, num_threads=num_threads)
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
        # Kill entire process group (worker + child subprocesses like bnb_solver)
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


FIELDNAMES = ['n', 'k', 'seed', 'num_edges', 'mu', 'algorithm', 'time',
              'success', 'timed_out', 'modularity', 'optimal', 'error',
              'graph_type', 'timestamp']


def run_experiment(csv_path):
    """Run the full noise scaling experiment."""
    algorithms = ['gurobi', 'bnb_c']

    total_runs = len(MU_VALUES) * NUM_SEEDS * len(algorithms)
    current_run = 0

    # Write CSV header
    write_header = not os.path.exists(csv_path)
    if write_header:
        with open(csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    print(f"Starting Experiment: LFR Noise Scaling")
    print(f"Total runs: {total_runs}")
    print(f"MU values: {MU_VALUES}")
    print(f"k = {K}, seeds = {NUM_SEEDS}, timeout = {TIMEOUT}s")
    print(f"Results file: {csv_path}")
    print("=" * 60)

    for mu in MU_VALUES:
        print(f"\n--- Testing mu = {mu} ---")

        for seed in range(NUM_SEEDS):
            G = generate_lfr_graph(mu, seed)
            num_edges = G.number_of_edges()

            print(f"  Seed {seed}: |V|={N}, |E|={num_edges}")

            for alg in algorithms:
                current_run += 1
                print(f"    [{current_run}/{total_runs}] Running {alg}...", end=" ", flush=True)

                result = run_with_timeout(alg, G, K, TIMEOUT, NUM_THREADS)

                record = {
                    'n': N,
                    'k': K,
                    'seed': seed,
                    'num_edges': num_edges,
                    'mu': mu,
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

                # Append result to CSV immediately
                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(record)

                if result.get('timed_out'):
                    print(f"TIMEOUT")
                elif result.get('success'):
                    print(f"{result['time']:.2f}s, Q={result['modularity']:.4f}")
                else:
                    print(f"FAILED: {result.get('error', 'unknown')}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"experiment_6_lfr_results_{timestamp}.csv")

    run_experiment(csv_path)

    print(f"\nExperiment complete! Results saved to: {csv_path}")


if __name__ == "__main__":
    main()