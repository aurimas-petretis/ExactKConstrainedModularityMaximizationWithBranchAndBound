"""
Experiment 6: Minimal Noise Scaling Analysis on Relaxed Caveman Graph

Objective: Determine how each algorithm scales with noise (rewiring probability p)
on relaxed caveman graphs.
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
N = 2000
K = 2
NUM_CLIQUES = 2
CLIQUE_SIZE = N // NUM_CLIQUES
NUM_SEEDS = 50
TIMEOUT = 30  # seconds
NUM_THREADS = 8  # for C implementation

# Rewiring probability values (noise parameter)
P_VALUES = [
    0.0001, 0.0002, 0.0005,
    0.001, 0.002, 0.003, 0.004, 0.005
]


def generate_relaxed_caveman_graph(p, seed):
    """Generate relaxed caveman graph with given rewiring probability and seed."""
    G = nx.relaxed_caveman_graph(NUM_CLIQUES, CLIQUE_SIZE, p, seed=seed)
    return G


def run_algorithm_worker(args):
    """Worker function to run algorithm in separate process."""
    algorithm_name, G_data, k, num_threads = args

    if 'graph_params' in G_data:
        # Generate graph in worker to avoid serializing large edge lists
        params = G_data['graph_params']
        G = nx.relaxed_caveman_graph(params['num_cliques'], params['clique_size'],
                                      params['p'], seed=params['seed'])
    else:
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


def run_with_timeout(algorithm_name, G, k, timeout, num_threads=8, graph_params=None):
    """Run algorithm with timeout using multiprocessing."""
    if graph_params is not None:
        # Pass generation params instead of edge list to avoid serialization overhead
        G_data = {'graph_params': graph_params}
    else:
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


FIELDNAMES = ['n', 'actual_n', 'k', 'num_cliques', 'clique_size', 'seed', 'num_edges', 'p',
              'algorithm', 'time', 'success', 'timed_out', 'modularity', 'optimal', 'error',
              'graph_type', 'timestamp']


def run_experiment(csv_path):
    """Run the full noise scaling experiment."""
    algorithms = ['gurobi', 'bnb_c']

    total_runs = len(P_VALUES) * NUM_SEEDS * len(algorithms)
    current_run = 0

    # Write CSV header
    write_header = not os.path.exists(csv_path)
    if write_header:
        with open(csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    print(f"Starting Experiment: Relaxed Caveman Noise Scaling")
    print(f"Total runs: {total_runs}")
    print(f"P values: {P_VALUES}")
    print(f"k = {K}, num_cliques = {NUM_CLIQUES}, clique_size = {CLIQUE_SIZE}")
    print(f"seeds = {NUM_SEEDS}, timeout = {TIMEOUT}s")
    print(f"Results file: {csv_path}")
    print("=" * 60)

    for p in P_VALUES:
        print(f"\n--- Testing p = {p} ---")

        for seed in range(NUM_SEEDS):
            G = generate_relaxed_caveman_graph(p, seed)
            actual_n = G.number_of_nodes()
            num_edges = G.number_of_edges()

            print(f"  Seed {seed}: |V|={actual_n}, |E|={num_edges}, p={p}")

            graph_params = {
                'num_cliques': NUM_CLIQUES,
                'clique_size': CLIQUE_SIZE,
                'p': p,
                'seed': seed,
            }

            for alg in algorithms:
                current_run += 1
                print(f"    [{current_run}/{total_runs}] Running {alg}...", end=" ", flush=True)

                result = run_with_timeout(alg, G, K, TIMEOUT, NUM_THREADS, graph_params=graph_params)

                record = {
                    'n': N,
                    'actual_n': actual_n,
                    'k': K,
                    'num_cliques': NUM_CLIQUES,
                    'clique_size': CLIQUE_SIZE,
                    'seed': seed,
                    'num_edges': num_edges,
                    'p': p,
                    'algorithm': alg,
                    'time': result.get('time'),
                    'success': result.get('success', False),
                    'timed_out': result.get('timed_out', False),
                    'modularity': result.get('modularity'),
                    'optimal': result.get('optimal'),
                    'error': result.get('error'),
                    'graph_type': 'relaxed_caveman',
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
    csv_path = os.path.join(output_dir, f"experiment_6_relaxed_caveman_results_{timestamp}.csv")

    run_experiment(csv_path)

    print(f"\nExperiment complete! Results saved to: {csv_path}")


if __name__ == "__main__":
    main()