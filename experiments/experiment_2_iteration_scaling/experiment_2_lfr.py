"""
Experiment: Greedy Algorithm Iteration Scaling

Objective: Test how well the greedy algorithm (solve_bnb_k_modularity_cleiden)
finds the optimum when the iteration counter is increased.

For each iteration count, runs 20 times and records the modularity found.
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

from algorithms.bnb_c.bnb_cleiden import solve_bnb_k_modularity_cleiden
from algorithms.bnb_c.bnb_c import solve_bnb_k_modularity_c


# Configuration
ITERATION_VALUES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
MU_VALUES = [0.05, 0.1, 0.15]  # Mixing parameter values
NUM_RUNS = 100  # Number of different graphs (seeds) to test
K = 2  # Number of communities
N = 100  # Graph size
TIMEOUT = 60  # seconds

# LFR parameters
TAU1 = 2.05
TAU2 = 1.25


def generate_lfr_graph(n, mu, seed):
    """Generate LFR benchmark graph with given size, mu, and seed."""
    try:
        G = nx.LFR_benchmark_graph(
            n=n,
            tau1=TAU1,
            tau2=TAU2,
            mu=mu,
            min_degree=2,
            max_degree=max(3, n // 5),
            min_community=max(3, n // 4),
            max_community=max(4, n // 2),
            seed=seed
        )
        for node in G.nodes():
            if 'community' in G.nodes[node]:
                del G.nodes[node]['community']
        return G
    except nx.ExceededMaxIterations:
        return nx.planted_partition_graph(2, n // 2, 0.5, 0.05, seed=seed)


def run_greedy_worker(args):
    """Worker function to run greedy algorithm in separate process."""
    G_data, k, iterations = args

    G = nx.Graph()
    G.add_nodes_from(range(G_data['num_nodes']))
    G.add_edges_from(G_data['edges'])

    start_time = time.time()

    try:
        solution, _ = solve_bnb_k_modularity_cleiden(G, k, iterations=iterations)
        # Recalculate modularity using NetworkX for consistent comparison
        modularity = nx.community.modularity(G, solution)
        elapsed = time.time() - start_time
        return {
            'success': True,
            'time': elapsed,
            'modularity': modularity,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'success': False,
            'time': elapsed,
            'error': str(e)
        }


def run_exact_worker(args):
    """Worker function to run exact algorithm to get optimal modularity."""
    G_data, k = args

    G = nx.Graph()
    G.add_nodes_from(range(G_data['num_nodes']))
    G.add_edges_from(G_data['edges'])

    start_time = time.time()

    try:
        solution, _, _ = solve_bnb_k_modularity_c(G, k, num_threads=8)
        # Recalculate modularity using NetworkX for consistent comparison
        modularity = nx.community.modularity(G, solution)
        elapsed = time.time() - start_time
        return {
            'success': True,
            'time': elapsed,
            'modularity': modularity,
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


def run_with_timeout(worker_func, args, timeout):
    """Run algorithm with timeout using multiprocessing."""
    result_queue = mp.Queue()
    process = mp.Process(
        target=_run_worker_in_group,
        args=(worker_func, args, result_queue)
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
    """Run the iteration scaling experiment."""
    results = []

    print(f"Experiment: Greedy Algorithm Iteration Scaling")
    print(f"Graph: LFR with n={N}, k={K}")
    print(f"Mu values: {MU_VALUES}")
    print(f"Iteration values: {ITERATION_VALUES}")
    print(f"Number of different graphs (seeds) per mu: {NUM_RUNS}")
    print("=" * 60)

    for mu in MU_VALUES:
        print(f"\n{'='*60}")
        print(f"MU = {mu}")
        print(f"{'='*60}")

        # Pre-generate all graphs and find their optimal modularities
        print(f"\nGenerating {NUM_RUNS} graphs and finding optimal modularities...")
        graphs_data = []

        for seed in range(NUM_RUNS):
            G = generate_lfr_graph(N, mu, seed)
            num_edges = G.number_of_edges()

            G_data = {
                'num_nodes': G.number_of_nodes(),
                'edges': list(G.edges())
            }

            # Find optimal modularity for this graph
            exact_result = run_with_timeout(run_exact_worker, (G_data, K), TIMEOUT)
            if exact_result.get('success'):
                optimal_modularity = exact_result['modularity']
                print(f"  Graph {seed}: |E|={num_edges}, optimal Q={optimal_modularity:.6f} ({exact_result['time']:.2f}s)")
            else:
                optimal_modularity = None
                print(f"  Graph {seed}: |E|={num_edges}, optimal NOT FOUND ({exact_result.get('error', 'unknown')})")

            graphs_data.append({
                'seed': seed,
                'mu': mu,
                'G_data': G_data,
                'num_edges': num_edges,
                'optimal_modularity': optimal_modularity
            })

        total_runs = len(ITERATION_VALUES) * NUM_RUNS
        current_run = 0

        print(f"\n--- Running greedy algorithm experiments for mu={mu} ---")

        for iterations in ITERATION_VALUES:
            print(f"\nIterations = {iterations}:")

            for graph_info in graphs_data:
                current_run += 1
                seed = graph_info['seed']
                print(f"  [{current_run}/{total_runs}] Seed {seed}...", end=" ", flush=True)

                result = run_with_timeout(
                    run_greedy_worker,
                    (graph_info['G_data'], K, iterations),
                    TIMEOUT
                )

                optimal_modularity = graph_info['optimal_modularity']

                record = {
                    'n': N,
                    'k': K,
                    'mu': mu,
                    'graph_seed': seed,
                    'num_edges': graph_info['num_edges'],
                    'iterations': iterations,
                    'run': seed,
                    'time': result.get('time'),
                    'success': result.get('success', False),
                    'timed_out': result.get('timed_out', False),
                    'modularity': result.get('modularity'),
                    'optimal_modularity': optimal_modularity,
                    'error': result.get('error'),
                    'timestamp': datetime.now().isoformat()
                }

                # Calculate if optimal was found
                if optimal_modularity is not None and result.get('modularity') is not None:
                    record['found_optimal'] = abs(result['modularity'] - optimal_modularity) < 1e-9
                    record['gap'] = optimal_modularity - result['modularity']
                else:
                    record['found_optimal'] = None
                    record['gap'] = None

                results.append(record)

                if result.get('timed_out'):
                    print(f"TIMEOUT")
                elif result.get('success'):
                    gap_str = f", gap={record['gap']:.6f}" if record['gap'] is not None else ""
                    print(f"{result['time']:.3f}s, Q={result['modularity']:.6f}{gap_str}")
                else:
                    print(f"FAILED: {result.get('error', 'unknown')}")

    return results


def save_results(results, output_dir):
    """Save results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"experiment_2_lfr_{timestamp}.csv")
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