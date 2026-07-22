"""
Experiment 5: K Scaling Analysis on Karate Club Graph

Objective: Determine how each algorithm scales with the number of communities k (2..10)
on Zachary's karate club graph. Compare exact methods: bnb_c, gurobi, igraph, and bayan.

Note: igraph and bayan find optimal modularity without k constraint (unconstrained k),
so they return the same result regardless of k. They are included for reference.
"""

import time
import os
import signal
import subprocess
import json
import multiprocessing as mp
from datetime import datetime
import csv

import networkx as nx

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.gurobi_solution import gurobi_k_mod_max
from algorithms.igraph_solution_unconstrained import igraph_mod_max_from_nx
from algorithms.bnb_c.bnb_c import solve_bnb_k_modularity_c


# Configuration
# K_VALUES = list(range(2, 35))  # k = 2, 3, ..., 5
K_VALUES = [34, 33]
NUM_SEEDS = 1  # Karate graph is deterministic, but we keep seeds for bnb_c randomness
TIMEOUT = 600  # seconds
NUM_THREADS = 8  # for C implementation


def get_karate_graph():
    """Return Zachary's karate club graph."""
    return nx.karate_club_graph()


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
            n_communities = len(solution)
            optimal = True
        elif algorithm_name == 'bnb_c':
            solution, modularity, initial_lb = solve_bnb_k_modularity_c(
                G, k, leiden_iterations=10000, num_threads=num_threads
            )
            n_communities = len(solution)
            optimal = True
        elif algorithm_name == 'igraph':
            vertex_clustering, modularity = igraph_mod_max_from_nx(G)
            n_communities = max(vertex_clustering.membership) + 1
            optimal = True
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        elapsed = time.time() - start_time
        return {
            'success': True,
            'time': elapsed,
            'modularity': modularity,
            'optimal': optimal,
            'n_communities': n_communities,
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


def run_bayan(G, timeout):
    """Run bayan in a separate Python process (bayan/Gurobi hangs in multiprocessing)."""
    edges = list(G.edges())
    script = f"""
import json, time, networkx as nx, bayanpy
G = nx.Graph()
G.add_nodes_from(range({G.number_of_nodes()}))
G.add_edges_from({edges})
start = time.time()
modularity, gap, solution, model_time, solve_time = bayanpy.bayan(G, threshold=0.01, time_allowed={timeout}, resolution=1)
elapsed = time.time() - start
n_communities = len(set(map(tuple, solution))) if solution is not None else None
print(json.dumps({{"time": elapsed, "modularity": modularity, "gap": gap, "n_communities": n_communities}}))
"""
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout + 5,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT}
        )
        if result.returncode == 0:
            # Parse the last line as JSON (earlier lines may be Gurobi license output)
            for line in reversed(result.stdout.strip().split('\n')):
                try:
                    data = json.loads(line)
                    return {
                        'success': True,
                        'time': data['time'],
                        'modularity': data['modularity'],
                        'optimal': data['gap'] <= 0.01,
                        'n_communities': data['n_communities'],
                    }
                except json.JSONDecodeError:
                    continue
        return {'success': False, 'time': time.time() - start_time, 'error': result.stderr[-200:]}
    except subprocess.TimeoutExpired:
        return {'success': False, 'time': timeout, 'error': 'timeout', 'timed_out': True}


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


FIELDNAMES = ['k', 'seed', 'n', 'num_edges', 'algorithm', 'time',
              'success', 'timed_out', 'modularity', 'optimal',
              'n_communities', 'error', 'graph_type', 'timestamp']


def run_experiment(csv_path):
    """Run the full k scaling experiment on karate graph."""
    k_constrained_algs = ['bnb_c', 'gurobi']
    # unconstrained_algs = ['igraph', 'bayan']
    unconstrained_algs = []

    G = get_karate_graph()
    n = G.number_of_nodes()
    num_edges = G.number_of_edges()

    all_algorithms = k_constrained_algs + unconstrained_algs
    total_runs = len(K_VALUES) * NUM_SEEDS * len(k_constrained_algs) + len(unconstrained_algs)
    current_run = 0

    # Write CSV header
    write_header = not os.path.exists(csv_path)
    if write_header:
        with open(csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    print(f"Starting Experiment 5: K Scaling on Karate Club Graph")
    print(f"Graph: |V|={n}, |E|={num_edges}")
    print(f"Algorithms ({len(all_algorithms)}): {all_algorithms}")
    print(f"  K-constrained (run per k): {k_constrained_algs}")
    print(f"  Unconstrained (run once):  {unconstrained_algs}")
    print(f"K values: {K_VALUES}")
    print(f"Seeds = {NUM_SEEDS}, timeout = {TIMEOUT}s")
    print(f"Total algorithm runs: {total_runs}")
    print(f"Results file: {csv_path}")
    print("=" * 60)

    def write_record(alg, k, seed, result):
        record = {
            'k': k, 'seed': seed, 'n': n, 'num_edges': num_edges,
            'algorithm': alg, 'time': result.get('time'),
            'success': result.get('success', False),
            'timed_out': result.get('timed_out', False),
            'modularity': result.get('modularity'),
            'optimal': result.get('optimal'),
            'n_communities': result.get('n_communities'),
            'error': result.get('error'),
            'graph_type': 'karate',
            'timestamp': datetime.now().isoformat()
        }
        with open(csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(record)

    # Unconstrained algorithms run once (result independent of k)
    print(f"\n--- Unconstrained algorithms (run once) ---")
    for alg in unconstrained_algs:
        current_run += 1
        print(f"    [{current_run}/{total_runs}] Running {alg}...", end=" ", flush=True)
        if alg == 'bayan':
            result = run_bayan(G, TIMEOUT)
        else:
            result = run_with_timeout(alg, G, None, TIMEOUT, NUM_THREADS)
        k_found = result.get('n_communities')
        if result.get('success'):
            print(f"{result['time']:.2f}s, Q={result['modularity']:.4f}, k_found={k_found}*")
        elif result.get('timed_out'):
            print(f"TIMEOUT")
        else:
            print(f"FAILED: {result.get('error', 'unknown')}")
        write_record(alg, None, 0, result)

    # K-constrained algorithms run for each k
    for k in K_VALUES:
        print(f"\n--- Testing k = {k} ---")
        for seed in range(NUM_SEEDS):
            for alg in k_constrained_algs:
                current_run += 1
                print(f"    [{current_run}/{total_runs}] Running {alg}...", end=" ", flush=True)
                result = run_with_timeout(alg, G, k, TIMEOUT, NUM_THREADS)
                write_record(alg, k, seed, result)

                if result.get('timed_out'):
                    print(f"TIMEOUT")
                elif result.get('success'):
                    print(f"{result['time']:.2f}s, Q={result['modularity']:.4f}, k_found={result.get('n_communities')}")
                else:
                    print(f"FAILED: {result.get('error', 'unknown')}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"experiment_5_karate_results_{timestamp}.csv")

    run_experiment(csv_path)

    print(f"\nExperiment complete! Results saved to: {csv_path}")


if __name__ == "__main__":
    main()