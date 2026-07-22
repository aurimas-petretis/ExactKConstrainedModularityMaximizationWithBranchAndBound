"""
Experiment 1: Size Scaling Analysis on LFR Benchmark

Objective: Determine how each algorithm scales with graph size n on LFR graphs.
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
from algorithms.bnb_c.bnb_cleiden import solve_bnb_k_modularity_cleiden
from algorithms.pyomo_solution import pyomo_k_mod_max
from algorithms.bnb_py.bnb import BnBModMaxSolver


# Configuration
N_VALUES = [
    20, 40, 60, 80, 100,
    120, 140,
    160, 180, 200
]
K = 2
NUM_SEEDS = 20
TIMEOUT = 30  # seconds
NUM_THREADS = 8  # for C implementation

# LFR parameters
TAU1 = 2.05
TAU2 = 1.25
MU = 0.1


def generate_lfr_graph(n, seed):
    """Generate LFR benchmark graph with given size and seed."""
    try:
        G = nx.LFR_benchmark_graph(
            n=n,
            tau1=TAU1,
            tau2=TAU2,
            mu=MU,
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
        elif algorithm_name == 'bnb_c_seq':
            solution, modularity, initial_lb = solve_bnb_k_modularity_c(G, k)
            optimal = True
        elif algorithm_name == 'bnb_c':
            solution, modularity, initial_lb = solve_bnb_k_modularity_c(G, k, num_threads=num_threads)
            optimal = True
        elif algorithm_name == 'pyomo_cplex':
            solution, modularity = pyomo_k_mod_max(G, k)
            optimal = True
        elif algorithm_name == 'bnb_py':
            solver = BnBModMaxSolver(G)
            solution, modularity = solver.solve(k)
            optimal = True
        elif algorithm_name == 'leiden_c_x100':
            solution, modularity = solve_bnb_k_modularity_cleiden(G, k)
            optimal = None  # Will be determined by comparison with exact algorithms
        elif algorithm_name == 'igraph':
            vertex_clustering, modularity = igraph_mod_max_from_nx(G)
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


def run_bayan(G, timeout):
    """Run bayan in a separate Python process (bayan/Gurobi hangs in multiprocessing)."""
    edges = list(G.edges())
    script = f"""
import json, time, numpy as np, networkx as nx, bayanpy
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
            for line in reversed(result.stdout.strip().split('\n')):
                try:
                    data = json.loads(line)
                    return {
                        'success': True,
                        'time': data['time'],
                        'modularity': data['modularity'],
                        'optimal': data['gap'] <= 0.01,
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


FIELDNAMES = ['n', 'k', 'seed', 'num_edges', 'algorithm', 'time',
              'success', 'timed_out', 'modularity', 'optimal', 'error',
              'graph_type', 'timestamp']


def run_experiment(csv_path):
    """Run the full size scaling experiment."""
    # algorithms = ['gurobi', 'bnb_c_seq', 'bnb_c', 'bnb_c_pruned', 'pyomo_cplex', 'bnb_py', 'leiden_c_x100', 'igraph', 'bayan']
    algorithms = ['igraph', 'bayan']

    total_runs = len(N_VALUES) * NUM_SEEDS * len(algorithms)
    current_run = 0

    # Write CSV header
    write_header = not os.path.exists(csv_path)
    if write_header:
        with open(csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    print(f"Starting Experiment: LFR Size Scaling")
    print(f"Total runs: {total_runs}")
    print(f"N values: {N_VALUES}")
    print(f"k = {K}, seeds = {NUM_SEEDS}, timeout = {TIMEOUT}s")
    print(f"Results file: {csv_path}")
    print("=" * 60)

    for n in N_VALUES:
        print(f"\n--- Testing n = {n} ---")

        for seed in range(NUM_SEEDS):
            G = generate_lfr_graph(n, seed)
            num_edges = G.number_of_edges()

            print(f"  Seed {seed}: |V|={n}, |E|={num_edges}")

            for alg in algorithms:
                current_run += 1
                if alg in ('pyomo_cplex') and n > 100:
                    print(f"    [{current_run}/{total_runs}] Skipping {alg}...")
                    continue
                if alg in ('bnb_py') and n > 140:
                    print(f"    [{current_run}/{total_runs}] Skipping {alg}...")
                    continue
                print(f"    [{current_run}/{total_runs}] Running {alg}...", end=" ", flush=True)

                if alg == 'bayan':
                    result = run_bayan(G, TIMEOUT)
                else:
                    result = run_with_timeout(alg, G, K, TIMEOUT, NUM_THREADS)

                record = {
                    'n': n,
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
    csv_path = os.path.join(output_dir, f"experiment_1_lfr_results_{timestamp}.csv")

    run_experiment(csv_path)

    print(f"\nExperiment complete! Results saved to: {csv_path}")


if __name__ == "__main__":
    main()