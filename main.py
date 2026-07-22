import time

import bayanpy
from algorithms.bnb_py.bnb import BnBModMaxSolver
from algorithms.bnb_c.bnb_c import solve_bnb_k_modularity_c
from algorithms.bnb_c.bnb_cleiden import solve_bnb_k_modularity_cleiden
from algorithms.gurobi_solution import gurobi_k_mod_max
from algorithms.igraph_solution_unconstrained import igraph_mod_max_from_nx
from graph_collection import get_graph
from painter import grapher
from algorithms.pyomo_solution import pyomo_k_mod_max

G = get_graph()
k = 2

# note: unconstrained algorithm
start_time = time.time()
bayan_modularity, optimality_gap, bayan_solution, modeling_time, solve_time = bayanpy.bayan(G, threshold=0.01, time_allowed=60, resolution=1)
bayan_time = time.time() - start_time

# note: unconstrained algorithm
start_time = time.time()
igraph_solution, igraph_modularity = igraph_mod_max_from_nx(G)
igraph_time = time.time() - start_time
#
# # pulp is very slow compared to others
# # start_time = time.time()
# # pulp_solution, pulp_modularity = pulp_k_mod_max(G, k)
# # pulp_time = time.time() - start_time
#
start_time = time.time()
gurobi_solution, gurobi_modularity = gurobi_k_mod_max(G, k)
gurobi_time = time.time() - start_time
#
start_time = time.time()
cplexpyomo_solution, cplexpyomo_modularity = pyomo_k_mod_max(G, k)
cplexpyomo_time = time.time() - start_time

start_time = time.time()
bnbpy_solution, bnbpy_modularity = BnBModMaxSolver(G).solve(k)
bnbpy_time = time.time() - start_time

start_time = time.time()
bnbc_solution, bnbc_modularity, bnbc_initial_lower_bound = solve_bnb_k_modularity_c(G, k, num_threads=8)
bnbc_time = time.time() - start_time

start_time = time.time()
cleiden_solution, cleiden_modularity = solve_bnb_k_modularity_cleiden(G, k, iterations=100)
cleiden_time = time.time() - start_time

print('Bayan results:')
print('time: ', bayan_time, 's')
print('solution', bayan_solution)
print('modularity', bayan_modularity)

print('Igraph results:')
print('time: ', igraph_time, 's')
print('solution', igraph_solution)
print('modularity', igraph_modularity)

# # print('Pulp results:')
# # print('time: ', pulp_time, 's')
# # print('solution', pulp_solution)
# # print('modularity', pulp_modularity)
#
print('Gurobi results:')
print('time: ', gurobi_time, 's')
print('solution', gurobi_solution)
print('modularity', gurobi_modularity)
#
print('CPLEX results:')
print('time: ', cplexpyomo_time, 's')
print('solution', cplexpyomo_solution)
print('modularity', cplexpyomo_modularity)

print('Branch and bound with Python results:')
print('time: ', bnbpy_time, 's')
print('solution', bnbpy_solution)
print('modularity', bnbpy_modularity)

print('Branch and bound with C results:')
print('time: ', bnbc_time, 's')
print('solution', bnbc_solution)
print('modularity', bnbc_modularity)
print('initial lower bound', bnbc_initial_lower_bound)

print('Leiden with C results:')
print('time: ', cleiden_time, 's')
print('solution', cleiden_solution)
print('modularity', cleiden_modularity)

grapher.draw_white_graph(G, 'test_graph_white')
grapher.draw_set_solution(G, bayan_solution, 'test_graph_optimal_communities')
grapher.draw_set_solution(G, bnbc_solution, 'test_graph_' + str(k) + '_communities')
