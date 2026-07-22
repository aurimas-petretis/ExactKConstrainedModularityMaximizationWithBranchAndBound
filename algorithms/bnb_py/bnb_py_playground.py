import networkx as nx
import numpy as np

from algorithms.bnb_py.bnb import BnBModMaxSolver
from painter import grapher


graph3v = nx.from_numpy_array(np.matrix([[0, 0, 0],
                                             [0, 0, 1],
                                             [0, 1, 0]]))
graph4v_bal = nx.from_numpy_array(np.matrix([[0, 1, 0, 0],
                                                 [1, 0, 0, 0],
                                                 [0, 0, 0, 1],
                                                 [0, 0, 1, 0]]))
graph4v_imbal = nx.from_numpy_array(np.matrix([[0, 0, 0, 0],
                                                   [0, 0, 1, 0],
                                                   [0, 1, 0, 1],
                                                   [0, 0, 1, 0]]))
graph4v_messy = nx.from_numpy_array(np.matrix([[0, 1, 1, 0],
                                                   [1, 0, 1, 0],
                                                   [1, 1, 0, 1],
                                                   [0, 0, 1, 0]]))
graph5v = nx.from_numpy_array(np.matrix([[0, 0, 0, 0, 0],
                                             [0, 0, 1, 0, 0],
                                             [0, 1, 0, 1, 0],
                                             [0, 0, 1, 0, 0],
                                             [0, 0, 0, 0, 0]]))
graph5v_imbal = nx.from_numpy_array(np.matrix([[0, 0, 0, 1, 0],
                                                   [0, 0, 1, 0, 1],
                                                   [0, 1, 0, 0, 0],
                                                   [1, 0, 0, 0, 0],
                                                   [0, 1, 0, 0, 0]]))
graph5v_line = nx.from_numpy_array(np.matrix([[0, 0, 0, 1, 0],
                                                  [0, 0, 1, 0, 1],
                                                  [0, 1, 0, 1, 0],
                                                  [1, 0, 1, 0, 0],
                                                  [0, 1, 0, 0, 0]]))
graph6v_2c_bal = nx.from_numpy_array(np.matrix([[0, 1, 1, 0, 0, 0],
                                                    [1, 0, 1, 0, 0, 0],
                                                    [1, 1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 1, 1],
                                                    [0, 0, 0, 1, 0, 1],
                                                    [0, 0, 0, 1, 1, 0]]))
graph6v_3c_bal = nx.from_numpy_array(np.matrix([[0, 1, 0, 0, 0, 0],
                                                    [1, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 1, 0, 0],
                                                    [0, 0, 1, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 1],
                                                    [0, 0, 0, 0, 1, 0]]))
graph8v_2c_bal = nx.from_numpy_array(np.matrix([[0, 1, 1, 1, 0, 0, 0, 0],
                                                    [1, 0, 1, 1, 0, 0, 0, 0],
                                                    [1, 1, 0, 1, 0, 0, 0, 0],
                                                    [1, 1, 1, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 1, 1],
                                                    [0, 0, 0, 0, 1, 0, 1, 1],
                                                    [0, 0, 0, 0, 1, 1, 0, 1],
                                                    [0, 0, 0, 0, 1, 1, 1, 0]]))
graph8v_full = nx.from_numpy_array(np.matrix([[0, 1, 1, 1, 1, 1, 1, 1],
                                                  [1, 0, 1, 1, 1, 1, 1, 1],
                                                  [1, 1, 0, 1, 1, 1, 1, 1],
                                                  [1, 1, 1, 0, 1, 1, 1, 1],
                                                  [1, 1, 1, 1, 0, 1, 1, 1],
                                                  [1, 1, 1, 1, 1, 0, 1, 1],
                                                  [1, 1, 1, 1, 1, 1, 0, 1],
                                                  [1, 1, 1, 1, 1, 1, 1, 0]]))
star_graph_8v = nx.from_numpy_array(np.matrix([[0, 1, 1, 1, 1, 1, 1, 1],
                                                   [1, 0, 0, 0, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0, 0, 0, 0]]))
graph8v_2c_imbal = nx.from_numpy_array(np.matrix([[0, 1, 1, 0, 0, 0, 0, 1],
                                                      [1, 0, 1, 0, 0, 0, 0, 1],
                                                      [1, 1, 0, 1, 0, 0, 0, 0],
                                                      [0, 0, 1, 0, 0, 1, 0, 0],
                                                      [0, 0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 1, 0, 1, 1],
                                                      [0, 0, 0, 0, 0, 1, 0, 1],
                                                      [1, 1, 0, 0, 1, 1, 1, 0]]))
ring_of_cliques_4_3_plus = nx.from_numpy_array(np.matrix([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                                                              [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                              [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                                              [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                                                              [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                              [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                                                              [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                                                              [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                                                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
                                                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                                                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
graph_karate_club = nx.karate_club_graph()
barbel_2c_8_graph = nx.barbell_graph(3, 1)
ring_of_cliques_4_3 = nx.ring_of_cliques(4, 3)
ring_of_cliques_4_5 = nx.ring_of_cliques(4, 5)
ring_of_cliques_4_8 = nx.ring_of_cliques(4, 8)
ring_of_cliques_10_20 = nx.ring_of_cliques(10, 20)
ring_of_cliques_3_20 = nx.ring_of_cliques(3, 20)

k = 2
graph = graph_karate_club
print(graph)
adj = nx.to_numpy_array(graph, weight='none')
print(adj)

for line in adj:
    for point in line:
        print(int(point), end=' ')
    print()

print('[', end='')
for line in adj:
    print('[', end='')
    for point in line:
        print(int(point), end=',')
    print(']')
print(']')

solver = BnBModMaxSolver(graph)
solution, best_value = solver.solve(k, None, None)
print(f'Solution with {len(solution) if solution else 0} communities:', solution)
print("Modularity value:", best_value)

if solver.params.track_bound_history:
    grapher.plot_bounds_evolution(solver.upper_bound_history, solver.lower_bound_history, None, 'Bounds evolution')

grapher.draw_set_solution(graph, solution, None, '')

print('=============== Modularity verification ===============')
bnb_mod = solver.modularity(solution)
bnb_nx_mod = nx.community.modularity(graph, solution, weight='none')
print(f'bnb modularity: {bnb_mod}, networkx modularity: {bnb_nx_mod}')

