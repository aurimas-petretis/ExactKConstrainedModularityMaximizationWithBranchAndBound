import gurobipy as gp
from gurobipy import GRB


def gurobi_k_mod_max(G, k):
    nodes = list(G.nodes())
    n = len(nodes)

    m = G.number_of_edges()

    two_m = 2 * m
    degree = dict(G.degree())

    model = gp.Model("k_modularity_maximization")
    x = model.addVars(n, k, vtype=GRB.BINARY, name="x")

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, c] for c in range(k)) == 1, name=f"assign_{i}")
    for c in range(k):
        model.addConstr(gp.quicksum(x[i, c] for i in range(n)) >= 1, name=f"nonempty_{c}")

    model.addConstr(x[0, 0] == 1, name="symmetry_node0")
    for c in range(1, k):
        for i in range(c):
            model.addConstr(x[i, c] == 0, name=f"symmetry_{i}_{c}")


    obj = gp.QuadExpr()
    for i in range(n):
        node_i = nodes[i]
        d_i = degree[node_i]
        for j in range(i + 1, n):
            node_j = nodes[j]
            d_j = degree[node_j]
            A_ij = 1 if G.has_edge(node_i, node_j) else 0
            B_ij = A_ij - (d_i * d_j) / two_m
            for c in range(k):
                obj += 2 * B_ij * x[i, c] * x[j, c]
    for i in range(n):
        node_i = nodes[i]
        d_i = degree[node_i]
        B_ii = 0 - (d_i * d_i) / two_m
        obj += B_ii

    model.setObjective(obj / two_m, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        # partition = {}
        # for i in range(n):
        #     for bnb_c in range(k):
        #         if x[i, bnb_c].X > 0.5:
        #             partition[nodes[i]] = bnb_c
        #             break

        partition = []
        for c in range(k):
            community = set()
            for i in range(n):
                if x[i, c].X > 0.5:
                    community.add(nodes[i])
            partition.append(community)

        return partition, model.ObjVal
    else:
        raise ValueError(f"Optimization failed with status {model.status}")