from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, LpStatus


def pulp_k_mod_max(G, k):
    """
    Solve k-constrained modularity maximization using PuLP.

    This linearizes the quadratic objective by introducing auxiliary variables
    y[i,j,bnb_c] = x[i,bnb_c] * x[j,bnb_c] for pairs of nodes (i,j) and communities bnb_c.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    m = G.number_of_edges()
    two_m = 2 * m
    degree = dict(G.degree())

    model = LpProblem("k_modularity_maximization", LpMaximize)

    # Binary variables: x[i,bnb_c] = 1 if node i is in community bnb_c
    x = {(i, c): LpVariable(f"x_{i}_{c}", cat=LpBinary)
         for i in range(n) for c in range(k)}

    # Each node belongs to exactly one community
    for i in range(n):
        model += lpSum(x[i, c] for c in range(k)) == 1, f"assign_{i}"

    # Each community must have at least one node
    for c in range(k):
        model += lpSum(x[i, c] for i in range(n)) >= 1, f"nonempty_{c}"

    # Symmetry breaking constraints
    model += x[0, 0] == 1, "symmetry_node0"
    for c in range(1, k):
        for i in range(c):
            model += x[i, c] == 0, f"symmetry_{i}_{c}"

    # Linearization: y[i,j,bnb_c] = x[i,bnb_c] * x[j,bnb_c] for i < j
    # Using McCormick envelope for binary variables:
    # y <= x[i,bnb_c], y <= x[j,bnb_c], y >= x[i,bnb_c] + x[j,bnb_c] - 1, y >= 0
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            for c in range(k):
                y_var = LpVariable(f"y_{i}_{j}_{c}", lowBound=0, upBound=1)
                y[i, j, c] = y_var
                model += y_var <= x[i, c], f"lin1_{i}_{j}_{c}"
                model += y_var <= x[j, c], f"lin2_{i}_{j}_{c}"
                model += y_var >= x[i, c] + x[j, c] - 1, f"lin3_{i}_{j}_{c}"

    # Build objective: maximize modularity
    # Q = (1/2m) * sum_{i,j} B_ij * delta(c_i, c_j)
    # where B_ij = A_ij - (d_i * d_j) / (2m)
    obj_terms = []

    # Off-diagonal terms (i < j), counted twice in original formula
    for i in range(n):
        node_i = nodes[i]
        d_i = degree[node_i]
        for j in range(i + 1, n):
            node_j = nodes[j]
            d_j = degree[node_j]
            A_ij = 1 if G.has_edge(node_i, node_j) else 0
            B_ij = A_ij - (d_i * d_j) / two_m
            for c in range(k):
                obj_terms.append(2 * B_ij * y[i, j, c])

    # Diagonal terms B_ii (constant, but included for completeness)
    constant_term = 0
    for i in range(n):
        node_i = nodes[i]
        d_i = degree[node_i]
        B_ii = 0 - (d_i * d_i) / two_m
        constant_term += B_ii

    # Objective: (sum of terms + constant) / (2m)
    model += (lpSum(obj_terms) + constant_term) / two_m

    # Solve
    model.solve()

    if LpStatus[model.status] == "Optimal":
        partition = []
        for c in range(k):
            community = set()
            for i in range(n):
                if x[i, c].varValue is not None and x[i, c].varValue > 0.5:
                    community.add(nodes[i])
            partition.append(community)

        obj_value = model.objective.value()
        return partition, obj_value
    else:
        raise ValueError(f"Optimization failed with status {LpStatus[model.status]}")