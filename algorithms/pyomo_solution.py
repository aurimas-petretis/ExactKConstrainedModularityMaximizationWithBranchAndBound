from pyomo.environ import (
    ConcreteModel,
    Var,
    Binary,
    Objective,
    Constraint,
    maximize,
    SolverFactory,
    value,
)


def pyomo_k_mod_max(G, k, solver_name="cplex_direct"):
    nodes = list(G.nodes())
    n = len(nodes)

    m = G.number_of_edges()

    two_m = 2 * m
    degree = dict(G.degree())

    model = ConcreteModel("k_modularity_maximization")

    node_indices = range(n)
    community_indices = range(k)

    model.x = Var(node_indices, community_indices, domain=Binary)

    def assignment_rule(model, i):
        return sum(model.x[i, c] for c in community_indices) == 1

    model.assignment = Constraint(node_indices, rule=assignment_rule)

    def nonempty_rule(model, c):
        return sum(model.x[i, c] for i in node_indices) >= 1

    model.nonempty = Constraint(community_indices, rule=nonempty_rule)

    model.symmetry_node0 = Constraint(expr=model.x[0, 0] == 1)

    def symmetry_rule(model, i, c):
        if c >= 1 and i < c:
            return model.x[i, c] == 0
        return Constraint.Skip

    model.symmetry = Constraint(node_indices, community_indices, rule=symmetry_rule)

    obj_expr = 0
    for i in range(n):
        node_i = nodes[i]
        d_i = degree[node_i]
        for j in range(i + 1, n):
            node_j = nodes[j]
            d_j = degree[node_j]
            A_ij = 1 if G.has_edge(node_i, node_j) else 0
            B_ij = A_ij - (d_i * d_j) / two_m
            for c in range(k):
                obj_expr += 2 * B_ij * model.x[i, c] * model.x[j, c]
    for i in range(n):
        node_i = nodes[i]
        d_i = degree[node_i]
        B_ii = 0 - (d_i * d_i) / two_m
        obj_expr += B_ii

    model.objective = Objective(expr=obj_expr / two_m, sense=maximize)

    solver = SolverFactory(solver_name)
    result = solver.solve(model, tee=False)

    if result.solver.termination_condition.value == "optimal":
        partition = []
        for c in range(k):
            community = set()
            for i in range(n):
                if value(model.x[i, c]) > 0.5:
                    community.add(nodes[i])
            partition.append(community)

        return partition, value(model.objective)
    else:
        raise ValueError(
            f"Optimization failed with status {result.solver.termination_condition}"
        )