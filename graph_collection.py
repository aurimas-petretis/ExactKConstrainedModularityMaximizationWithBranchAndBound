import networkx as nx


def get_ring_of_cliques(num_cliques=2, clique_size=495):
    """Ring of cliques - clear community structure."""
    return nx.ring_of_cliques(num_cliques, clique_size)


def get_clique_with_antenna(clique_size=20):
    G = nx.caveman_graph(1, clique_size)
    G.add_node(clique_size)
    G.add_edge(0, clique_size)
    return G


def get_clique_with_t_antennas(clique_size=20, t=2):
    G = nx.caveman_graph(1, clique_size)
    for i in range(t):
        G.add_node(clique_size)
        G.add_edge(0, clique_size+i)
    return G


def get_bipartite_graph_first_vertices_connected(size_1=2, size_2=3):
    G = nx.bipartite.complete_bipartite_graph(size_1, size_2)
    G.add_edge(0, 1)
    return G


def get_lfr_benchmark(n=90, tau1=2.05, tau2=1.25, mu=0.1, seed=0):
    """
    LFR benchmark graph with planted communities.

    Args:
        n: Number of nodes
        tau1: Power-law exponent for degree distribution
        tau2: Power-law exponent for community size distribution
        mu: Mixing parameter (lower = better separated communities)
        seed: Random seed
    """
    return nx.LFR_benchmark_graph(
        n=n,
        tau1=tau1,
        tau2=tau2,
        mu=mu,
        min_degree=2,
        max_degree=n,
        min_community=n // 3 + 1,
        max_community=2 * n // 3,
        seed=seed
    )


def get_caveman(num_cliques=2, clique_size=900):
    """Caveman graph - perfect community structure (disconnected cliques)."""
    return nx.caveman_graph(num_cliques, clique_size)


def get_connected_caveman(num_cliques=2, clique_size=900):
    """Connected caveman graph - cliques with single connecting edges."""
    return nx.connected_caveman_graph(num_cliques, clique_size)


def get_relaxed_caveman(num_cliques=2, clique_size=900, p=0.001, seed=0):
    """Relaxed caveman graph - cliques with rewired edges."""
    return nx.relaxed_caveman_graph(num_cliques, clique_size, p, seed=seed)


def get_planted_partition(num_groups=2, group_size=60, p_in=0.5, p_out=0.05, seed=0):
    """Planted partition graph - uniform community sizes with controlled edge probabilities."""
    return nx.planted_partition_graph(num_groups, group_size, p_in, p_out, seed=seed)


def get_stochastic_block_model(sizes=None, p_matrix=None, seed=0):
    """
    Stochastic block model - flexible community structure.

    Args:
        sizes: List of community sizes (default: [60, 60])
        p_matrix: Probability matrix for inter/intra community edges
        seed: Random seed
    """
    if sizes is None:
        sizes = [60, 60]
    if p_matrix is None:
        n_communities = len(sizes)
        p_in, p_out = 0.5, 0.05
        p_matrix = [[p_in if i == j else p_out for j in range(n_communities)]
                    for i in range(n_communities)]
    return nx.stochastic_block_model(sizes, p_matrix, seed=seed)


def get_degree_clustered(n_high=50, n_low=50, p_high=0.6, p_low=0.15, p_between=0.02, seed=0):
    """
    Graph where high-degree nodes cluster together and low-degree nodes cluster together.

    Creates two communities with distinct degree profiles:
    - High-degree community: densely connected hub nodes
    - Low-degree community: sparsely connected peripheral nodes

    Args:
        n_high: Number of nodes in high-degree community
        n_low: Number of nodes in low-degree community
        p_high: Edge probability within high-degree community (dense)
        p_low: Edge probability within low-degree community (sparse)
        p_between: Edge probability between communities (very sparse)
        seed: Random seed
    """
    # Use stochastic block model with asymmetric density
    sizes = [n_high, n_low]
    p_matrix = [
        [p_high, p_between],
        [p_between, p_low]
    ]
    return nx.stochastic_block_model(sizes, p_matrix, seed=seed)


def get_abcd_benchmark(n=140, xi=0.1, min_degree=2, max_degree=None, min_community=10, max_community=None, seed=0):
    """
    ABCD (Artificial Benchmark for Community Detection) graph.

    Faster and more interpretable alternative to LFR with similar properties.
    Requires: pip install abcd-graph[networkx]

    Args:
        n: Number of nodes
        xi: Noise parameter (similar to LFR's mu, lower = better separated communities)
        min_degree: Minimum node degree
        max_degree: Maximum node degree (default: n // 10)
        min_community: Minimum community size
        max_community: Maximum community size (default: n // 3)
        seed: Random seed
    """
    from abcd_graph import ABCDGraph, ABCDParams

    if max_degree is None:
        max_degree = max(min_degree + 1, n // 10)
    if max_community is None:
        max_community = max(min_community + 1, n // 3)

    params = ABCDParams(
        vcount=n,
        xi=xi,
        min_degree=min_degree,
        max_degree=max_degree,
        min_community_size=min_community,
        max_community_size=max_community
    )
    graph = ABCDGraph(params).build()
    return graph.exporter.to_networkx()


def get_graph():
    # benchmark graphs
    # G = nx.karate_club_graph()
    # G = nx.florentine_families_graph()
    G = nx.les_miserables_graph()

    # bnb outperforms gurobi in (near) perfect graphs
    # but becomes slower when more noise is introduced
    # G = get_caveman()
    # G = get_connected_caveman()
    # G = get_relaxed_caveman()
    # G = get_ring_of_cliques(2, 495)  # gurobi limit - n=3000,k=2 (30s) (sometimes it's much lower?)
    # G = get_clique_with_t_antennas(10, 15)
    # G = get_bipartite_graph_first_vertices_connected(2, 5)

    # gurobi outperforms bnb in these type of graphs
    # - mainly because pruning is less effective with vertices of about equal degrees
    # G = get_planted_partition()
    # G = get_stochastic_block_model()
    # G = get_degree_clustered()

    # gurobi limit - n=180,tau1=2.35,k=2 (8s)
    # varies for different graphs
    # G = get_lfr_benchmark(n=140, tau1=3, tau2=1.25, mu=0.1, seed=0)

    # with vcount=120 gurobi fails?
    # varies for different graphs
    # G = get_abcd_benchmark(n=120)

    return nx.convert_node_labels_to_integers(G, label_attribute='name')