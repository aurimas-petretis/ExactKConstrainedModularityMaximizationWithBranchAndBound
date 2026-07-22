import igraph
import networkx as nx

def igraph_mod_max_from_nx(graph):
    edges = list(graph.edges())
    n_nodes = graph.number_of_nodes()
    ig_graph = igraph.Graph(n_nodes, edges)
    # Call igraph's exact optimal modularity method
    vertex_clustering = ig_graph.community_optimal_modularity()
    # Convert igraph's membership to NetworkX partition format
    igraph_partition = []
    n_communities = max(vertex_clustering.membership) + 1
    for comm_id in range(n_communities):
        community = {node for node, comm in enumerate(vertex_clustering.membership) if comm == comm_id}
        igraph_partition.append(community)
    print(f'Igraph partition with {len(igraph_partition)} communities: {igraph_partition}')
    igraph_exact_modularity = nx.community.modularity(graph, igraph_partition, weight='none')
    print(f'Igraph exact modularity (theirs): {vertex_clustering.modularity:.6f}, (ours): {igraph_exact_modularity:.6f}')
    return vertex_clustering, igraph_exact_modularity