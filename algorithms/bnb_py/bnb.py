import math
import networkx as nx
import random


class BnBParameters:
    def __init__(self,
                 max_recursive_calls=5000000,
                 vertex_ordering='degree_desc',
                 enable_pruning=True,
                 enable_debugging=False,
                 leiden_lower_bound_iterations=100,
                 track_bound_history=True,
                 provide_partial_clustering=False,
                 search_up_to_k_communities=False,
                 search_strategy='dfs'):
        """
        Initialize BnB parameters.

        Args:
            max_recursive_calls: Maximum number of recursive calls allowed in the search tree
            vertex_ordering: Vertex ordering technique. Options:
                - 'degree_desc': Order by degree in descending order
                - 'random': Shuffle randomly
                - 'binomial': Adjust degrees to binomial distribution
                - 'none': No ordering changes applied (original order)
            enable_pruning: Enable branch pruning
            enable_debugging: Enable debug output
            leiden_lower_bound_iterations: Number of independent Leiden iterations to run for initial lower bound.
                - 0 or None: No Leiden initialization
                - n > 0: Run n independent iterations and use the best result
            track_bound_history: Track bound evolution over iterations
            provide_partial_clustering: Provide partial known clustering to lower the search space
            search_up_to_k_communities: Search for up to k communities. If false, then search for exactly k.
            search_strategy: Search strategy to use. Options:
                - 'dfs': Depth-first search - explores deep branches first, lower memory usage
                - 'bfs': Breadth-first search - explores level by level, more uniform exploration
                - 'bestfs': Best-first search by community - processes next vertex, explores community assignments sorted by modularity gain
                - 'localfs': Local-first search - initially dives deep following heuristic solution path, then explores alternatives
        """
        self.max_recursive_calls = max_recursive_calls
        self.vertex_ordering = vertex_ordering
        self.enable_pruning = enable_pruning
        self.enable_debugging = enable_debugging
        self.leiden_lower_bound_iterations = leiden_lower_bound_iterations
        self.track_bound_history = track_bound_history
        self.provide_partial_clustering = provide_partial_clustering
        self.search_up_to_k_communities = search_up_to_k_communities
        self.search_strategy = search_strategy


class IncrementalBoundState:

    def __init__(self, solver, remaining_nodes):
        self.solver = solver
        self.remaining_set = set(remaining_nodes)
        self.cluster_degree_sums = []
        self.edges_between_remaining = 0
        for i, node1 in enumerate(remaining_nodes):
            for node2 in remaining_nodes[i+1:]:
                if solver.adj[node1][node2]:
                    self.edges_between_remaining += 1
        self.node_to_cluster_edges = {node: [] for node in remaining_nodes}
        self.remaining_degree_penalty = sum(
            (solver.degrees[node] / (2 * solver.m)) ** 2
            for node in remaining_nodes
        )


    def copy(self):
        """Create a shallow copy for branching."""
        new_state = IncrementalBoundState.__new__(IncrementalBoundState)
        new_state.solver = self.solver
        new_state.remaining_set = self.remaining_set.copy()
        new_state.cluster_degree_sums = self.cluster_degree_sums.copy()
        new_state.edges_between_remaining = self.edges_between_remaining
        # Deep copy the node_to_cluster_edges dict (each value is a list)
        new_state.node_to_cluster_edges = {
            node: edges.copy()
            for node, edges in self.node_to_cluster_edges.items()
        }
        new_state.remaining_degree_penalty = self.remaining_degree_penalty
        return new_state


    def update_for_join_cluster(self, vertex, cluster_idx):
        """Update state when vertex joins an existing cluster."""
        solver = self.solver

        # 1. Update cluster degree sum
        self.cluster_degree_sums[cluster_idx] += solver.degrees[vertex]

        # 2. Remove vertex from remaining
        self.remaining_set.remove(vertex)
        del self.node_to_cluster_edges[vertex]

        # 3. Update edges_between_remaining: subtract edges from vertex to other remaining
        for other in self.remaining_set:
            if solver.adj[vertex][other]:
                self.edges_between_remaining -= 1

        # 4. Update node_to_cluster_edges for each remaining node
        #    Each remaining node now has vertex in cluster cluster_idx
        for other in self.remaining_set:
            if solver.adj[vertex][other]:
                self.node_to_cluster_edges[other][cluster_idx] += 1

        # 5. Update remaining degree penalty
        self.remaining_degree_penalty -= (solver.degrees[vertex] / (2 * solver.m)) ** 2


    def update_for_new_singleton(self, vertex):
        """Update state when vertex creates a new singleton cluster."""
        solver = self.solver

        # 1. Add new cluster with this vertex's degree
        self.cluster_degree_sums.append(solver.degrees[vertex])

        # 2. Remove vertex from remaining
        self.remaining_set.remove(vertex)
        del self.node_to_cluster_edges[vertex]

        # 3. Update edges_between_remaining: subtract edges from vertex to other remaining
        for other in self.remaining_set:
            if solver.adj[vertex][other]:
                self.edges_between_remaining -= 1

        # 4. Add new column to node_to_cluster_edges for each remaining node
        for other in self.remaining_set:
            edges_to_new_cluster = 1 if solver.adj[vertex][other] else 0
            self.node_to_cluster_edges[other].append(edges_to_new_cluster)

        # 5. Update remaining degree penalty
        self.remaining_degree_penalty -= (solver.degrees[vertex] / (2 * solver.m)) ** 2


class BnBModMaxSolver:
    def __init__(self, graph, parameters=BnBParameters()):
        self.graph = graph
        self.params = parameters
        self.m = graph.size(weight='none')
        self.n = graph.number_of_nodes()
        self.degrees = dict(graph.degree(weight='none'))
        self.adj = nx.to_numpy_array(graph, weight='none')
        self.nodes = list(graph.nodes())
        self.lower_bound = -1
        self.k = None
        self.best_partition = None
        self.recursive_calls = 0
        self.branches_pruned = 0
        self.pruning_depths = []
        self.upper_bound_history = []  # List of (iteration, upper_bound) tuples
        self.lower_bound_history = []  # List of (iteration, lower_bound) tuples
        self.k_leiden = None
        self.heuristic_partition = None  # Best heuristic partition for guiding localfs search
        self.explored_leaves = 0  # Cumulative count of explored/pruned leaf nodes
        self.subtree_sizes = None  # Precomputed subtree sizes for accurate progress tracking
        self.total_leaves = None  # Total search tree size


    def _compute_subtree_sizes(self):
        """
        Precompute subtree sizes for accurate progress tracking.

        f(bnb_c, r) = number of leaves in subtree with bnb_c clusters and r remaining nodes
        Recurrence: f(bnb_c, r) = bnb_c × f(bnb_c, r-1) + [bnb_c < k] × f(bnb_c+1, r-1)

        This gives the exact search tree size accounting for the constraint that
        we can only create new clusters when we have fewer than k.
        """
        self.subtree_sizes = {}

        # Base case: 0 remaining nodes = 1 leaf (reached end of path)
        for c in range(self.k + 1):
            self.subtree_sizes[(c, 0)] = 1

        # Fill table bottom-up (increasing r)
        for r in range(1, self.n + 1):
            for c in range(self.k, -1, -1):
                if c == 0:
                    # Must create first cluster (only 1 option)
                    self.subtree_sizes[(0, r)] = self.subtree_sizes.get((1, r - 1), 0)
                elif c < self.k:
                    # Can join any of bnb_c existing clusters OR create new cluster
                    join_existing = c * self.subtree_sizes.get((c, r - 1), 0)
                    create_new = self.subtree_sizes.get((c + 1, r - 1), 0)
                    self.subtree_sizes[(c, r)] = join_existing + create_new
                else:  # bnb_c == k
                    # Can only join existing clusters (k options)
                    self.subtree_sizes[(c, r)] = c * self.subtree_sizes.get((c, r - 1), 0)

        # Total tree size starting from 0 clusters and n nodes
        self.total_leaves = self.subtree_sizes.get((0, self.n), 1)


    def _get_subtree_size(self, num_clusters, num_remaining):
        """Get the number of leaves in a subtree with given clusters and remaining nodes."""
        if self.subtree_sizes is None:
            # Fallback to simple estimate if not precomputed
            return self.k ** num_remaining
        return self.subtree_sizes.get((num_clusters, num_remaining), 1)


    def modularity(self, partition):
        """Calculate modularity using community-based approach."""
        modularity_sum = 0

        for community in partition:
            edges_in_community = 0
            for node1 in community:
                for node2 in community:
                    if node1 < node2 and self.graph.has_edge(node1, node2):
                        edges_in_community += 1

            sum_degrees = sum(self.degrees[node] for node in community)

            contribution = edges_in_community / self.m - (sum_degrees / (2 * self.m)) ** 2
            modularity_sum += contribution

        return modularity_sum


    def _local_modularity_change(self, subset, new_node):
        additional_edges = sum(self.adj[new_node][j] for j in subset)
        subset_degree_sum = sum(self.degrees[j] for j in subset)
        new_node_degree = self.degrees[new_node]

        edge_change = additional_edges / self.m
        degree_change = - 2 * subset_degree_sum * int(new_node_degree) / (2 * self.m) ** 2
        return edge_change + degree_change


    def _size_constrained_upper_bound(self, current_mod, clusters, remaining):
        """
        Calculate upper bound respecting min/max community size constraints.

        Key improvements over unconstrained bound:
        1. Edge contribution: Accounts for capacity limits (max_size) when assigning nodes
        2. Degree distribution: Respects size constraints when optimizing degree balance

        Args:
            current_mod: Current modularity
            clusters: Current partial clustering
            remaining: Nodes not yet assigned

        Returns:
            Upper bound on achievable modularity
        """
        if not remaining:
            return current_mod

        cluster_degree_sums = [sum(self.degrees[node] for node in cluster) for cluster in clusters]

        # Count edges between remaining nodes (unchanged)
        edges_between_remaining = 0
        for i, node1 in enumerate(remaining):
            for node2 in remaining[i+1:]:
                if self.adj[node1][node2]:
                    edges_between_remaining += 1

        # Capacity-aware edge contribution from remaining nodes to existing clusters
        max_cluster_edges_to_remaining = 0

        # Calculate available capacity in each cluster
        cluster_capacities = []
        for i, cluster in enumerate(clusters):
            if self.max_community_size is not None:
                capacity = self.max_community_size - len(cluster)
            else:
                capacity = len(remaining)  # Unlimited
            cluster_capacities.append(max(0, capacity))

        # For each remaining node, find best reachable edges considering capacity
        for remaining_node in remaining:
            # Calculate edges to each cluster
            edges_to_clusters = []
            for i, cluster in enumerate(clusters):
                if cluster_capacities[i] > 0:
                    edges = sum(self.adj[remaining_node][c_node] for c_node in cluster)
                    edges_to_clusters.append((edges, i))

            if edges_to_clusters:
                # Best cluster this node can join
                best_edges, best_cluster_idx = max(edges_to_clusters)
                max_cluster_edges_to_remaining += best_edges
                # Decrement capacity (optimistic: assume we use it)
                cluster_capacities[best_cluster_idx] -= 1

        optimistic_edge_contribution = (edges_between_remaining + max_cluster_edges_to_remaining) / self.m

        # Size-constrained degree distribution optimization
        remaining_degrees = [self.degrees[node] for node in remaining]
        remaining_degrees.sort(reverse=True)  # Greedy assignment works better sorted

        # Calculate final cluster sizes respecting constraints
        current_sizes = [len(c) for c in clusters]
        final_sizes = list(current_sizes) + [0] * (self.k - len(clusters))

        # Calculate deficits: how many more nodes each cluster needs to reach min_size
        deficits = []
        if self.min_community_size is not None:
            for i in range(self.k):
                deficit = max(0, self.min_community_size - final_sizes[i])
                deficits.append(deficit)
            total_deficit = sum(deficits)
        else:
            deficits = [0] * self.k
            total_deficit = 0

        # Phase 1: Assign nodes to satisfy minimum size constraints
        # Use LOWEST degree nodes first (pessimistic but valid upper bound)
        if total_deficit > 0 and total_deficit <= len(remaining_degrees):
            # Take the lowest-degree nodes to fill deficits
            deficit_degrees = remaining_degrees[-total_deficit:]  # Lowest degrees
            remaining_degrees = remaining_degrees[:-total_deficit]  # Keep high-degree for phase 2

            # Distribute deficit nodes to undersized clusters
            deficit_idx = 0
            for i in range(self.k):
                for _ in range(deficits[i]):
                    if deficit_idx < len(deficit_degrees):
                        node_degree = deficit_degrees[deficit_idx]
                        if i < len(cluster_degree_sums):
                            cluster_degree_sums[i] += node_degree
                        else:
                            while len(cluster_degree_sums) <= i:
                                cluster_degree_sums.append(0)
                            cluster_degree_sums[i] += node_degree
                        final_sizes[i] += 1
                        deficit_idx += 1

        # Phase 2: Distribute remaining nodes to minimize degree variance
        for node_degree in remaining_degrees:
            # Find cluster that minimizes degree sum of squares after assignment
            best_cluster_idx = None
            best_penalty_increase = float('inf')

            for i in range(self.k):
                # Check if we can add to this cluster
                can_add = True
                if self.max_community_size is not None and final_sizes[i] >= self.max_community_size:
                    can_add = False

                if can_add:
                    # Calculate penalty increase
                    old_deg = cluster_degree_sums[i] if i < len(cluster_degree_sums) else 0
                    new_deg = old_deg + node_degree
                    penalty_increase = (new_deg / (2 * self.m)) ** 2 - (old_deg / (2 * self.m)) ** 2

                    if penalty_increase < best_penalty_increase:
                        best_penalty_increase = penalty_increase
                        best_cluster_idx = i

            # Assign to best cluster
            if best_cluster_idx is not None:
                if best_cluster_idx < len(cluster_degree_sums):
                    cluster_degree_sums[best_cluster_idx] += node_degree
                else:
                    # Extend to handle future clusters
                    while len(cluster_degree_sums) <= best_cluster_idx:
                        cluster_degree_sums.append(0)
                    cluster_degree_sums[best_cluster_idx] += node_degree
                final_sizes[best_cluster_idx] += 1

        # Ensure we have exactly k clusters for penalty calculation
        while len(cluster_degree_sums) < self.k:
            cluster_degree_sums.append(0)

        # Calculate current and optimistic penalties
        current_cluster_degs = [sum(self.degrees[node] for node in cluster) for cluster in clusters]
        current_degree_penalty = sum((deg / (2 * self.m)) ** 2 for deg in current_cluster_degs) + \
                                sum((self.degrees[node] / (2 * self.m)) ** 2 for node in remaining)

        optimistic_degree_penalty = sum((deg / (2 * self.m)) ** 2 for deg in cluster_degree_sums[:self.k])

        degree_improvement = current_degree_penalty - optimistic_degree_penalty

        # Upper bound
        upper_bound = current_mod + optimistic_edge_contribution + degree_improvement

        return min(upper_bound, 1 - 1 / self.k)


    def _updated_upper_bound(self, current_mod, clusters, remaining):
        if not remaining:
            return current_mod

        cluster_degree_sums = [sum(self.degrees[node] for node in cluster) for cluster in clusters]

        # Count edges between remaining nodes
        edges_between_remaining = 0
        for i, node1 in enumerate(remaining):
            for node2 in remaining[i+1:]:
                if self.adj[node1][node2]:
                    edges_between_remaining += 1

        # Count edges from existing cluster nodes to remaining nodes
        edges_cluster_to_remaining = 0
        max_cluster_edges_to_remaining = 0
        for remaining_node in remaining:
            max_cluster_edges_to_remaining_node = 0
            for cluster in clusters:
                current_cluster_edges_to_remaining_node = 0
                for cluster_node in cluster:
                    if self.adj[cluster_node][remaining_node]:
                        edges_cluster_to_remaining += 1
                        current_cluster_edges_to_remaining_node += 1
                if current_cluster_edges_to_remaining_node > max_cluster_edges_to_remaining_node:
                    max_cluster_edges_to_remaining_node = current_cluster_edges_to_remaining_node
            max_cluster_edges_to_remaining += max_cluster_edges_to_remaining_node

        # Calculate remaining degree sum from edge counts
        remaining_degree_sum = 2 * edges_between_remaining + edges_cluster_to_remaining

        # Optimistic edge contribution
        optimistic_edge_contribution = (edges_between_remaining + max_cluster_edges_to_remaining) / self.m

        # Build optimistic final degree distribution with balanced approach
        # Ideal: all k clusters have equal degrees = 2m/k to minimize Σ(d_i)²
        total_degree_sum = 2 * self.m
        ideal_degree_per_cluster = total_degree_sum / self.k

        # Start with existing cluster degrees + empty clusters
        final_degrees = list(cluster_degree_sums) + [0] * (self.k - len(cluster_degree_sums))

        # Identify clusters already at or above ideal (these get no additional degrees)
        locked_indices = set()
        for i in range(len(cluster_degree_sums)):
            if cluster_degree_sums[i] >= ideal_degree_per_cluster:
                locked_indices.add(i)

        locked_degree_sum = sum(cluster_degree_sums[i] for i in locked_indices)

        # Distribute degrees among non-locked clusters to achieve balance
        non_locked_count = self.k - len(locked_indices)

        if non_locked_count > 0:
            # Budget available for non-locked clusters = total - locked
            available_for_non_locked = total_degree_sum - locked_degree_sum
            target_per_non_locked = available_for_non_locked / non_locked_count

            # Set non-locked clusters to target value
            for i in range(self.k):
                if i not in locked_indices:
                    final_degrees[i] = target_per_non_locked
            # Locked clusters keep their current values (already set above)
        else:
            # All clusters exceed ideal - distribute remaining equally to all
            per_cluster = remaining_degree_sum / self.k
            for i in range(self.k):
                final_degrees[i] += per_cluster

        # Calculate degree penalties
        # Current penalty from existing clusters AND remaining singletons
        current_degree_penalty = sum((deg_sum / (2 * self.m)) ** 2 for deg_sum in cluster_degree_sums) + \
                                sum((int(self.degrees[node]) / (2 * self.m)) ** 2 for node in remaining)

        # Optimistic penalty from final balanced distribution
        optimistic_degree_penalty = sum((deg / (2 * self.m)) ** 2 for deg in final_degrees)

        # Improvement from better degree distribution (current -> optimistic)
        degree_improvement = current_degree_penalty - optimistic_degree_penalty

        # Upper bound
        upper_bound = current_mod + optimistic_edge_contribution + degree_improvement

        return min(upper_bound, 1 - 1 / self.k)


    def _updated_upper_bound_incremental(self, current_mod, state):
        if not state.remaining_set:
            return current_mod

        cluster_degree_sums = state.cluster_degree_sums
        edges_between_remaining = state.edges_between_remaining

        # Calculate max_cluster_edges_to_remaining from pre-computed data
        # For each remaining node, take the max edges to any cluster
        max_cluster_edges_to_remaining = 0
        for node in state.remaining_set:
            node_edges = state.node_to_cluster_edges[node]
            if node_edges:
                max_cluster_edges_to_remaining += max(node_edges)

        optimistic_edge_contribution = (edges_between_remaining + max_cluster_edges_to_remaining) / self.m

        total_degree_sum = 2 * self.m
        ideal_degree_per_cluster = total_degree_sum / self.k

        # Start with existing cluster degrees + empty clusters
        final_degrees = list(cluster_degree_sums) + [0] * (self.k - len(cluster_degree_sums))

        # Identify clusters already at or above ideal
        locked_indices = set()
        for i in range(len(cluster_degree_sums)):
            if cluster_degree_sums[i] >= ideal_degree_per_cluster:
                locked_indices.add(i)

        locked_degree_sum = sum(cluster_degree_sums[i] for i in locked_indices)

        # Distribute degrees among non-locked clusters
        non_locked_count = self.k - len(locked_indices)

        if non_locked_count > 0:
            available_for_non_locked = total_degree_sum - locked_degree_sum
            target_per_non_locked = available_for_non_locked / non_locked_count

            for i in range(self.k):
                if i not in locked_indices:
                    final_degrees[i] = target_per_non_locked
        else:
            # All clusters exceed ideal - need remaining_degree_sum
            # Calculate it from state
            edges_cluster_to_remaining = sum(
                sum(state.node_to_cluster_edges[node])
                for node in state.remaining_set
            )
            remaining_degree_sum = 2 * edges_between_remaining + edges_cluster_to_remaining
            per_cluster = remaining_degree_sum / self.k
            for i in range(self.k):
                final_degrees[i] += per_cluster

        # Calculate degree penalties using pre-computed remaining_degree_penalty
        current_degree_penalty = (
            sum((deg_sum / (2 * self.m)) ** 2 for deg_sum in cluster_degree_sums) +
            state.remaining_degree_penalty
        )

        optimistic_degree_penalty = sum((deg / (2 * self.m)) ** 2 for deg in final_degrees)

        degree_improvement = current_degree_penalty - optimistic_degree_penalty

        upper_bound = current_mod + optimistic_edge_contribution + degree_improvement

        return min(upper_bound, 1 - 1 / self.k)


    def _k_constrained_leiden_iteration(self, k, initial_partition, max_moves=None):
        """
        Run one iteration of k-constrained Leiden algorithm.

        The key constraint: only allow moves that maintain exactly k non-empty communities.
        A node can move to another community only if its source community has more than 1 node.

        Args:
            k: Target number of communities
            initial_partition: Starting partition (list of sets)
            max_moves: Maximum number of moves to try (None = unlimited)

        Returns:
            Optimized partition with exactly k communities
        """
        # Create membership array for fast lookup
        membership = {}
        for comm_idx, community in enumerate(initial_partition):
            for node in community:
                membership[node] = comm_idx

        # Create mutable partition
        partition = [set(comm) for comm in initial_partition]

        # Precompute community degree sums and internal edges for efficiency
        comm_degree_sums = [sum(self.degrees[node] for node in comm) for comm in partition]
        comm_internal_edges = []
        for comm in partition:
            internal = 0
            for node in comm:
                for other in comm:
                    if node < other and self.adj[node][other]:
                        internal += 1
            comm_internal_edges.append(internal)

        # Local move phase with k-constraint
        improved = True
        total_moves = 0

        while improved:
            improved = False
            nodes_to_check = list(self.nodes)
            random.shuffle(nodes_to_check)

            for node in nodes_to_check:
                if max_moves is not None and total_moves >= max_moves:
                    break

                current_comm = membership[node]

                # Can only move if current community has more than 1 node (to maintain k communities)
                if len(partition[current_comm]) <= 1:
                    continue

                # Calculate current contribution of this node
                node_degree = self.degrees[node]
                edges_to_current = sum(self.adj[node][other] for other in partition[current_comm] if other != node)

                # Current modularity contribution from this node in its community
                # (simplified: we only need the delta, not absolute values)

                best_delta = 0
                best_target = -1

                # Try moving to each other community
                for target_comm in range(k):
                    if target_comm == current_comm:
                        continue

                    # Calculate edges to target community
                    edges_to_target = sum(self.adj[node][other] for other in partition[target_comm])

                    # Calculate modularity delta for this move
                    # Delta = (edges_to_target - edges_to_current) / m
                    #         - node_degree * (comm_degree_sums[target] - comm_degree_sums[current] + node_degree) / (2m)^2

                    target_degree_sum = comm_degree_sums[target_comm]
                    current_degree_sum = comm_degree_sums[current_comm] - node_degree

                    edge_delta = (edges_to_target - edges_to_current) / self.m

                    # Degree penalty change
                    # Old penalty: (current_sum/2m)^2 + (target_sum/2m)^2
                    # New penalty: ((current_sum - node_deg)/2m)^2 + ((target_sum + node_deg)/2m)^2
                    two_m = 2 * self.m
                    old_penalty = (comm_degree_sums[current_comm] / two_m) ** 2 + (target_degree_sum / two_m) ** 2
                    new_penalty = (current_degree_sum / two_m) ** 2 + ((target_degree_sum + node_degree) / two_m) ** 2
                    penalty_delta = old_penalty - new_penalty

                    delta = edge_delta + penalty_delta

                    if delta > best_delta:
                        best_delta = delta
                        best_target = target_comm

                # Make the move if it improves modularity
                if best_target >= 0 and best_delta > 1e-10:
                    # Update partition
                    partition[current_comm].remove(node)
                    partition[best_target].add(node)
                    membership[node] = best_target

                    # Update degree sums
                    comm_degree_sums[current_comm] -= node_degree
                    comm_degree_sums[best_target] += node_degree

                    improved = True
                    total_moves += 1

            if max_moves is not None and total_moves >= max_moves:
                break

        return partition

    def _initialize_k_partition_random(self, k, seed=None):
        """
        Initialize a k-partition with random assignment.

        Args:
            k: Number of communities
            seed: Random seed for reproducibility

        Returns:
            Initial partition as list of k sets
        """
        if seed is not None:
            random.seed(seed)

        nodes = list(self.nodes)
        random.shuffle(nodes)

        # Ensure each community gets at least one node
        partition = [{nodes[i]} for i in range(min(k, len(nodes)))]

        # Assign remaining nodes randomly
        for node in nodes[k:]:
            comm_idx = random.randint(0, k - 1)
            partition[comm_idx].add(node)

        return partition

    def _initialize_with_leiden(self, k, highest_mod, num_iterations):
        """
        Initialize the best partition using k-constrained Leiden algorithm.

        This is a modified Leiden that enforces exactly k communities by:
        1. Starting with a random k-partition initialization
        2. Only allowing moves that maintain exactly k non-empty communities
        3. Running multiple iterations with different initializations

        Args:
            k: Target number of communities
            highest_mod: Current highest modularity to compare against
            num_iterations: Number of independent iterations to run

        Returns:
            The best modularity found across all iterations
        """
        if num_iterations is None or num_iterations <= 0:
            return highest_mod

        try:
            for iteration in range(num_iterations):
                initial_partition = self._initialize_k_partition_random(k, seed=iteration)

                # Run k-constrained Leiden optimization
                optimized_partition = self._k_constrained_leiden_iteration(k, initial_partition)

                # Evaluate the result
                partition_mod = self.modularity(optimized_partition)

                if partition_mod > highest_mod:
                    highest_mod = partition_mod
                    # Deep copy the partition
                    self.best_partition = [set(comm) for comm in optimized_partition]
                    self.heuristic_partition = [set(comm) for comm in optimized_partition]

                    if self.params.enable_debugging:
                        print(f'k-Leiden iteration {iteration + 1}/{num_iterations}: modularity = {partition_mod:.6f}')

            if self.params.enable_debugging and num_iterations > 1:
                print(f'Best k-Leiden modularity across {num_iterations} iterations: {highest_mod:.6f}')

            return highest_mod

        except Exception as e:
            if self.params.enable_debugging:
                print(f'k-constrained Leiden failed: {e}')
            return highest_mod


    def _get_heuristic_vertex_assignment(self):
        """
        Create a mapping from vertex to heuristic community index.

        Returns:
            dict: {vertex: community_index} mapping based on heuristic_partition,
                  or empty dict if no heuristic partition is available.
        """
        if self.heuristic_partition is None:
            return {}

        vertex_to_community = {}
        for community_idx, community in enumerate(self.heuristic_partition):
            for vertex in community:
                vertex_to_community[vertex] = community_idx
        return vertex_to_community


    def _apply_vertex_ordering(self):
        """Apply vertex ordering based on the selected technique."""
        ordering = self.params.vertex_ordering

        if ordering == 'degree_desc':
            # Order by degree in descending order
            self.nodes = sorted(self.nodes, key=lambda node: self.degrees[node], reverse=True)

        elif ordering == 'random':
            # Shuffle randomly
            random.shuffle(self.nodes)

        elif ordering == 'binomial':
            # Place high-degree nodes preferentially in the middle using binomial distribution
            sorted_nodes = sorted(self.nodes, key=lambda node: self.degrees[node], reverse=True)
            n = len(sorted_nodes)
            p = 0.5
            position_weights = [math.comb(n-1, i) * (p ** i) * ((1-p) ** (n-1-i)) for i in range(n)]
            self.nodes = [None] * n
            available_positions = list(range(n))
            for node in sorted_nodes:
                current_weights = [position_weights[pos] for pos in available_positions]
                total_weight = sum(current_weights)
                normalized_weights = [w / total_weight for w in current_weights]

                chosen_idx = random.choices(range(len(available_positions)),
                                           weights=normalized_weights,
                                           k=1)[0]
                chosen_position = available_positions.pop(chosen_idx)
                self.nodes[chosen_position] = node

        elif ordering == 'none':
            # No ordering changes - keep original order
            pass

        else:
            raise ValueError(f"Unknown vertex ordering: {ordering}. "
                           f"Valid options: 'degree_desc', 'random', 'binomial', 'none'")


    def _validate_inputs(self, k):
        """Validate inputs and handle edge cases. Returns (is_valid, result) tuple."""
        if self.graph.number_of_nodes() == 0:
            if k == 0:
                return False, ([], 0.0)
            else:
                return False, (None, float('-inf'))  # Can't create k>0 communities with 0 nodes

        if k <= 0:
            return False, (None, float('-inf'))

        if k > self.graph.number_of_nodes():
            return False, (None, float('-inf'))

        # Validate size constraints
        if self.min_community_size is not None:
            if self.min_community_size < 1:
                return False, (None, float('-inf'))
            if k * self.min_community_size > self.n:
                # Not enough nodes to satisfy minimum size for k communities
                return False, (None, float('-inf'))

        if self.max_community_size is not None:
            if self.max_community_size < 1:
                return False, (None, float('-inf'))
            if k * self.max_community_size < self.n:
                # Too many nodes to fit in k communities with max size
                return False, (None, float('-inf'))

        if self.min_community_size is not None and self.max_community_size is not None:
            if self.min_community_size > self.max_community_size:
                return False, (None, float('-inf'))

        if self.m == 0:
            singleton_partition = [{node} for node in self.nodes]
            singleton_modularity = self.modularity(singleton_partition)
            return False, (singleton_partition, singleton_modularity)

        return True, None


    def _generate_branches(self, clustered, remaining, current_mod):
        """
        Generate all possible branches for the next node assignment (lazily).
        Respects min/max community size constraints if specified.

        Args:
            clustered: Current partial clustering (list of communities)
            remaining: Nodes not yet assigned
            current_mod: Current modularity value

        Yields:
            Tuples of (new_clustered, new_remaining, new_mod)
        """
        if not remaining:
            return

        first = remaining[0]
        remaining_rest = remaining[1:]

        # Option 1: Add to each existing community
        for i in range(len(clustered)):
            # Check max_community_size constraint
            if self.max_community_size is not None and len(clustered[i]) >= self.max_community_size:
                continue  # Cannot add to this cluster (already at max size)

            # Check min_community_size constraint: would this addition leave insufficient nodes
            # for other undersized clusters?
            if self.min_community_size is not None:
                # After adding 'first' to cluster i, calculate remaining deficit
                new_cluster_size = len(clustered[i]) + 1
                deficit_this_cluster = max(0, self.min_community_size - new_cluster_size)
                # Deficit in other clusters
                deficit_other_clusters = sum(max(0, self.min_community_size - len(clustered[j]))
                                             for j in range(len(clustered)) if j != i)
                total_deficit = deficit_this_cluster + deficit_other_clusters

                # Nodes remaining after this assignment
                nodes_remaining_after = len(remaining_rest)

                if total_deficit > nodes_remaining_after:
                    # Not enough remaining nodes to fill all undersized clusters
                    continue

            delta_mod = self._local_modularity_change(clustered[i], first)
            new_mod = current_mod + delta_mod
            new_subset = clustered[i] | {first}
            new_clustered = clustered[:i] + [new_subset] + clustered[i + 1:]
            yield (new_clustered, remaining_rest, new_mod)

        # Option 2: Create new singleton community (if space available)
        if len(clustered) < self.k:
            # Check if we have enough remaining nodes to create a new singleton
            # and still satisfy min_community_size for all clusters
            if self.min_community_size is not None:
                # Calculate deficit: nodes needed to bring existing clusters to min size
                deficit = sum(max(0, self.min_community_size - len(c)) for c in clustered)
                # If we create a new singleton, we need:
                # - 'deficit' nodes for existing undersized clusters
                # - 'min_community_size - 1' more nodes for the new singleton (already has 1)
                # Check if remaining nodes (after taking current node) can satisfy this
                if deficit + self.min_community_size > len(remaining):
                    # Not enough nodes - creating singleton would be infeasible
                    # Only fill existing clusters from now on
                    return

            new_mod = current_mod
            new_clustered = clustered + [{first}]
            yield (new_clustered, remaining_rest, new_mod)


    def _dfs(self, clustered, remaining, current_mod, upper_bound):
        """Depth first search."""
        self.recursive_calls += 1
        if self.params.enable_debugging and self.recursive_calls % (self.params.max_recursive_calls / 20) == 0:
            print('Recursive calls:', self.recursive_calls, clustered, remaining, current_mod, self.lower_bound, upper_bound)

        if self.params.track_bound_history:
            self.upper_bound_history.append((self.recursive_calls, upper_bound))
            self.lower_bound_history.append((self.recursive_calls, self.lower_bound))

        if self.recursive_calls > self.params.max_recursive_calls:
            if self.params.enable_debugging:
                print('terminated, parameters:', clustered, remaining, current_mod, self.lower_bound, upper_bound)
            return

        if not remaining:
            if len(clustered) == self.k or self.params.search_up_to_k_communities and len(clustered) <= self.k:
                if current_mod >= self.lower_bound:
                    self.lower_bound = current_mod
                    self.best_partition = clustered
                    if self.params.enable_debugging:
                        print('Best partition:', self.best_partition, current_mod, self.lower_bound, upper_bound)
            return

        branches = self._generate_branches(clustered, remaining, current_mod)

        for new_clustered, new_remaining, new_mod in branches:
            if self.params.enable_pruning:
                # Use size-constrained bound if constraints are present
                if self.min_community_size is not None or self.max_community_size is not None:
                    new_upper_bound = self._size_constrained_upper_bound(new_mod, new_clustered, new_remaining)
                else:
                    new_upper_bound = self._updated_upper_bound(new_mod, new_clustered, new_remaining)

                if new_upper_bound < self.lower_bound:
                    self.branches_pruned += 1
                    self.pruning_depths.append(len(new_clustered))
                    continue
            else:
                new_upper_bound = upper_bound

            self._dfs(new_clustered, new_remaining, new_mod, new_upper_bound)


    def _dfs_incremental(self, clustered, remaining, current_mod, upper_bound, state):
        """
        Depth first search with incremental upper bound calculation.

        Uses IncrementalBoundState to avoid O(n²) recalculations at each node.
        """
        self.recursive_calls += 1
        if self.params.enable_debugging and self.recursive_calls % (self.params.max_recursive_calls / 500) == 0:
            explored_pct = 100.0 * self.explored_leaves / self.total_leaves
            print('Recursive calls:', self.recursive_calls, '|', clustered, '|', remaining, '|', current_mod, '|',
                  self.lower_bound, '|', upper_bound, f"| Explored {explored_pct:.6f}% of search tree")

        if self.params.track_bound_history:
            self.upper_bound_history.append((self.recursive_calls, upper_bound))
            self.lower_bound_history.append((self.recursive_calls, self.lower_bound))

        if self.recursive_calls > self.params.max_recursive_calls:
            explored_pct = 100.0 * self.explored_leaves / self.total_leaves
            if self.params.enable_debugging:
                print(f"Terminated due to max_recursive_calls reached. Explored {explored_pct:.6f}% of search tree, best solution: {self.lower_bound:.6f}")
            return

        if not remaining:
            self.explored_leaves += 1
            if len(clustered) == self.k or self.params.search_up_to_k_communities and len(clustered) <= self.k:
                if current_mod >= self.lower_bound - 0.000000001:
                    self.lower_bound = current_mod
                    self.best_partition = clustered
                    if self.params.enable_debugging:
                        print('Best partition:', self.best_partition, current_mod, self.lower_bound, upper_bound)
            return

        vertex = remaining[0]
        branches = self._generate_branches(clustered, remaining, current_mod)

        for new_clustered, new_remaining, new_mod in branches:
            if self.params.enable_pruning:
                new_state = state.copy()

                if len(new_clustered) > len(clustered):
                    new_state.update_for_new_singleton(vertex)
                else:
                    cluster_idx = None
                    for i in range(len(clustered)):
                        if vertex in new_clustered[i]:
                            cluster_idx = i
                            break
                    new_state.update_for_join_cluster(vertex, cluster_idx)

                new_upper_bound = self._updated_upper_bound_incremental(new_mod, new_state)

                if new_upper_bound < self.lower_bound - 0.000000001:
                    self.branches_pruned += 1
                    self.pruning_depths.append(len(new_clustered))
                    pruned_leaves = self._get_subtree_size(len(new_clustered), len(new_remaining))
                    self.explored_leaves += pruned_leaves
                    continue
            else:
                new_upper_bound = upper_bound
                new_state = state

            self._dfs_incremental(new_clustered, new_remaining, new_mod, new_upper_bound, new_state)


    def _bestfs(self, clustered, remaining, current_mod, upper_bound):
        """Best First Search: explore assignments in order of modularity contribution."""
        self.recursive_calls += 1
        if self.params.enable_debugging and self.recursive_calls % (self.params.max_recursive_calls / 20) == 0:
            print('Recursive calls:', self.recursive_calls, clustered, remaining, current_mod, self.lower_bound, upper_bound)

        if self.params.track_bound_history:
            self.upper_bound_history.append((self.recursive_calls, upper_bound))
            self.lower_bound_history.append((self.recursive_calls, self.lower_bound))

        if self.recursive_calls > self.params.max_recursive_calls:
            if self.params.enable_debugging:
                print('terminated, parameters:', clustered, remaining, current_mod, self.lower_bound, upper_bound)
            return

        if not remaining:
            if len(clustered) == self.k or self.params.search_up_to_k_communities and len(clustered) <= self.k:
                if current_mod >= self.lower_bound:
                    self.lower_bound = current_mod
                    self.best_partition = clustered
                    if self.params.enable_debugging:
                        print('Best partition:', self.best_partition, current_mod, self.lower_bound, upper_bound)
            return

        branches = list(self._generate_branches(clustered, remaining, current_mod))

        # Sort by modularity gain (descending = best first)
        branches_with_delta = [(new_clustered, new_remaining, new_mod, new_mod - current_mod)
                                for new_clustered, new_remaining, new_mod in branches]
        branches_with_delta.sort(key=lambda x: x[3], reverse=True)

        for new_clustered, new_remaining, new_mod, _ in branches_with_delta:
            if self.params.enable_pruning:
                if self.min_community_size is not None or self.max_community_size is not None:
                    new_upper_bound = self._size_constrained_upper_bound(new_mod, new_clustered, new_remaining)
                else:
                    new_upper_bound = self._updated_upper_bound(new_mod, new_clustered, new_remaining)

                if new_upper_bound < self.lower_bound:
                    self.branches_pruned += 1
                    self.pruning_depths.append(len(new_clustered))
                    continue
            else:
                new_upper_bound = upper_bound

            self._bestfs(new_clustered, new_remaining, new_mod, new_upper_bound)


    def _bestfs_incremental(self, clustered, remaining, current_mod, upper_bound, state):
        """
        Best First Search with incremental upper bound calculation.

        Uses IncrementalBoundState to avoid O(n²) recalculations at each node.
        The state is updated incrementally when vertices move from remaining to clusters.

        Args:
            clustered: Current partial clustering (list of communities)
            remaining: Nodes not yet assigned
            current_mod: Current modularity value
            upper_bound: Current upper bound
            state: IncrementalBoundState with pre-computed values
        """
        self.recursive_calls += 1
        if self.params.enable_debugging and self.recursive_calls % (self.params.max_recursive_calls / 500) == 0:
            explored_pct = 100.0 * self.explored_leaves / self.total_leaves
            print('Recursive calls:', self.recursive_calls, '|', clustered, '|', remaining, '|', current_mod, '|',
                  self.lower_bound, '|', upper_bound, f"| Explored {explored_pct:.6f}% of search tree")
            self.lower_bound_history.append((self.recursive_calls, self.lower_bound))

        if self.recursive_calls > self.params.max_recursive_calls:
            explored_pct = 100.0 * self.explored_leaves / self.total_leaves
            if self.params.enable_debugging:
                print(f"Terminated due to max_recursive_calls reached. Explored {explored_pct:.6f}% of search tree, best solution: {self.lower_bound:.6f}")
            return

        if not remaining:
            # Reached a leaf - count it
            self.explored_leaves += 1
            if len(clustered) == self.k or self.params.search_up_to_k_communities and len(clustered) <= self.k:
                if current_mod >= self.lower_bound - 0.000000001:
                    self.lower_bound = current_mod
                    self.best_partition = clustered
                    if self.params.enable_debugging:
                        print('Best partition:', self.best_partition, current_mod, self.lower_bound, upper_bound)
            return

        vertex = remaining[0]
        branches = list(self._generate_branches(clustered, remaining, current_mod))

        # Sort by modularity gain (descending = best first)
        branches_with_delta = [(new_clustered, new_remaining, new_mod, new_mod - current_mod)
                               for new_clustered, new_remaining, new_mod in branches]
        branches_with_delta.sort(key=lambda x: x[3], reverse=True)

        for new_clustered, new_remaining, new_mod, _ in branches_with_delta:
            if self.params.enable_pruning:
                new_state = state.copy()

                if len(new_clustered) > len(clustered):
                    new_state.update_for_new_singleton(vertex)
                else:
                    cluster_idx = None
                    for i in range(len(clustered)):
                        if vertex in new_clustered[i]:
                            cluster_idx = i
                            break
                    new_state.update_for_join_cluster(vertex, cluster_idx)

                new_upper_bound = self._updated_upper_bound_incremental(new_mod, new_state)

                if new_upper_bound < self.lower_bound - 0.000000001:
                    self.branches_pruned += 1
                    self.pruning_depths.append(len(new_clustered))
                    pruned_leaves = self._get_subtree_size(len(new_clustered), len(new_remaining))
                    self.explored_leaves += pruned_leaves
                    continue
            else:
                new_upper_bound = upper_bound
                new_state = state

            self._bestfs_incremental(new_clustered, new_remaining, new_mod, new_upper_bound, new_state)


    def _localfs(self, clustered, remaining, current_mod, upper_bound, heuristic_assignment, community_mapping):
        """
        Local-First Search: dive deep following heuristic solution first, then explore alternatives.

        This strategy prioritizes branches that match the heuristic (Louvain) solution,
        allowing the algorithm to quickly reach a good solution and establish a strong
        lower bound for pruning. After the heuristic path, it explores alternatives
        sorted by modularity gain.

        Args:
            clustered: Current partial clustering (list of communities)
            remaining: Nodes not yet assigned
            current_mod: Current modularity value
            upper_bound: Current upper bound
            heuristic_assignment: Dict mapping vertex -> heuristic community index
            community_mapping: Dict mapping heuristic community index -> current clustered index
                               (tracks how heuristic communities map to our growing partition)
        """
        self.recursive_calls += 1
        if self.params.enable_debugging and self.recursive_calls % (self.params.max_recursive_calls / 20) == 0:
            print('Recursive calls:', self.recursive_calls, clustered, remaining, current_mod, self.lower_bound, upper_bound)

        if self.params.track_bound_history:
            self.upper_bound_history.append((self.recursive_calls, upper_bound))
            self.lower_bound_history.append((self.recursive_calls, self.lower_bound))

        if self.recursive_calls > self.params.max_recursive_calls:
            if self.params.enable_debugging:
                print('terminated, parameters:', clustered, remaining, current_mod, self.lower_bound, upper_bound)
            return

        if not remaining:
            if len(clustered) == self.k or self.params.search_up_to_k_communities and len(clustered) <= self.k:
                if current_mod >= self.lower_bound:
                    self.lower_bound = current_mod
                    self.best_partition = clustered
                    if self.params.enable_debugging:
                        print('Best partition:', self.best_partition, current_mod, self.lower_bound, upper_bound)
            return

        branches = list(self._generate_branches(clustered, remaining, current_mod))

        first = remaining[0]
        heuristic_community_idx = heuristic_assignment.get(first)

        # Determine which branch matches the heuristic assignment (if any)
        heuristic_branch_idx = None
        if heuristic_community_idx is not None:
            mapped_cluster_idx = community_mapping.get(heuristic_community_idx)
            if mapped_cluster_idx is not None and mapped_cluster_idx < len(clustered):
                # Heuristic says: add to existing community at mapped_cluster_idx
                for idx, (new_clustered, new_remaining, new_mod) in enumerate(branches):
                    # Check if this branch adds 'first' to the mapped cluster
                    if len(new_clustered) == len(clustered):  # Same number of clusters = added to existing
                        if first in new_clustered[mapped_cluster_idx]:
                            heuristic_branch_idx = idx
                            break
            elif mapped_cluster_idx is None:
                # This heuristic community hasn't been started yet -> prefer singleton
                for idx, (new_clustered, new_remaining, new_mod) in enumerate(branches):
                    if len(new_clustered) > len(clustered):  # New singleton created
                        heuristic_branch_idx = idx
                        break

        # Sort by modularity gain (descending = best first)
        branches_with_delta = [(new_clustered, new_remaining, new_mod, new_mod - current_mod, idx)
                               for idx, (new_clustered, new_remaining, new_mod) in enumerate(branches)]
        branches_with_delta.sort(key=lambda x: x[3], reverse=True)

        # Reorder: put heuristic branch first if it exists
        if heuristic_branch_idx is not None:
            heuristic_branch = None
            other_branches = []
            for branch in branches_with_delta:
                if branch[4] == heuristic_branch_idx:
                    heuristic_branch = branch
                else:
                    other_branches.append(branch)
            if heuristic_branch is not None:
                branches_with_delta = [heuristic_branch] + other_branches

        for new_clustered, new_remaining, new_mod, _, original_idx in branches_with_delta:
            if self.params.enable_pruning:
                # Use size-constrained bound if constraints are present
                if self.min_community_size is not None or self.max_community_size is not None:
                    new_upper_bound = self._size_constrained_upper_bound(new_mod, new_clustered, new_remaining)
                else:
                    new_upper_bound = self._updated_upper_bound(new_mod, new_clustered, new_remaining)

                if new_upper_bound < self.lower_bound:
                    self.branches_pruned += 1
                    self.pruning_depths.append(len(new_clustered))
                    continue
            else:
                new_upper_bound = upper_bound

            # Update community mapping for new singleton
            new_community_mapping = community_mapping.copy()
            if len(new_clustered) > len(clustered):
                # A new singleton was created - map heuristic community to this new cluster
                if heuristic_community_idx is not None and heuristic_community_idx not in new_community_mapping:
                    new_community_mapping[heuristic_community_idx] = len(clustered)

            self._localfs(new_clustered, new_remaining, new_mod, new_upper_bound,
                         heuristic_assignment, new_community_mapping)


    def _localfs_incremental(self, clustered, remaining, current_mod, upper_bound, heuristic_assignment, community_mapping, state):
        """
        Local-First Search with incremental upper bound calculation.

        Uses IncrementalBoundState to avoid O(n²) recalculations at each node.
        """
        self.recursive_calls += 1
        if self.params.enable_debugging and self.recursive_calls % (self.params.max_recursive_calls / 50) == 0:
            explored_pct = 100.0 * self.explored_leaves / self.total_leaves
            print('Recursive calls:', self.recursive_calls, '|', clustered, '|', remaining, '|', current_mod, '|',
                  self.lower_bound, '|', upper_bound, f"| Explored {explored_pct:.6f}% of search tree")
            self.lower_bound_history.append((self.recursive_calls, self.lower_bound))

        if self.recursive_calls > self.params.max_recursive_calls:
            explored_pct = 100.0 * self.explored_leaves / self.total_leaves
            if self.params.enable_debugging:
                print(f"Terminated. Explored {explored_pct:.6f}% of search tree, recursive calls: {self.recursive_calls}, best solution: {self.lower_bound:.6f}")
            return

        if not remaining:
            # Reached a leaf - count it
            self.explored_leaves += 1
            if len(clustered) == self.k or self.params.search_up_to_k_communities and len(clustered) <= self.k:
                if current_mod >= self.lower_bound:
                    self.lower_bound = current_mod
                    self.best_partition = clustered
                    if self.params.enable_debugging:
                        print('Best partition:', self.best_partition, current_mod, self.lower_bound, upper_bound)
            return

        branches = list(self._generate_branches(clustered, remaining, current_mod))

        first = remaining[0]
        heuristic_community_idx = heuristic_assignment.get(first)

        # Determine which branch matches the heuristic assignment (if any)
        heuristic_branch_idx = None
        if heuristic_community_idx is not None:
            mapped_cluster_idx = community_mapping.get(heuristic_community_idx)
            if mapped_cluster_idx is not None and mapped_cluster_idx < len(clustered):
                for idx, (new_clustered, new_remaining, new_mod) in enumerate(branches):
                    if len(new_clustered) == len(clustered):
                        if first in new_clustered[mapped_cluster_idx]:
                            heuristic_branch_idx = idx
                            break
            elif mapped_cluster_idx is None:
                for idx, (new_clustered, new_remaining, new_mod) in enumerate(branches):
                    if len(new_clustered) > len(clustered):
                        heuristic_branch_idx = idx
                        break

        # Sort by modularity gain (descending = best first)
        branches_with_delta = [(new_clustered, new_remaining, new_mod, new_mod - current_mod, idx)
                               for idx, (new_clustered, new_remaining, new_mod) in enumerate(branches)]
        branches_with_delta.sort(key=lambda x: x[3], reverse=True)

        # Reorder: put heuristic branch first if it exists
        if heuristic_branch_idx is not None:
            heuristic_branch = None
            other_branches = []
            for branch in branches_with_delta:
                if branch[4] == heuristic_branch_idx:
                    heuristic_branch = branch
                else:
                    other_branches.append(branch)
            if heuristic_branch is not None:
                branches_with_delta = [heuristic_branch] + other_branches

        for new_clustered, new_remaining, new_mod, _, original_idx in branches_with_delta:
            if self.params.enable_pruning:
                new_state = state.copy()

                if len(new_clustered) > len(clustered):
                    new_state.update_for_new_singleton(first)
                else:
                    cluster_idx = None
                    for i in range(len(clustered)):
                        if first in new_clustered[i]:
                            cluster_idx = i
                            break
                    new_state.update_for_join_cluster(first, cluster_idx)

                new_upper_bound = self._updated_upper_bound_incremental(new_mod, new_state)

                if new_upper_bound < self.lower_bound:
                    self.branches_pruned += 1
                    self.pruning_depths.append(len(new_clustered))
                    # Count all leaves in the pruned subtree (accurate calculation)
                    pruned_leaves = self._get_subtree_size(len(new_clustered), len(new_remaining))
                    self.explored_leaves += pruned_leaves
                    continue
            else:
                new_upper_bound = upper_bound
                new_state = state

            # Update community mapping for new singleton
            new_community_mapping = community_mapping.copy()
            if len(new_clustered) > len(clustered):
                if heuristic_community_idx is not None and heuristic_community_idx not in new_community_mapping:
                    new_community_mapping[heuristic_community_idx] = len(clustered)

            self._localfs_incremental(new_clustered, new_remaining, new_mod, new_upper_bound,
                                      heuristic_assignment, new_community_mapping, new_state)


    def _bfs(self, states_at_current_level):
        """Breadth-first search."""
        if not states_at_current_level:
            return

        if self.recursive_calls > self.params.max_recursive_calls:
            if self.params.enable_debugging:
                print('terminated due to max_recursive_calls reached')
            return

        next_level_states = []

        for state in states_at_current_level:
            clustered, remaining, current_mod, upper_bound = state

            self.recursive_calls += 1
            if self.params.enable_debugging and self.recursive_calls % (self.params.max_recursive_calls / 20) == 0:
                print('Recursive calls:', self.recursive_calls, 'Level size:', len(states_at_current_level),
                      'Current mod:', current_mod, 'Lower bound:', self.lower_bound, 'Upper bound:', upper_bound)

            # Prune this state if its upper bound is below current lower bound
            if self.params.enable_pruning and upper_bound < self.lower_bound:
                # print('Pruning state at processing time:', self.recursive_calls, 'lower_bound:', self.lower_bound, 'upper_bound:', upper_bound)
                self.branches_pruned += 1
                self.pruning_depths.append(len(clustered))
                continue

            if self.params.track_bound_history:
                self.upper_bound_history.append((self.recursive_calls, upper_bound))
                self.lower_bound_history.append((self.recursive_calls, self.lower_bound))

            if self.recursive_calls > self.params.max_recursive_calls:
                return

            # Base case: no more nodes to assign
            if not remaining:
                if len(clustered) == self.k or self.params.search_up_to_k_communities and len(clustered) <= self.k:
                    if current_mod >= self.lower_bound:
                        self.lower_bound = current_mod
                        self.best_partition = clustered
                        if self.params.enable_debugging:
                            print('Best partition:', self.best_partition, current_mod, self.lower_bound, upper_bound)
                continue

            for new_clustered, new_remaining, new_mod in self._generate_branches(clustered, remaining, current_mod):
                if self.params.enable_pruning:
                    # Use size-constrained bound if constraints are present
                    if self.min_community_size is not None or self.max_community_size is not None:
                        new_upper_bound = self._size_constrained_upper_bound(new_mod, new_clustered, new_remaining)
                    else:
                        new_upper_bound = self._updated_upper_bound(new_mod, new_clustered, new_remaining)

                    if new_upper_bound < self.lower_bound:
                        self.branches_pruned += 1
                        self.pruning_depths.append(len(new_clustered))
                        continue
                else:
                    new_upper_bound = upper_bound

                next_level_states.append((new_clustered, new_remaining, new_mod, new_upper_bound))

        if self.params.enable_pruning and next_level_states:
            initial_count = len(next_level_states)
            filtered_next_level = []
            for state in next_level_states:
                clustered, remaining, current_mod, upper_bound = state
                if upper_bound >= self.lower_bound:
                    filtered_next_level.append(state)
                else:
                    # Prune this state retroactively due to improved lower bound
                    self.branches_pruned += 1
                    self.pruning_depths.append(len(clustered))
                    if self.params.enable_debugging:
                        print(f'Retroactively pruned: upper_bound={upper_bound:.4f} < updated lower_bound={self.lower_bound:.4f}')
            pruned_count = initial_count - len(filtered_next_level)
            if self.params.enable_debugging and pruned_count > 0:
                print(f'Retroactive filtering: {pruned_count} states pruned out of {initial_count} (kept {len(filtered_next_level)})')
            next_level_states = filtered_next_level

        # Recursively process next level
        self._bfs(next_level_states)


    def set_partially_known_clustering(self):
        # define known clustering there
        # warning: if not optimal clustering is provided and heuristic lower bound is calculated, algorithm may fail to
        # surpass heuristic solution
        clustered = [
            {4, 5, 6, 10, 16},
            {14, 15, 18, 20, 22, 32, 33},
        ]
        assigned_nodes = set()
        for cluster in clustered:
            assigned_nodes.update(cluster)
        remaining = [node for node in self.nodes if node not in assigned_nodes]
        return clustered, remaining


    def solve(self, k, min_community_size=None, max_community_size=None):
        """
        Solve k-constrained modularity maximization with optional size constraints.

        Args:
            k: Number of communities
            min_community_size: Minimum nodes per community (None = no constraint)
            max_community_size: Maximum nodes per community (None = no constraint)

        Returns:
            (best_partition, modularity) or (None, -inf) if infeasible
        """
        self.min_community_size = min_community_size
        self.max_community_size = max_community_size

        is_valid, result = self._validate_inputs(k)
        if not is_valid:
            return result

        self._apply_vertex_ordering()

        singleton_partition = [{node} for node in self.nodes]
        singleton_modularity = self.modularity(singleton_partition)
        upper_bound = 1 - 1/k
        self.k = k
        self._compute_subtree_sizes()
        print(f'Total search tree size: {self.total_leaves:.2e} leaves')
        if self.params.leiden_lower_bound_iterations and self.params.leiden_lower_bound_iterations > 0:
            self.k_leiden = self._initialize_with_leiden(k, singleton_modularity, self.params.leiden_lower_bound_iterations)
            self.lower_bound = self.k_leiden
        else:
            self.lower_bound = singleton_modularity

        print(f'Initial vertex ordering .......:', self.nodes)
        print(f'Initial vertex ordering degrees:', [self.degrees[node] for node in self.nodes])
        print(f'Initial lower bound: {self.lower_bound}')
        if self.min_community_size is not None or self.max_community_size is not None:
            constraint_info = []
            if self.min_community_size is not None:
                constraint_info.append(f'min_size={self.min_community_size}')
            if self.max_community_size is not None:
                constraint_info.append(f'max_size={self.max_community_size}')
            print(f'Community size constraints active: {", ".join(constraint_info)}')
        if self.params.enable_debugging:
            print(f'Vertex ordering technique: {self.params.vertex_ordering}')
            print(f'Initial modularity: {singleton_modularity}')
            print(f'Initial upper bound: {upper_bound}')
            if self.min_community_size is not None or self.max_community_size is not None:
                constraint_info = []
                if self.min_community_size is not None:
                    constraint_info.append(f'min={self.min_community_size}')
                if self.max_community_size is not None:
                    constraint_info.append(f'max={self.max_community_size}')
                print(f'Community size constraints: {", ".join(constraint_info)}')

        if self.params.provide_partial_clustering:
            clustered, nodes = self.set_partially_known_clustering()
        else:
            clustered, nodes = [], self.nodes
        if clustered:
            full_partition = clustered + [{node} for node in nodes]
            current_mod = self.modularity(full_partition)
            if self.params.enable_debugging:
                print(f'Starting with warm-start clustered: {clustered}')
                print(f'Warm-start modularity: {current_mod:.6f}')
                print(f'Remaining nodes to assign: {len(nodes)}')
        else:
            current_mod = singleton_modularity

        if self.params.enable_debugging:
            print('recursive calls | clustered and remaining | current_mod | lower_bound | upper_bound | explored area')

        # Dispatch to the appropriate search strategy
        # All strategies (except bfs) use incremental upper bound calculation for better performance
        if self.params.search_strategy == 'dfs':
            # self._dfs(clustered, nodes, current_mod, upper_bound)
            initial_state = IncrementalBoundState(self, nodes)
            self._dfs_incremental(clustered, nodes, current_mod, upper_bound, initial_state)
        elif self.params.search_strategy == 'bfs':
            self._bfs([(clustered, nodes, current_mod, upper_bound)])
        elif self.params.search_strategy == 'bestfs':
            # self._bestfs(clustered, nodes, current_mod, upper_bound)
            initial_state = IncrementalBoundState(self, nodes)
            self._bestfs_incremental(clustered, nodes, current_mod, upper_bound, initial_state)
        elif self.params.search_strategy == 'localfs':
            heuristic_assignment = self._get_heuristic_vertex_assignment()
            # self._localfs(clustered, nodes, current_mod, upper_bound, heuristic_assignment, {})
            initial_state = IncrementalBoundState(self, nodes)
            self._localfs_incremental(clustered, nodes, current_mod, upper_bound, heuristic_assignment, {}, initial_state)
        else:
            raise ValueError(f"Unknown search strategy: {self.params.search_strategy}. "
                           f"Valid options: 'dfs', 'bfs', 'bestfs', 'localfs'")

        if self.params.enable_debugging:
            print(f"Performance stats - recursive calls: {self.recursive_calls}, branches pruned: {self.branches_pruned}")
            if self.pruning_depths:
                # print("Pruning depths for debugging:", self.pruning_depths)
                avg_pruning_depth = sum(self.pruning_depths) / len(self.pruning_depths)
                print(f"Average pruning depth: {avg_pruning_depth:.2f}")
            else:
                print("No pruning occurred")
        return self.best_partition, self.lower_bound
