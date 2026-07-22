#include "../include/bnb_types.h"
#include <string.h>
#include <math.h>
#include <float.h>

/* Forward declarations */
void partition_add_community(Partition* p);
void partition_pop_community(Partition* p);
void partition_push_node(Partition* p, int comm_idx, int node);
void partition_pop_node(Partition* p, int comm_idx);
void partition_add_singleton(Partition* p, int node);
Partition* partition_create(int capacity);
void partition_free(Partition* p);
Partition* partition_copy(const Partition* p);
IncrementalState* state_create(const Graph* g, int k);
IncrementalState* state_copy(const IncrementalState* s);
void state_free(IncrementalState* s);
void state_update_join_cluster(IncrementalState* s, int vertex, int cluster_idx);
void state_update_new_singleton(IncrementalState* s, int vertex);
void graph_order_by_degree_desc(Graph* g);
/* Undo stack operations */
void state_apply_join_cluster(IncrementalState* s, int vertex, int cluster_idx,
                               StateUndo* undo);
void state_apply_new_singleton(IncrementalState* s, int vertex, StateUndo* undo);
void state_rollback(IncrementalState* s, const StateUndo* undo);

/* Random number generator (simple LCG) */
static unsigned int g_rand_state = 12345;

static unsigned int rand_next(void) {
    g_rand_state = g_rand_state * 1103515245 + 12345;
    return (g_rand_state >> 16) & 0x7fff;
}

static void rand_seed(unsigned int seed) {
    g_rand_state = seed;
}

/* Fisher-Yates shuffle */
static void shuffle(int* arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand_next() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

/* Calculate modularity */
double solver_modularity(const BnBSolver* s, const Partition* p) {
    const Graph* g = s->graph;
    double mod_sum = 0.0;
    double two_m = 2.0 * g->m;

    for (int c = 0; c < p->num_communities; c++) {
        const Community* comm = &p->communities[c];
        int edges_in_community = 0;
        int degree_sum = 0;

        for (int i = 0; i < comm->size; i++) {
            int node_i = comm->members[i];
            degree_sum += g->degrees[node_i];
            /* Count edges to other nodes in community (NOT self-loops) */
            for (int j = i + 1; j < comm->size; j++) {
                if (graph_has_edge(g, node_i, comm->members[j])) {
                    edges_in_community++;
                }
            }
        }

        double contrib = (double)edges_in_community / g->m
                       - square(degree_sum / two_m);
        mod_sum += contrib;
    }

    return mod_sum;
}

/* Local modularity change when adding node to community - FAST version using precomputed state */
static inline double solver_local_mod_change_fast(const BnBSolver* s,
                                                   const IncrementalState* state,
                                                   int cluster_idx, int new_node) {
    const Graph* g = s->graph;

    /* Use precomputed values - O(1) instead of O(community_size) */
    int additional_edges = state->node_to_cluster_edges[new_node * s->k + cluster_idx];
    double subset_degree_sum = state->cluster_degree_sums[cluster_idx];

    double two_m = 2.0 * g->m;
    double edge_change = (double)additional_edges / g->m;
    double degree_change = -2.0 * subset_degree_sum * g->degrees[new_node] / (two_m * two_m);

    return edge_change + degree_change;
}

/* Local modularity change when adding node to community - original for Leiden */
double solver_local_mod_change(const BnBSolver* s, const Community* comm, int new_node) {
    const Graph* g = s->graph;
    int additional_edges = 0;
    int subset_degree_sum = 0;

    for (int i = 0; i < comm->size; i++) {
        int node = comm->members[i];
        if (graph_has_edge(g, new_node, node)) {
            additional_edges++;
        }
        subset_degree_sum += g->degrees[node];
    }

    double two_m = 2.0 * g->m;
    double edge_change = (double)additional_edges / g->m;
    double degree_change = -2.0 * subset_degree_sum * g->degrees[new_node] / (two_m * two_m);

    return edge_change + degree_change;
}


/* Upper bound calculation - uses pre-allocated workspaces to avoid malloc in hot path */
double solver_upper_bound(const BnBSolver* s, double current_mod, const IncrementalState* state) {
    if (state->remaining_count == 0) {
        return current_mod;
    }

    const Graph* g = state->graph;
    int k = s->k;
    double two_m = 2.0 * g->m;

    /* Max cluster edges from remaining nodes - original admissible bound */
    int max_cluster_edges = 0;
    if (state->num_clusters > 0) {
        for (int node = 0; node < g->n; node++) {
            if (state->remaining[node]) {
                int* edges = &state->node_to_cluster_edges[node * k];
                int best = edges[0];
                for (int c = 1; c < state->num_clusters; c++) {
                    if (edges[c] > best) best = edges[c];
                }
                max_cluster_edges += best;
            }
        }
    }

    double optimistic_edge_contrib =
        (double)(state->edges_between_remaining + max_cluster_edges) / g->m;

    /* Balanced degree distribution - use pre-allocated workspace */
    double total_degree = two_m;
    double ideal_per_cluster = total_degree / k;

    double* final_degrees = s->ub_final_degrees;  /* Pre-allocated */
    for (int i = 0; i < k; i++) {
        final_degrees[i] = (i < state->num_clusters) ? state->cluster_degree_sums[i] : 0.0;
    }

    /* Identify locked clusters - use pre-allocated workspace */
    int* locked = s->ub_locked;  /* Pre-allocated */
    memset(locked, 0, k * sizeof(int));  /* Clear instead of calloc */
    double locked_sum = 0.0;
    int locked_count = 0;
    for (int i = 0; i < state->num_clusters; i++) {
        if (state->cluster_degree_sums[i] >= ideal_per_cluster) {
            locked[i] = 1;
            locked_sum += state->cluster_degree_sums[i];
            locked_count++;
        }
    }

    int non_locked_count = k - locked_count;
    if (non_locked_count > 0) {
        double target = (total_degree - locked_sum) / non_locked_count;
        for (int i = 0; i < k; i++) {
            if (!locked[i]) {
                final_degrees[i] = target;
            }
        }
    }

    /* Calculate penalties */
    double current_penalty = 0.0;
    for (int i = 0; i < state->num_clusters; i++) {
        current_penalty += square(state->cluster_degree_sums[i] / two_m);
    }
    current_penalty += state->remaining_degree_penalty;

    double optimistic_penalty = 0.0;
    for (int i = 0; i < k; i++) {
        optimistic_penalty += square(final_degrees[i] / two_m);
    }

    double degree_improvement = current_penalty - optimistic_penalty;
    double upper_bound = current_mod + optimistic_edge_contrib + degree_improvement;

    /* No free needed - workspaces are pre-allocated */

    double max_possible = 1.0 - 1.0 / k;
    return (upper_bound < max_possible) ? upper_bound : max_possible;
}

/* DFS search - uses undo stack for efficient backtracking (no state_copy/state_free) */
void solver_dfs(BnBSolver* s, Partition* clustered, int* remaining, int remaining_count,
                double current_mod, IncrementalState* state) {

    s->recursive_calls++;

    /* Base case: all nodes assigned */
    if (remaining_count == 0) {
        if (clustered->num_communities == s->k && current_mod > s->lower_bound) {
            s->lower_bound = current_mod;
            partition_free(s->best_partition);
            s->best_partition = partition_copy(clustered);
        }
        return;
    }

    int vertex = remaining[0];
    int* remaining_rest = remaining + 1;
    int remaining_rest_count = remaining_count - 1;

    StateUndo undo;  /* Stack-allocated undo info */

    /* Branch 1: Add to each existing cluster */
    for (int i = 0; i < clustered->num_communities; i++) {
        /* Use fast O(1) lookup instead of O(community_size) scan */
        double delta_mod = solver_local_mod_change_fast(s, state, i, vertex);
        double new_mod = current_mod + delta_mod;

        /* Apply state change in-place (saves undo info) */
        state_apply_join_cluster(state, vertex, i, &undo);

        /* Calculate upper bound */
        double new_upper = solver_upper_bound(s, new_mod, state);

        /* Prune if upper bound < lower bound */
        if (new_upper < s->lower_bound) {
            state_rollback(state, &undo);
            continue;
        }

        /* Add vertex to cluster i */
        partition_push_node(clustered, i, vertex);

        /* Recurse */
        solver_dfs(s, clustered, remaining_rest, remaining_rest_count,
                   new_mod, state);

        /* Backtrack: restore partition and state */
        partition_pop_node(clustered, i);
        state_rollback(state, &undo);
    }

    /* Branch 2: Create new singleton (if < k clusters) */
    if (clustered->num_communities < s->k) {
        double new_mod = current_mod;  /* No modularity change for singleton */

        /* Apply state change in-place (saves undo info) */
        state_apply_new_singleton(state, vertex, &undo);

        double new_upper = solver_upper_bound(s, new_mod, state);

        if (new_upper >= s->lower_bound) {
            /* Add new singleton cluster */
            partition_add_singleton(clustered, vertex);

            solver_dfs(s, clustered, remaining_rest, remaining_rest_count,
                       new_mod, state);

            /* Backtrack */
            partition_pop_community(clustered);
        }

        state_rollback(state, &undo);
    }
}

/* K-constrained Leiden: random initialization */
void leiden_init_random(const Graph* g, Partition* p, int k, unsigned seed) {
    rand_seed(seed);

    int* nodes = (int*)malloc(g->n * sizeof(int));
    memcpy(nodes, g->nodes, g->n * sizeof(int));
    shuffle(nodes, g->n);

    /* Each community gets at least one node */
    int init_count = (k < g->n) ? k : g->n;
    for (int i = 0; i < init_count; i++) {
        partition_add_singleton(p, nodes[i]);
    }

    /* Assign remaining randomly */
    for (int i = k; i < g->n; i++) {
        int comm_idx = rand_next() % k;
        partition_push_node(p, comm_idx, nodes[i]);
    }

    free(nodes);
}

/* K-constrained Leiden: optimize partition */
void leiden_optimize(const Graph* g, Partition* p, int k) {
    double two_m = 2.0 * g->m;

    /* Create membership array */
    int* membership = (int*)malloc(g->n * sizeof(int));
    for (int c = 0; c < p->num_communities; c++) {
        for (int i = 0; i < p->communities[c].size; i++) {
            membership[p->communities[c].members[i]] = c;
        }
    }

    /* Create community sets for fast removal - use parallel arrays */
    int** comm_members = (int**)malloc(k * sizeof(int*));
    int* comm_sizes = (int*)malloc(k * sizeof(int));
    int* comm_degree_sums = (int*)calloc(k, sizeof(int));

    for (int c = 0; c < k; c++) {
        comm_members[c] = (int*)malloc(g->n * sizeof(int));
        comm_sizes[c] = p->communities[c].size;
        memcpy(comm_members[c], p->communities[c].members,
               p->communities[c].size * sizeof(int));
        for (int i = 0; i < comm_sizes[c]; i++) {
            comm_degree_sums[c] += g->degrees[comm_members[c][i]];
        }
    }

    /* Local move phase */
    int improved = 1;
    while (improved) {
        improved = 0;

        for (int node = 0; node < g->n; node++) {
            int current_comm = membership[node];

            /* Can only move if current community has > 1 node */
            if (comm_sizes[current_comm] <= 1) {
                continue;
            }

            int node_degree = g->degrees[node];

            /* Calculate edges to current community */
            int edges_to_current = 0;
            for (int i = 0; i < comm_sizes[current_comm]; i++) {
                int other = comm_members[current_comm][i];
                if (other != node && graph_has_edge(g, node, other)) {
                    edges_to_current++;
                }
            }

            double best_delta = 0.0;
            int best_target = -1;

            /* Try moving to each other community */
            for (int target_comm = 0; target_comm < k; target_comm++) {
                if (target_comm == current_comm) continue;

                /* Calculate edges to target */
                int edges_to_target = 0;
                for (int i = 0; i < comm_sizes[target_comm]; i++) {
                    if (graph_has_edge(g, node, comm_members[target_comm][i])) {
                        edges_to_target++;
                    }
                }

                /* Calculate modularity delta */
                double edge_delta = (double)(edges_to_target - edges_to_current) / g->m;

                int target_degree_sum = comm_degree_sums[target_comm];
                int current_degree_sum = comm_degree_sums[current_comm] - node_degree;

                double old_penalty = square(comm_degree_sums[current_comm] / two_m)
                                   + square(target_degree_sum / two_m);
                double new_penalty = square(current_degree_sum / two_m)
                                   + square((target_degree_sum + node_degree) / two_m);
                double penalty_delta = old_penalty - new_penalty;

                double delta = edge_delta + penalty_delta;

                if (delta > best_delta + 1e-10) {
                    best_delta = delta;
                    best_target = target_comm;
                }
            }

            /* Make the move if it improves */
            if (best_target >= 0) {
                /* Remove from current */
                int idx = -1;
                for (int i = 0; i < comm_sizes[current_comm]; i++) {
                    if (comm_members[current_comm][i] == node) {
                        idx = i;
                        break;
                    }
                }
                if (idx >= 0) {
                    comm_members[current_comm][idx] =
                        comm_members[current_comm][comm_sizes[current_comm] - 1];
                    comm_sizes[current_comm]--;
                }

                /* Add to target */
                comm_members[best_target][comm_sizes[best_target]++] = node;

                membership[node] = best_target;
                comm_degree_sums[current_comm] -= node_degree;
                comm_degree_sums[best_target] += node_degree;

                improved = 1;
            }
        }
    }

    /* Copy back to partition */
    for (int c = 0; c < k; c++) {
        p->communities[c].size = 0;
        for (int i = 0; i < comm_sizes[c]; i++) {
            partition_push_node(p, c, comm_members[c][i]);
        }
    }

    /* Cleanup */
    for (int c = 0; c < k; c++) {
        free(comm_members[c]);
    }
    free(comm_members);
    free(comm_sizes);
    free(comm_degree_sums);
    free(membership);
}

/* Initialize solver with Leiden */
void solver_init_leiden(BnBSolver* s, int k) {
    if (s->leiden_iterations <= 0) return;

    double best_mod = s->lower_bound;

    for (int iter = 0; iter < s->leiden_iterations; iter++) {
        Partition* p = partition_create(k);

        leiden_init_random(s->graph, p, k, (unsigned)iter);

        leiden_optimize(s->graph, p, k);

        double mod = solver_modularity(s, p);

        if (mod > best_mod) {
            best_mod = mod;
            partition_free(s->best_partition);
            s->best_partition = partition_copy(p);
        }

        partition_free(p);
    }

    s->lower_bound = best_mod;
}

/* Calculate singleton modularity */
double solver_singleton_modularity(BnBSolver* s) {
    const Graph* g = s->graph;
    double mod_sum = 0.0;
    double two_m = 2.0 * g->m;

    for (int i = 0; i < g->n; i++) {
        /* Self-loop contribution is 0 (we don't count self-loops in edges) */
        /* Only degree penalty */
        mod_sum -= square(g->degrees[i] / two_m);
    }

    return mod_sum;
}

/* Main solve function */
void solver_solve(BnBSolver* s, int k) {
    s->k = k;
    s->lower_bound = -DBL_MAX;
    s->recursive_calls = 0;

    if (s->best_partition) {
        partition_free(s->best_partition);
    }
    s->best_partition = partition_create(k);

    /* Validate */
    if (k <= 0 || k > s->graph->n || s->graph->m == 0) {
        return;
    }

    /* Allocate/reallocate workspaces for upper_bound calculation */
    free(s->ub_final_degrees);
    free(s->ub_locked);
    s->ub_final_degrees = (double*)malloc(k * sizeof(double));
    s->ub_locked = (int*)malloc(k * sizeof(int));

    /* Order vertices by degree descending */
    graph_order_by_degree_desc(s->graph);

    /* Initialize lower bound */
    double singleton_mod = solver_singleton_modularity(s);
    s->lower_bound = singleton_mod;

    /* Initialize with Leiden */
    if (s->leiden_iterations > 0) {
        solver_init_leiden(s, k);
    }

    /* Record initial lower bound (from Leiden) before BnB search */
    s->initial_lower_bound = s->lower_bound;

    /* Initialize state */
    IncrementalState* initial_state = state_create(s->graph, k);

    /* Run DFS */
    Partition* clustered = partition_create(k);
    int* remaining = (int*)malloc(s->graph->n * sizeof(int));
    memcpy(remaining, s->graph->nodes, s->graph->n * sizeof(int));

    solver_dfs(s, clustered, remaining, s->graph->n,
               singleton_mod, initial_state);

    partition_free(clustered);
    state_free(initial_state);
    free(remaining);
}

BnBSolver* solver_create(Graph* g, int leiden_iters) {
    BnBSolver* s = (BnBSolver*)malloc(sizeof(BnBSolver));
    s->graph = g;
    s->k = 0;
    s->lower_bound = -DBL_MAX;
    s->initial_lower_bound = -DBL_MAX;
    s->best_partition = NULL;
    s->recursive_calls = 0;
    s->leiden_iterations = leiden_iters;
    /* Workspaces allocated lazily in solver_solve when k is known */
    s->ub_final_degrees = NULL;
    s->ub_locked = NULL;
    return s;
}

void solver_free(BnBSolver* s) {
    if (s) {
        if (s->best_partition) {
            partition_free(s->best_partition);
        }
        free(s->ub_final_degrees);
        free(s->ub_locked);
        free(s);
    }
}