#ifndef BNB_TYPES_H
#define BNB_TYPES_H

#include <stdint.h>
#include <stdlib.h>

/* Graph structure */
typedef struct {
    int n;              /* Number of vertices */
    int m;              /* Number of edges */
    int* degrees;       /* degrees[n] */
    int* adj;           /* Flattened adjacency matrix adj[n*n], row-major */
    int* nodes;         /* Vertex ordering nodes[n] */
} Graph;

/* Community (single cluster) */
typedef struct {
    int* members;       /* Node indices */
    int size;           /* Current size */
    int capacity;       /* Allocated capacity */
} Community;

/* Partition (collection of communities) */
typedef struct {
    Community* communities;
    int num_communities;
    int capacity;
} Partition;

/* Incremental bound state for DFS */
typedef struct {
    const Graph* graph;
    int k;                          /* Target k */

    int* remaining;                 /* Bitset: remaining[i] = 1 if node i is remaining */
    int remaining_count;

    double* cluster_degree_sums;    /* cluster_degree_sums[num_clusters] */
    int num_clusters;

    int edges_between_remaining;

    int* node_to_cluster_edges;     /* node_to_cluster_edges[node * k + cluster] */

    double remaining_degree_penalty;
} IncrementalState;

/* Undo information for state rollback (avoids expensive state_copy) */
typedef struct {
    int vertex;                     /* Vertex that was assigned */
    int cluster_idx;                /* Cluster it was assigned to */
    int is_new_singleton;           /* 1 if created new cluster, 0 if joined existing */

    /* Saved state for rollback */
    int prev_edges_between_remaining;
    double prev_remaining_degree_penalty;
    int prev_num_clusters;
    double prev_cluster_degree_sum; /* Only the affected cluster's previous value */
} StateUndo;

/* BnB Solver */
typedef struct {
    Graph* graph;
    int k;
    double lower_bound;
    double initial_lower_bound;     /* Lower bound after Leiden init, before BnB */
    Partition* best_partition;
    int64_t recursive_calls;
    int leiden_iterations;
    /* Pre-allocated workspaces for upper_bound calculation (avoid malloc in hot path) */
    double* ub_final_degrees;       /* Workspace of size k */
    int* ub_locked;                 /* Workspace of size k */
} BnBSolver;

/* Inline accessor for adjacency */
static inline int graph_has_edge(const Graph* g, int i, int j) {
    return g->adj[i * g->n + j];
}

static inline double square(double x) {
    return x * x;
}

#endif /* BNB_TYPES_H */