#include "../include/bnb_types.h"
#include <string.h>

IncrementalState* state_create(const Graph* g, int k) {
    IncrementalState* s = (IncrementalState*)malloc(sizeof(IncrementalState));
    s->graph = g;
    s->k = k;

    /* Initialize remaining bitset - all nodes are remaining initially */
    s->remaining = (int*)malloc(g->n * sizeof(int));
    for (int i = 0; i < g->n; i++) {
        s->remaining[i] = 1;
    }
    s->remaining_count = g->n;

    /* Initialize cluster degree sums */
    s->cluster_degree_sums = (double*)calloc(k, sizeof(double));
    s->num_clusters = 0;

    /* Count edges between remaining nodes (all nodes initially) */
    s->edges_between_remaining = 0;
    for (int i = 0; i < g->n; i++) {
        for (int j = i + 1; j < g->n; j++) {
            if (graph_has_edge(g, i, j)) {
                s->edges_between_remaining++;
            }
        }
    }

    /* Initialize node_to_cluster_edges */
    s->node_to_cluster_edges = (int*)calloc(g->n * k, sizeof(int));

    /* Calculate remaining degree penalty */
    double two_m = 2.0 * g->m;
    s->remaining_degree_penalty = 0.0;
    for (int i = 0; i < g->n; i++) {
        double deg_ratio = g->degrees[i] / two_m;
        s->remaining_degree_penalty += deg_ratio * deg_ratio;
    }

    return s;
}

IncrementalState* state_copy(const IncrementalState* s) {
    IncrementalState* copy = (IncrementalState*)malloc(sizeof(IncrementalState));
    copy->graph = s->graph;
    copy->k = s->k;

    int n = s->graph->n;
    int k = s->k;

    copy->remaining = (int*)malloc(n * sizeof(int));
    memcpy(copy->remaining, s->remaining, n * sizeof(int));
    copy->remaining_count = s->remaining_count;

    copy->cluster_degree_sums = (double*)malloc(k * sizeof(double));
    memcpy(copy->cluster_degree_sums, s->cluster_degree_sums, k * sizeof(double));
    copy->num_clusters = s->num_clusters;

    copy->edges_between_remaining = s->edges_between_remaining;

    copy->node_to_cluster_edges = (int*)malloc(n * k * sizeof(int));
    memcpy(copy->node_to_cluster_edges, s->node_to_cluster_edges, n * k * sizeof(int));

    copy->remaining_degree_penalty = s->remaining_degree_penalty;

    return copy;
}

void state_free(IncrementalState* s) {
    if (s) {
        free(s->remaining);
        free(s->cluster_degree_sums);
        free(s->node_to_cluster_edges);
        free(s);
    }
}

void state_update_join_cluster(IncrementalState* s, int vertex, int cluster_idx) {
    const Graph* g = s->graph;
    double two_m = 2.0 * g->m;

    /* Update cluster degree sum */
    s->cluster_degree_sums[cluster_idx] += g->degrees[vertex];

    /* Remove from remaining */
    s->remaining[vertex] = 0;
    s->remaining_count--;

    /* Update edges_between_remaining and node_to_cluster_edges */
    for (int other = 0; other < g->n; other++) {
        if (s->remaining[other] && graph_has_edge(g, vertex, other)) {
            s->edges_between_remaining--;
            s->node_to_cluster_edges[other * s->k + cluster_idx]++;
        }
    }

    /* Update remaining degree penalty */
    double deg_ratio = g->degrees[vertex] / two_m;
    s->remaining_degree_penalty -= deg_ratio * deg_ratio;
}

void state_update_new_singleton(IncrementalState* s, int vertex) {
    const Graph* g = s->graph;
    double two_m = 2.0 * g->m;
    int new_cluster_idx = s->num_clusters;

    /* Add new cluster with this vertex's degree */
    s->cluster_degree_sums[new_cluster_idx] = g->degrees[vertex];
    s->num_clusters++;

    /* Remove from remaining */
    s->remaining[vertex] = 0;
    s->remaining_count--;

    /* Update edges_between_remaining and add new column to node_to_cluster_edges */
    for (int other = 0; other < g->n; other++) {
        if (s->remaining[other]) {
            if (graph_has_edge(g, vertex, other)) {
                s->edges_between_remaining--;
                s->node_to_cluster_edges[other * s->k + new_cluster_idx] = 1;
            } else {
                s->node_to_cluster_edges[other * s->k + new_cluster_idx] = 0;
            }
        }
    }

    /* Update remaining degree penalty */
    double deg_ratio = g->degrees[vertex] / two_m;
    s->remaining_degree_penalty -= deg_ratio * deg_ratio;
}

/* UNDO STACK OPERATIONS - In-place state modification with rollback support */

/* Apply join cluster operation and save undo information */
void state_apply_join_cluster(IncrementalState* s, int vertex, int cluster_idx,
                               StateUndo* undo) {
    const Graph* g = s->graph;
    double two_m = 2.0 * g->m;

    /* Save undo information */
    undo->vertex = vertex;
    undo->cluster_idx = cluster_idx;
    undo->is_new_singleton = 0;
    undo->prev_edges_between_remaining = s->edges_between_remaining;
    undo->prev_remaining_degree_penalty = s->remaining_degree_penalty;
    undo->prev_num_clusters = s->num_clusters;
    undo->prev_cluster_degree_sum = s->cluster_degree_sums[cluster_idx];

    /* Update cluster degree sum */
    s->cluster_degree_sums[cluster_idx] += g->degrees[vertex];

    /* Remove from remaining */
    s->remaining[vertex] = 0;
    s->remaining_count--;

    /* Update edges_between_remaining and node_to_cluster_edges */
    for (int other = 0; other < g->n; other++) {
        if (s->remaining[other] && graph_has_edge(g, vertex, other)) {
            s->edges_between_remaining--;
            s->node_to_cluster_edges[other * s->k + cluster_idx]++;
        }
    }

    /* Update remaining degree penalty */
    double deg_ratio = g->degrees[vertex] / two_m;
    s->remaining_degree_penalty -= deg_ratio * deg_ratio;
}

/* Apply new singleton operation and save undo information */
void state_apply_new_singleton(IncrementalState* s, int vertex, StateUndo* undo) {
    const Graph* g = s->graph;
    double two_m = 2.0 * g->m;
    int new_cluster_idx = s->num_clusters;

    /* Save undo information */
    undo->vertex = vertex;
    undo->cluster_idx = new_cluster_idx;
    undo->is_new_singleton = 1;
    undo->prev_edges_between_remaining = s->edges_between_remaining;
    undo->prev_remaining_degree_penalty = s->remaining_degree_penalty;
    undo->prev_num_clusters = s->num_clusters;
    undo->prev_cluster_degree_sum = 0.0;  /* Not used for singleton */

    /* Add new cluster with this vertex's degree */
    s->cluster_degree_sums[new_cluster_idx] = g->degrees[vertex];
    s->num_clusters++;

    /* Remove from remaining */
    s->remaining[vertex] = 0;
    s->remaining_count--;

    /* Update edges_between_remaining and node_to_cluster_edges */
    for (int other = 0; other < g->n; other++) {
        if (s->remaining[other]) {
            if (graph_has_edge(g, vertex, other)) {
                s->edges_between_remaining--;
                s->node_to_cluster_edges[other * s->k + new_cluster_idx] = 1;
            } else {
                s->node_to_cluster_edges[other * s->k + new_cluster_idx] = 0;
            }
        }
    }

    /* Update remaining degree penalty */
    double deg_ratio = g->degrees[vertex] / two_m;
    s->remaining_degree_penalty -= deg_ratio * deg_ratio;
}

/* Rollback state to before the apply operation */
void state_rollback(IncrementalState* s, const StateUndo* undo) {
    const Graph* g = s->graph;
    int cluster_idx = undo->cluster_idx;
    int vertex = undo->vertex;

    /* Restore remaining FIRST (so rescan finds this vertex as remaining) */
    s->remaining[vertex] = 1;
    s->remaining_count++;

    /* Restore scalar values */
    s->edges_between_remaining = undo->prev_edges_between_remaining;
    s->remaining_degree_penalty = undo->prev_remaining_degree_penalty;

    if (undo->is_new_singleton) {
        /* Remove the cluster that was added */
        s->num_clusters = undo->prev_num_clusters;
        s->cluster_degree_sums[cluster_idx] = 0.0;

        /* Reset edge counts for remaining nodes connected to vertex */
        for (int other = 0; other < g->n; other++) {
            if (s->remaining[other] && other != vertex && graph_has_edge(g, vertex, other)) {
                s->node_to_cluster_edges[other * s->k + cluster_idx] = 0;
            }
        }
    } else {
        /* Restore cluster degree sum */
        s->cluster_degree_sums[cluster_idx] = undo->prev_cluster_degree_sum;

        /* Decrement edge counts for remaining nodes connected to vertex */
        for (int other = 0; other < g->n; other++) {
            if (s->remaining[other] && other != vertex && graph_has_edge(g, vertex, other)) {
                s->node_to_cluster_edges[other * s->k + cluster_idx]--;
            }
        }
    }
}