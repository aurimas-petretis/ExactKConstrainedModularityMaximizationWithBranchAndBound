#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "../include/bnb_types.h"

/* External declarations */
Graph* graph_from_adj_matrix(int n, const int* adj_matrix);
void graph_free(Graph* g);
void graph_order_by_degree_desc(Graph* g);
Partition* partition_create(int capacity);
void partition_free(Partition* p);
void leiden_init_random(const Graph* g, Partition* p, int k, unsigned seed);
void leiden_optimize(const Graph* g, Partition* p, int k);
double solver_modularity(const BnBSolver* s, const Partition* p);

static void print_usage(const char* prog_name) {
    fprintf(stderr, "Usage: %s <k> [leiden_iterations]\n", prog_name);
    fprintf(stderr, "  Reads adjacency matrix from stdin.\n");
    fprintf(stderr, "  Input format:\n");
    fprintf(stderr, "    First line: n (number of vertices)\n");
    fprintf(stderr, "    Next n lines: adjacency matrix rows (space-separated 0/1)\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    int k = atoi(argv[1]);
    int leiden_iterations = (argc > 2) ? atoi(argv[2]) : 100;

    /* Read n */
    int n;
    if (scanf("%d", &n) != 1 || n <= 0) {
        fprintf(stderr, "Error: Invalid number of vertices\n");
        return 1;
    }

    /* Read adjacency matrix */
    int* adj = (int*)malloc(n * n * sizeof(int));
    for (int i = 0; i < n * n; i++) {
        if (scanf("%d", &adj[i]) != 1) {
            fprintf(stderr, "Error: Failed to read adjacency matrix\n");
            free(adj);
            return 1;
        }
    }

    /* Build graph */
    Graph* g = graph_from_adj_matrix(n, adj);
    free(adj);

    /* Validate */
    if (k <= 0 || k > n || g->m == 0) {
        fprintf(stderr, "Error: Invalid parameters (k=%d, n=%d, m=%d)\n", k, n, g->m);
        graph_free(g);
        return 1;
    }

    /* Order vertices by degree descending (needed for Leiden) */
    graph_order_by_degree_desc(g);

    /* Create temporary solver struct for modularity calculation */
    BnBSolver temp_solver;
    temp_solver.graph = g;
    temp_solver.k = k;

    /* Run Leiden algorithm */
    double best_mod = -DBL_MAX;
    Partition* best_partition = NULL;

    for (int iter = 0; iter < leiden_iterations; iter++) {
        Partition* p = partition_create(k);

        leiden_init_random(g, p, k, (unsigned)iter);

        leiden_optimize(g, p, k);

        double mod = solver_modularity(&temp_solver, p);

        if (mod > best_mod) {
            best_mod = mod;
            if (best_partition) {
                partition_free(best_partition);
            }
            best_partition = p;
        } else {
            partition_free(p);
        }
    }

    /* Output JSON result */
    printf("{\n");
    printf("  \"modularity\": %.15f,\n", best_mod);

    printf("  \"partition\": [");
    if (best_partition) {
        for (int c = 0; c < best_partition->num_communities; c++) {
            printf("[");
            for (int i = 0; i < best_partition->communities[c].size; i++) {
                printf("%d", best_partition->communities[c].members[i]);
                if (i < best_partition->communities[c].size - 1) {
                    printf(", ");
                }
            }
            printf("]");
            if (c < best_partition->num_communities - 1) {
                printf(", ");
            }
        }
    }
    printf("]\n");
    printf("}\n");

    /* Cleanup */
    if (best_partition) {
        partition_free(best_partition);
    }
    graph_free(g);

    return 0;
}