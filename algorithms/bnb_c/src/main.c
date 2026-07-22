#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/bnb_types.h"

/* External declarations */
Graph* graph_from_adj_matrix(int n, const int* adj_matrix);
void graph_free(Graph* g);
BnBSolver* solver_create(Graph* g, int leiden_iters);
void solver_solve(BnBSolver* s, int k);
void solver_solve_steal_parallel(BnBSolver* s, int k, int num_threads);
void solver_free(BnBSolver* s);

static void print_usage(const char* prog_name) {
    fprintf(stderr, "Usage: %s <k> [leiden_iterations] [threads] [variant]\n", prog_name);
    fprintf(stderr, "  Reads adjacency matrix from stdin.\n");
    fprintf(stderr, "  Input format:\n");
    fprintf(stderr, "    First line: n (number of vertices)\n");
    fprintf(stderr, "    Next n lines: adjacency matrix rows (space-separated 0/1)\n");
    fprintf(stderr, "  threads: number of threads (0 or omit for sequential)\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    int k = atoi(argv[1]);
    int leiden_iterations = (argc > 2) ? atoi(argv[2]) : 0;
    int num_threads = (argc > 3) ? atoi(argv[3]) : 0;

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

    /* Build graph and solve */
    Graph* g = graph_from_adj_matrix(n, adj);
    free(adj);

    BnBSolver* solver = solver_create(g, leiden_iterations);

    if (num_threads > 0) {
        solver_solve_steal_parallel(solver, k, num_threads);
    } else {
        solver_solve(solver, k);
    }

    /* Output JSON result */
    printf("{\n");
    printf("  \"modularity\": %.15f,\n", solver->lower_bound);

    printf("  \"partition\": [");
    if (solver->best_partition) {
        for (int c = 0; c < solver->best_partition->num_communities; c++) {
            printf("[");
            for (int i = 0; i < solver->best_partition->communities[c].size; i++) {
                printf("%d", solver->best_partition->communities[c].members[i]);
                if (i < solver->best_partition->communities[c].size - 1) {
                    printf(", ");
                }
            }
            printf("]");
            if (c < solver->best_partition->num_communities - 1) {
                printf(", ");
            }
        }
    }
    printf("],\n");

    printf("  \"recursive_calls\": %lld,\n", (long long)solver->recursive_calls);
    printf("  \"initial_lower_bound\": %.15f\n", solver->initial_lower_bound);
    printf("}\n");

    /* Cleanup */
    solver_free(solver);
    graph_free(g);

    return 0;
}