#include "../include/bnb_types.h"
#include <string.h>

Graph* graph_create(int n) {
    Graph* g = (Graph*)malloc(sizeof(Graph));
    g->n = n;
    g->m = 0;
    g->degrees = (int*)calloc(n, sizeof(int));
    g->adj = (int*)calloc(n * n, sizeof(int));
    g->nodes = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        g->nodes[i] = i;
    }
    return g;
}

Graph* graph_from_adj_matrix(int n, const int* adj_matrix) {
    Graph* g = graph_create(n);

    /* Count edges including self-loops */
    for (int i = 0; i < n; i++) {
        /* Self-loop */
        if (adj_matrix[i * n + i] != 0) {
            g->adj[i * n + i] = 1;
            g->m++;
        }
        /* Regular edges (upper triangle) */
        for (int j = i + 1; j < n; j++) {
            if (adj_matrix[i * n + j] != 0) {
                g->adj[i * n + j] = 1;
                g->adj[j * n + i] = 1;
                g->m++;
            }
        }
    }

    /* Compute degrees */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (g->adj[i * n + j]) {
                if (i == j) {
                    g->degrees[i] += 2;  /* Self-loop contributes 2 */
                } else {
                    g->degrees[i] += 1;
                }
            }
        }
    }

    return g;
}

void graph_free(Graph* g) {
    if (g) {
        free(g->degrees);
        free(g->adj);
        free(g->nodes);
        free(g);
    }
}

/* Comparison context for qsort_r */
typedef struct {
    const int* degrees;
} SortCtx;

/* macOS qsort_r has different signature: (ctx, a, b) instead of (a, b, ctx) */
#ifdef __APPLE__
static int compare_by_degree_desc(void* ctx, const void* a, const void* b) {
    SortCtx* c = (SortCtx*)ctx;
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    return c->degrees[ib] - c->degrees[ia];  /* Descending */
}
#else
static int compare_by_degree_desc(const void* a, const void* b, void* ctx) {
    SortCtx* c = (SortCtx*)ctx;
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    return c->degrees[ib] - c->degrees[ia];  /* Descending */
}
#endif

void graph_order_by_degree_desc(Graph* g) {
    SortCtx ctx = { g->degrees };
    #ifdef __APPLE__
    qsort_r(g->nodes, g->n, sizeof(int), &ctx, compare_by_degree_desc);
    #elif defined(__GLIBC__)
    qsort_r(g->nodes, g->n, sizeof(int), compare_by_degree_desc, &ctx);
    #else
    /* Fallback: simple insertion sort */
    for (int i = 1; i < g->n; i++) {
        int key = g->nodes[i];
        int j = i - 1;
        while (j >= 0 && g->degrees[g->nodes[j]] < g->degrees[key]) {
            g->nodes[j + 1] = g->nodes[j];
            j--;
        }
        g->nodes[j + 1] = key;
    }
    #endif
}