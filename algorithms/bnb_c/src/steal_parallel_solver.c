#include "../include/bnb_types.h"
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Forward declarations from other modules */
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
void state_apply_join_cluster(IncrementalState* s, int vertex, int cluster_idx,
                               StateUndo* undo);
void state_apply_new_singleton(IncrementalState* s, int vertex, StateUndo* undo);
void state_rollback(IncrementalState* s, const StateUndo* undo);
void graph_order_by_degree_desc(Graph* g);
double solver_modularity(const BnBSolver* s, const Partition* p);
double solver_singleton_modularity(BnBSolver* s);
void solver_init_leiden(BnBSolver* s, int k);

/* ============================================================================
 * STEALABLE WORK ITEM
 * Represents a branch point that can be stolen by another thread
 * ============================================================================ */

typedef struct {
    IncrementalState* state;      /* State at this branch point */
    Partition* partition;          /* Partition at this branch point */
    int* remaining;                /* Remaining vertices */
    int remaining_count;
    double current_mod;            /* Current modularity */
    int depth;                     /* Depth in search tree */
} StealableWork;

/* ============================================================================
 * PER-THREAD WORK DEQUE
 * Local thread pushes/pops from top (LIFO for DFS)
 * Thieves steal from bottom (shallowest depth)
 * ============================================================================ */

#define DEQUE_INITIAL_CAPACITY 256

typedef struct {
    StealableWork* items;
    int capacity;
    int size;                      /* Number of items */
#ifdef _OPENMP
    omp_lock_t lock;               /* Protects steal operations */
#endif
} WorkDeque;

static void deque_init(WorkDeque* d) {
    d->items = (StealableWork*)malloc(DEQUE_INITIAL_CAPACITY * sizeof(StealableWork));
    d->capacity = DEQUE_INITIAL_CAPACITY;
    d->size = 0;
#ifdef _OPENMP
    omp_init_lock(&d->lock);
#endif
}

static void deque_free(WorkDeque* d) {
    /* Free any remaining work items */
    for (int i = 0; i < d->size; i++) {
        state_free(d->items[i].state);
        partition_free(d->items[i].partition);
        free(d->items[i].remaining);
    }
    free(d->items);
#ifdef _OPENMP
    omp_destroy_lock(&d->lock);
#endif
}

/* Push to top (local thread only, no lock needed for push) */
static void deque_push(WorkDeque* d, StealableWork* work) {
#ifdef _OPENMP
    omp_set_lock(&d->lock);
#endif
    if (d->size >= d->capacity) {
        d->capacity *= 2;
        d->items = (StealableWork*)realloc(d->items, d->capacity * sizeof(StealableWork));
    }
    d->items[d->size++] = *work;
#ifdef _OPENMP
    omp_unset_lock(&d->lock);
#endif
}

/* Pop from top (local thread) - returns 1 if got work, 0 if empty */
static int deque_pop(WorkDeque* d, StealableWork* out) {
#ifdef _OPENMP
    omp_set_lock(&d->lock);
#endif
    if (d->size == 0) {
#ifdef _OPENMP
        omp_unset_lock(&d->lock);
#endif
        return 0;
    }
    *out = d->items[--d->size];
#ifdef _OPENMP
    omp_unset_lock(&d->lock);
#endif
    return 1;
}

/* Steal from bottom (thief thread) - gets shallowest work */
static int deque_steal(WorkDeque* d, StealableWork* out) {
#ifdef _OPENMP
    omp_set_lock(&d->lock);
#endif
    if (d->size == 0) {
#ifdef _OPENMP
        omp_unset_lock(&d->lock);
#endif
        return 0;
    }
    /* Steal from index 0 (shallowest) */
    *out = d->items[0];
    /* Shift remaining items down */
    for (int i = 1; i < d->size; i++) {
        d->items[i-1] = d->items[i];
    }
    d->size--;
#ifdef _OPENMP
    omp_unset_lock(&d->lock);
#endif
    return 1;
}

/* Get depth of shallowest item (for choosing victim) */
static int deque_min_depth(WorkDeque* d) {
#ifdef _OPENMP
    omp_set_lock(&d->lock);
#endif
    int result = (d->size > 0) ? d->items[0].depth : INT32_MAX;
#ifdef _OPENMP
    omp_unset_lock(&d->lock);
#endif
    return result;
}

/* ============================================================================
 * UPPER BOUND CALCULATION (thread-local workspace)
 * ============================================================================ */

static double compute_upper_bound(const BnBSolver* s, double current_mod,
                                   const IncrementalState* state,
                                   double* final_degrees, int* locked) {
    if (state->remaining_count == 0) {
        return current_mod;
    }

    const Graph* g = state->graph;
    int k = s->k;
    double two_m = 2.0 * g->m;

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

    double total_degree = two_m;
    double ideal_per_cluster = total_degree / k;

    for (int i = 0; i < k; i++) {
        final_degrees[i] = (i < state->num_clusters) ? state->cluster_degree_sums[i] : 0.0;
    }

    memset(locked, 0, k * sizeof(int));
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

    double max_possible = 1.0 - 1.0 / k;
    return (upper_bound < max_possible) ? upper_bound : max_possible;
}

static inline double local_mod_change(const BnBSolver* s,
                                       const IncrementalState* state,
                                       int cluster_idx, int new_node) {
    const Graph* g = s->graph;
    int additional_edges = state->node_to_cluster_edges[new_node * s->k + cluster_idx];
    double subset_degree_sum = state->cluster_degree_sums[cluster_idx];

    double two_m = 2.0 * g->m;
    double edge_change = (double)additional_edges / g->m;
    double degree_change = -2.0 * subset_degree_sum * g->degrees[new_node] / (two_m * two_m);

    return edge_change + degree_change;
}

/* ============================================================================
 * THREAD CONTEXT
 * ============================================================================ */

typedef struct {
    BnBSolver* solver;
    int thread_id;
    int num_threads;
    WorkDeque* all_deques;         /* Array of all thread deques */
    WorkDeque* my_deque;           /* Pointer to this thread's deque */

    /* Thread-local workspaces */
    double* ub_final_degrees;
    int* ub_locked;

    /* Statistics */
    int64_t local_calls;
    int64_t stolen_count;

    /* Shared state */
#ifdef _OPENMP
    omp_lock_t* solution_lock;
#endif
    volatile int* global_done;
    volatile int* active_workers;  /* Count of threads currently doing work */
} ThreadContext;

/* ============================================================================
 * DFS WITH WORK SHARING
 * ============================================================================ */

#ifdef _OPENMP
static void try_update_solution(BnBSolver* s, double mod, Partition* partition,
                                 omp_lock_t* solution_lock) {
    omp_set_lock(solution_lock);
    if (mod > s->lower_bound) {
        s->lower_bound = mod;
        if (s->best_partition) {
            partition_free(s->best_partition);
        }
        s->best_partition = partition_copy(partition);
    }
    omp_unset_lock(solution_lock);
}

/* DFS that occasionally checks if it should share work */
static void dfs_with_sharing(ThreadContext* ctx,
                              Partition* partition,
                              int* remaining,
                              int remaining_count,
                              double current_mod,
                              IncrementalState* state,
                              int depth) {
    BnBSolver* s = ctx->solver;

    ctx->local_calls++;

    /* Base case */
    if (remaining_count == 0) {
        if (partition->num_communities == s->k) {
            try_update_solution(s, current_mod, partition, ctx->solution_lock);
        }
        return;
    }

    int vertex = remaining[0];
    int* remaining_rest = remaining + 1;
    int remaining_rest_count = remaining_count - 1;

    /* Count valid branches */
    int num_branches = partition->num_communities + (partition->num_communities < s->k ? 1 : 0);

    /* Get current lower bound */
    double current_lower;
    #pragma omp atomic read
    current_lower = s->lower_bound;

    /* At shallow depths with multiple branches, consider sharing */
    int share_remaining = 0;
    if (depth < s->graph->n / 2 && num_branches > 1) {
        /* Check if any thread needs work (simple heuristic) */
        for (int t = 0; t < ctx->num_threads; t++) {
            if (t != ctx->thread_id && ctx->all_deques[t].size == 0) {
                share_remaining = 1;
                break;
            }
        }
    }

    int branches_done = 0;
    StateUndo undo;

    /* Branch: Add to existing clusters */
    for (int c = 0; c < partition->num_communities; c++) {
        double delta = local_mod_change(s, state, c, vertex);
        double new_mod = current_mod + delta;

        state_apply_join_cluster(state, vertex, c, &undo);

        double upper = compute_upper_bound(s, new_mod, state,
                                            ctx->ub_final_degrees, ctx->ub_locked);

        /* Re-read lower bound */
        #pragma omp atomic read
        current_lower = s->lower_bound;

        if (upper >= current_lower) {
            /* If sharing and this isn't the first branch, push to deque for stealing */
            if (share_remaining && branches_done > 0) {
                StealableWork work = {
                    .state = state_copy(state),
                    .partition = partition_copy(partition),
                    .remaining = (int*)malloc(remaining_rest_count * sizeof(int)),
                    .remaining_count = remaining_rest_count,
                    .current_mod = new_mod,
                    .depth = depth + 1
                };
                partition_push_node(work.partition, c, vertex);
                memcpy(work.remaining, remaining_rest, remaining_rest_count * sizeof(int));
                deque_push(ctx->my_deque, &work);
            } else {
                /* Process locally */
                partition_push_node(partition, c, vertex);
                dfs_with_sharing(ctx, partition, remaining_rest, remaining_rest_count,
                                 new_mod, state, depth + 1);
                partition_pop_node(partition, c);
            }
        }

        state_rollback(state, &undo);
        branches_done++;
    }

    /* Branch: Create new singleton */
    if (partition->num_communities < s->k) {
        double new_mod = current_mod;

        state_apply_new_singleton(state, vertex, &undo);

        double upper = compute_upper_bound(s, new_mod, state,
                                            ctx->ub_final_degrees, ctx->ub_locked);

        #pragma omp atomic read
        current_lower = s->lower_bound;

        if (upper >= current_lower) {
            if (share_remaining && branches_done > 0) {
                StealableWork work = {
                    .state = state_copy(state),
                    .partition = partition_copy(partition),
                    .remaining = (int*)malloc(remaining_rest_count * sizeof(int)),
                    .remaining_count = remaining_rest_count,
                    .current_mod = new_mod,
                    .depth = depth + 1
                };
                partition_add_singleton(work.partition, vertex);
                memcpy(work.remaining, remaining_rest, remaining_rest_count * sizeof(int));
                deque_push(ctx->my_deque, &work);
            } else {
                partition_add_singleton(partition, vertex);
                dfs_with_sharing(ctx, partition, remaining_rest, remaining_rest_count,
                                 new_mod, state, depth + 1);
                partition_pop_community(partition);
            }
        }

        state_rollback(state, &undo);
    }
}

/* Try to steal work from another thread */
static int try_steal(ThreadContext* ctx, StealableWork* out) {
    int best_victim = -1;
    int best_depth = INT32_MAX;

    /* Find thread with shallowest stealable work */
    for (int t = 0; t < ctx->num_threads; t++) {
        if (t == ctx->thread_id) continue;

        int depth = deque_min_depth(&ctx->all_deques[t]);
        if (depth < best_depth) {
            best_depth = depth;
            best_victim = t;
        }
    }

    if (best_victim >= 0) {
        if (deque_steal(&ctx->all_deques[best_victim], out)) {
            ctx->stolen_count++;
            return 1;
        }
    }

    return 0;
}

/* Thread main loop */
static void thread_main(ThreadContext* ctx, StealableWork* initial_work) {
    /* Process initial work if given */
    if (initial_work) {
        #pragma omp atomic
        (*ctx->active_workers)++;

        dfs_with_sharing(ctx,
                         initial_work->partition,
                         initial_work->remaining,
                         initial_work->remaining_count,
                         initial_work->current_mod,
                         initial_work->state,
                         initial_work->depth);

        state_free(initial_work->state);
        partition_free(initial_work->partition);
        free(initial_work->remaining);

        #pragma omp atomic
        (*ctx->active_workers)--;
    }

    /* Work loop: process local deque, then try stealing */
    while (!(*ctx->global_done)) {
        StealableWork work;
        int got_work = 0;

        /* First try local deque */
        if (deque_pop(ctx->my_deque, &work)) {
            got_work = 1;
        }
        /* Then try stealing */
        else if (try_steal(ctx, &work)) {
            got_work = 1;
        }

        if (got_work) {
            #pragma omp atomic
            (*ctx->active_workers)++;

            dfs_with_sharing(ctx,
                             work.partition,
                             work.remaining,
                             work.remaining_count,
                             work.current_mod,
                             work.state,
                             work.depth);

            state_free(work.state);
            partition_free(work.partition);
            free(work.remaining);

            #pragma omp atomic
            (*ctx->active_workers)--;
            continue;
        }

        /* No work found - check termination */
        int active;
        #pragma omp atomic read
        active = *ctx->active_workers;

        if (active == 0) {
            /* Double-check: scan all deques */
            int all_empty = 1;
            for (int t = 0; t < ctx->num_threads; t++) {
                if (ctx->all_deques[t].size > 0) {
                    all_empty = 0;
                    break;
                }
            }
            if (all_empty) {
                *ctx->global_done = 1;
                break;
            }
        }

        /* Yield to let other threads work */
        #pragma omp taskyield
    }
}
#endif

/* ============================================================================
 * MAIN ENTRY POINT
 * ============================================================================ */

void solver_solve_steal_parallel(BnBSolver* s, int k, int num_threads) {
#ifdef _OPENMP
    s->k = k;
    s->lower_bound = -DBL_MAX;
    s->recursive_calls = 0;

    if (s->best_partition) {
        partition_free(s->best_partition);
    }
    s->best_partition = partition_create(k);

    if (k <= 0 || k > s->graph->n || s->graph->m == 0) {
        return;
    }

    graph_order_by_degree_desc(s->graph);

    double singleton_mod = solver_singleton_modularity(s);
    s->lower_bound = singleton_mod;

    if (s->leiden_iterations > 0) {
        solver_init_leiden(s, k);
    }
    s->initial_lower_bound = s->lower_bound;

    /* Create per-thread deques */
    WorkDeque* all_deques = (WorkDeque*)malloc(num_threads * sizeof(WorkDeque));
    for (int t = 0; t < num_threads; t++) {
        deque_init(&all_deques[t]);
    }

    omp_lock_t solution_lock;
    omp_init_lock(&solution_lock);

    volatile int global_done = 0;
    volatile int active_workers = 0;

    /* Create initial work for thread 0 */
    StealableWork initial_work = {
        .state = state_create(s->graph, k),
        .partition = partition_create(k),
        .remaining = (int*)malloc(s->graph->n * sizeof(int)),
        .remaining_count = s->graph->n,
        .current_mod = singleton_mod,
        .depth = 0
    };
    memcpy(initial_work.remaining, s->graph->nodes, s->graph->n * sizeof(int));

    fprintf(stderr, "Starting work-stealing parallel search with %d threads\n", num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        ThreadContext ctx = {
            .solver = s,
            .thread_id = tid,
            .num_threads = num_threads,
            .all_deques = all_deques,
            .my_deque = &all_deques[tid],
            .ub_final_degrees = (double*)malloc(k * sizeof(double)),
            .ub_locked = (int*)malloc(k * sizeof(int)),
            .local_calls = 0,
            .stolen_count = 0,
            .solution_lock = &solution_lock,
            .global_done = &global_done,
            .active_workers = &active_workers
        };

        /* Thread 0 starts with initial work, others wait/steal */
        if (tid == 0) {
            thread_main(&ctx, &initial_work);
        } else {
            thread_main(&ctx, NULL);
        }

        /* Aggregate statistics */
        #pragma omp atomic
        s->recursive_calls += ctx.local_calls;

        #pragma omp critical
        {
            if (ctx.stolen_count > 0) {
                fprintf(stderr, "  Thread %d: %lld calls, %lld stolen\n",
                        tid, (long long)ctx.local_calls, (long long)ctx.stolen_count);
            }
        }

        free(ctx.ub_final_degrees);
        free(ctx.ub_locked);
    }

    fprintf(stderr, "Search complete: %lld recursive calls\n",
            (long long)s->recursive_calls);

    /* Cleanup */
    omp_destroy_lock(&solution_lock);
    for (int t = 0; t < num_threads; t++) {
        deque_free(&all_deques[t]);
    }
    free(all_deques);

#else
    fprintf(stderr, "Error: OpenMP not available\n");
    (void)s; (void)k; (void)num_threads;
#endif
}