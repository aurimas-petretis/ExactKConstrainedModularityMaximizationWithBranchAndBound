#include "../include/bnb_types.h"
#include <string.h>

#define INITIAL_COMMUNITY_CAPACITY 16

static Community community_create(void) {
    Community c;
    c.members = (int*)malloc(INITIAL_COMMUNITY_CAPACITY * sizeof(int));
    c.size = 0;
    c.capacity = INITIAL_COMMUNITY_CAPACITY;
    return c;
}

static void community_free(Community* c) {
    free(c->members);
    c->members = NULL;
    c->size = 0;
    c->capacity = 0;
}

static void community_push(Community* c, int node) {
    if (c->size >= c->capacity) {
        c->capacity *= 2;
        c->members = (int*)realloc(c->members, c->capacity * sizeof(int));
    }
    c->members[c->size++] = node;
}

static void community_pop(Community* c) {
    if (c->size > 0) {
        c->size--;
    }
}

static Community community_copy(const Community* c) {
    Community copy;
    copy.capacity = c->capacity;
    copy.size = c->size;
    copy.members = (int*)malloc(copy.capacity * sizeof(int));
    memcpy(copy.members, c->members, c->size * sizeof(int));
    return copy;
}

Partition* partition_create(int capacity) {
    Partition* p = (Partition*)malloc(sizeof(Partition));
    p->communities = (Community*)malloc(capacity * sizeof(Community));
    p->num_communities = 0;
    p->capacity = capacity;
    return p;
}

void partition_free(Partition* p) {
    if (p) {
        for (int i = 0; i < p->num_communities; i++) {
            community_free(&p->communities[i]);
        }
        free(p->communities);
        free(p);
    }
}

Partition* partition_copy(const Partition* p) {
    Partition* copy = partition_create(p->capacity);
    copy->num_communities = p->num_communities;
    for (int i = 0; i < p->num_communities; i++) {
        copy->communities[i] = community_copy(&p->communities[i]);
    }
    return copy;
}

void partition_add_community(Partition* p) {
    if (p->num_communities >= p->capacity) {
        p->capacity *= 2;
        p->communities = (Community*)realloc(p->communities, p->capacity * sizeof(Community));
    }
    p->communities[p->num_communities++] = community_create();
}

void partition_pop_community(Partition* p) {
    if (p->num_communities > 0) {
        community_free(&p->communities[p->num_communities - 1]);
        p->num_communities--;
    }
}

void partition_push_node(Partition* p, int comm_idx, int node) {
    community_push(&p->communities[comm_idx], node);
}

void partition_pop_node(Partition* p, int comm_idx) {
    community_pop(&p->communities[comm_idx]);
}

void partition_add_singleton(Partition* p, int node) {
    partition_add_community(p);
    partition_push_node(p, p->num_communities - 1, node);
}