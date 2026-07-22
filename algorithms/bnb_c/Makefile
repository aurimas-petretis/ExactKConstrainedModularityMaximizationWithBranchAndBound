# Makefile for C Branch-and-Bound k-constrained Modularity Maximization

# 1. OS Detection
ifeq ($(OS),Windows_NT)
    # Windows Settings
    DETECTED_OS := Windows
    # On Windows, we use mkdir without -p or use a shell-agnostic trick
    MKDIR = if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
    RM = rmdir /s /q
    # Standard GCC on Windows handles OpenMP natively
    OMP_FLAGS = -fopenmp
    OMP_LIBS = -fopenmp
else
    # macOS / Linux Settings
    UNAME_S := $(shell uname -s)
    DETECTED_OS := $(UNAME_S)
    MKDIR = mkdir -p $(BUILD_DIR)
    RM = rm -rf $(BUILD_DIR)

    ifeq ($(UNAME_S),Darwin)
        # macOS specific OpenMP (Homebrew)
        LIBOMP_PREFIX = $(shell brew --prefix libomp 2>/dev/null || echo "/opt/homebrew/opt/libomp")
        OMP_FLAGS = -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
        OMP_LIBS = -L$(LIBOMP_PREFIX)/lib -lomp
    else
        # Linux
        OMP_FLAGS = -fopenmp
        OMP_LIBS = -lgomp
    endif
endif

CC = gcc

# Enhanced optimization flags
CFLAGS_BASE = -O3 -Wall -Wextra -std=c99 \
              -march=native \
              -mtune=native \
              -ffast-math \
              -funroll-loops \
              -flto \
              -DNDEBUG

CFLAGS = $(CFLAGS_BASE)
CFLAGS_OMP = $(CFLAGS_BASE) $(OMP_FLAGS)
LDFLAGS = -lm -flto
LDFLAGS_OMP = -lm -flto $(OMP_LIBS)

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

# --- [Source and Object definitions remain the same] ---
SRCS = $(SRC_DIR)/main.c $(SRC_DIR)/graph.c $(SRC_DIR)/partition.c \
       $(SRC_DIR)/incremental_state.c $(SRC_DIR)/solver.c \
       $(SRC_DIR)/steal_parallel_solver.c $(SRC_DIR)/steal_parallel_solver_pruned.c

SRCS_LEIDEN = $(SRC_DIR)/main_leiden.c $(SRC_DIR)/graph.c \
              $(SRC_DIR)/partition.c $(SRC_DIR)/solver.c

OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
OBJS_PAR = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%-omp.o,$(SRCS))
OBJS_LEIDEN = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%-leiden.o,$(SRCS_LEIDEN))

TARGET = bnb_solver
TARGET_PAR = bnb_solver_parallel
TARGET_LEIDEN = leiden_solver

.PHONY: all clean parallel leiden

all: $(BUILD_DIR) $(TARGET)

parallel: $(BUILD_DIR) $(TARGET_PAR)

leiden: $(BUILD_DIR) $(TARGET_LEIDEN)

$(BUILD_DIR):
	$(MKDIR)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_PAR): $(OBJS_PAR)
	$(CC) $(CFLAGS_OMP) -o $@ $^ $(LDFLAGS_OMP)

$(TARGET_LEIDEN): $(OBJS_LEIDEN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(INC_DIR)/bnb_types.h
	$(CC) $(CFLAGS) -I$(INC_DIR) -c -o $@ $<

$(BUILD_DIR)/%-omp.o: $(SRC_DIR)/%.c $(INC_DIR)/bnb_types.h
	$(CC) $(CFLAGS_OMP) -I$(INC_DIR) -c -o $@ $<

$(BUILD_DIR)/%-leiden.o: $(SRC_DIR)/%.c $(INC_DIR)/bnb_types.h
	$(CC) $(CFLAGS) -I$(INC_DIR) -c -o $@ $<

clean:
	$(RM) $(TARGET) $(TARGET_PAR) $(TARGET_LEIDEN)