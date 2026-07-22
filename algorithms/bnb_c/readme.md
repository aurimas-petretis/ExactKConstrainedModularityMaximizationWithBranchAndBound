## C code setup

You need first to build an executable of the program you want to launch with a make script. Make scripts work for all 
Windows/Linux/MacOS operating systems. When executable is build you can run it either directly or through Python script.

There are three different algorithms that are written in C for k-constrained modularity maximization problem:

1. **Branch and bound (exact)**

To compile: `make`.

To run (with example parameters): 

`echo "4           
  0 1 1 0
  1 0 1 1
  1 1 0 1
  0 1 1 0" | ./bnb_solver 2`


2. **Branch and bound parallelized (exact)**

To compile: `make parallel`.

To run (with example parameters): 

`echo "4           
  0 1 1 0
  1 0 1 1
  1 1 0 1
  0 1 1 0" | ./bnb_solver_parallel 2`


3. **Greedy (heuristic)**. It's the Leiden algorithm, but without merging/splitting clusters step (as this would change the 
number of clusters which is unwanted).

To compile: `make leiden`.

To run (with example parameters): 

`echo "4           
  0 1 1 0
  1 0 1 1
  1 1 0 1
  0 1 1 0" | ./leiden_solver 2`