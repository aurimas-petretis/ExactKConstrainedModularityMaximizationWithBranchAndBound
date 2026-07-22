# Project for research on exact k-constrained modularity maximization with branch and bound


## Installation

1. Clone or download the repository.
2. (Optional) Create a virtual environment, either through IDE or with these commands:

`python -m venv venv`

`source venv/bin/activate  # On Windows use: venv\Scripts\activate`

3. Install the required dependencies:

`pip install -r requirements.txt`

4. The test run with all algorithms can be run with this command:

`python main.py`


## C code setup

Go to c directory:

`cd algorithms/bnb_c`

Then follow the steps that are provided in that file. Executables there are necessary to build as they are also called 
from Python program too.


## Setup for other scripts

You need to get a commercial/academic gurobi licence to run gurobi scripts.


## Experiment environment

All performed experiments are in experiments directory. Experiment script files are called *experiment_x_lfr* or 
*experiment_x_abcd* (if they are performed on LFR or ABCD benchmark graphs) or *experiment_x_y* (if performed on other 
graph/graphs). Currently experiment scripts are runnable only in unix like systems (linux/macOS).