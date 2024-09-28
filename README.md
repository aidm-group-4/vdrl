# Verifiable Compositional Reinforcement Learning
TODO: put a cool badge here :D

Final Project
TU Delft course CS4210-B Intelligent Decision-Making Project

##### Contributors:
> Kenzo Boudier \
  Milan de Koning \
  Thijs Penning \
  Tyler Olson \
  Adit Whorra

##### Requirements:
* python >= 3.10
* numpy
* gurobipy
* pydantic

If using an emperical model (like the example from the reference paper) also requires:
* torch
* stable_bselines3
* minigrid

If using a verifiable model such as VIPER, also requires:
* scikit-learn


##### Last Updated:
May 26th, 2024


### Description
This repository contains an implementation of the method from this [paper](https://arxiv.org/abs/2309.06420).
We have reorganized [their code](https://github.com/cyrusneary/verifiable-compositional-rl)
and added the ability to use subsystems with formal guarantees.


### Quick start
First, complete the [Setup Instructions](#setup-instructions).

To run the minigrid labyrinth test world use:
```bash
examples/run_minigrid_labyrinth.py
```

The `examples` directory contains some other examples to try.


## Setup Instructions
Note: These setup instructions have been tested on Ubuntu 22.04 LTS using an
Nvidia RTX 3070 Ti Laptop edition. On other platforms your mileage may vary.

We recommend installing this project in a python virtual environment using the
standard `venv` python package.
1. Create the virtual environment and activate it.
    ```bash
    python -m venv <virtual-environment>
    ./<virtual-environment>/bin/activate
    ```
1. Make sure `pip` is up to date.
    ```bash
    pip install --upgrade pip
    ```

Install the necessary packages with:
```bash
pip install -r requirements.txt
```

We use the Gurobi Optimizer MILP solver which you must have a valid license to use.
As of the tie of writing, you can request a free Academic Gurobi license [here](https://www.gurobi.com/downloads/end-user-license-agreement-academic/).
You can follow their instructions [here](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer)
to install it.


## Contributing
We have a pre-commit set up for static code formatting and code quality
validation. Please install pre-commit by running `pip install pre-commit` and
run `pre-commit install` to set up the pre-commit on your machine.

For development the package can be installed as editable (the default if folling the installation instructions):
```bash
pip install -e .
```
