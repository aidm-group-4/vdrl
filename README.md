# Verifiable Compositional Reinforcement Learning

This repository builds upon [Verifiable Reinforcement Learning Systems via Compositionality](https://arxiv.org/abs/2309.06420) (Neary et al.) by replacing the empirical verification of the subsystems with formally verifiable subsystems.

As the use of Reinforcement Learning (RL) becomes more prevalent in society, it is crucial to verify the safety of these systems, especially in safety-critical domains like healthcare and autonomous vehicles where mistakes can lead to tragic consequences. These systems are usually empirically validated, but for some of these more critical problems, a stronger form of verification is required. Neary et al. propose a method for solving RL problems more eﬀiciently by decomposing an RL task into multiple smaller and simpler sub-tasks that can be solved and verified independently. They argue that hierarchically training an RL system in this manner allows for sub-tasks in an environment to be reordered or swapped in and out without having to retrain the entire system allowing for its reuse in similar but different environments with minimal retraining. However, they use an empirical method for measuring subsystem success probabilities rather than formal methods, limiting the applications to domains where statistical guarantees are acceptable.

The goal of this research is to provide formal guarantees on the subsystems of the decomposed method. We propose a modification to the algorithm provided by Neary et al., where each subsystem is formally verified instead of the current empirical validation method. We also expand their definition of success from a reachability specification to a reach-avoid specification; however, we limit its application to fully observable environments with a known transition function. Our verification algorithm simulates each pos- sible action at a given state to calculate the entire state distribution over a finite time horizon using knowledge of the transition function. We prove the correctness of our algorithm and show that the calculated lower bound on the success rate is tightly coupled to the actual success rate. Furthermore, we use experimental results to show that with under-approximation of this lower bound on success rate, the time and space requirements of the verification algorithm can be improved without significantly reducing verification accuracy.

The final report of our project can be found [here](https://drive.google.com/file/d/1X_vHpSyG6YaZBX4wLncKG7x1mB0Ge0j9/view?usp=sharing).

## Quick start
First, complete the [Setup Instructions](#setup-instructions).

To run the minigrid labyrinth test world use:
```bash
examples/run_minigrid_labyrinth.py
```

### Setup Instructions
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


### Contributing
We have a pre-commit set up for static code formatting and code quality
validation. Please install pre-commit by running `pip install pre-commit` and
run `pre-commit install` to set up the pre-commit on your machine.

For development the package can be installed as editable (the default if folling the installation instructions):
```bash
pip install -e .
```


##### Contributors:
> Kenzo Boudier \
  Milan de Koning \
  Thijs Penning \
  Tyler Olson \
  Adit Whorra

##### Supervisors
> Sterre Lutz (PhD) \
  Dr. Anna Lukina


