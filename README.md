# Machine Learning-driven improvement of the ground state configuration

This repository implements physical lattice models as `gym` environments to find their ground states with reinforcement learning.
Specifically the following models are implemented:
- Ising model,
- Falicov-Kimball model,
- XY model,
- model with Dzyaloshinskii-Moriya interaction,

and can be found in `gym-latticemodels/gym_latticemodels/envs/` directory. 

`scripts` contains scripts used to learn specific environment, and also to test them.
Use `paralleize.sh` with specific script to train multiple agents in parallel.

`src` directory contains additional utils such as cosine annealing, custom CNN, and wrapper that returns reward at every step.


## Installation

Create the environment from the `environment.yml` file:
```bash
conda env create -f environment.yml
```
Install `gym-latticemodels`
```bash
pip install -e gym-latticemodels/
```
Install `src` helpers
```bash
pip install -e .
```
Ready to go!



## Data

Data from performed experiment available here:
https://tensorboard.dev/experiment/F9pMBpqeQvGwr7xayu9lOQ/