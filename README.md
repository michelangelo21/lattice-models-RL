[![DOI](https://zenodo.org/badge/460968633.svg)](https://zenodo.org/badge/latestdoi/460968633)


# Machine Learning-driven improvement of the ground state configuration

This repository implements physical lattice models as `gym` environments to find their ground states with reinforcement learning.

It was created during bachelor thesis project supervised by prof. dr hab. Maciej M. Maśka.

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
The easiest way of running this environment is with `conda` 
(https://conda.io/projects/conda/en/latest/user-guide/install/index.html or 
https://docs.anaconda.com/anaconda/install/index.html).

Firstly create the environment from the `environment.yml` file:
```bash
conda env create -f environment.yml
```
Then activate the new environment
```bash
conda activate latticerl
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


## Abstract
The search for ground states in physical models is a fundamental problem in physics, as it allows for a deeper understanding of the properties and behavior of complex systems. This thesis presents a new method for searching ground states of physical models using deep reinforcement learning. The method utilizes the proximal policy optimization (PPO) algorithm, a widely used algorithm in deep reinforcement learning. It was applied to various classical spin models such as the Ising model, the XY model, and the model with Dzyaloshinskii-Moriya interaction, as well as a simple quantum fermionic spinless Falicov-Kimball model. The results of the experiments show that the method presents a promising approach for finding ground states in physical models, however, it may encounter challenges when dealing with larger lattice sizes. The research provides a new perspective on solving the challenging problem of finding ground states in classical and quantum systems and suggests areas for future research in this field. 

### Keywords
Ground state, Energy minimization, Reinforcement Learning, Deep Reinforcement Learning, Spin Models, Fermionic models, Neural Network, Machine Learning, Markov Decision Process, Proximal Policy Optimization



## Data availability

Data from performed experiments is available here:
https://tensorboard.dev/experiment/F9pMBpqeQvGwr7xayu9lOQ/
