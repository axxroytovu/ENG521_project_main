# ENGN 521 Project: Deep Galerkin Method for Solving Engineering PDEs

This code package is a comparison between physics informed feed forward networks (PINNs) and the Deep Galerkin Method (DGM). 

## Navigation

### deep_learning

The `deep_learning` folder contains the core code that generates the DGM and PINN networks. The `deep_network_core.py` file contains class definitions for the various network models, and a base class for the loss function. The `utils.py` file contains a variety of utility functions, such as a Laplacian function and a Curl function which are often utilized as part of differential equations.

### fluid_flow

The `fluid_flow` example shows the generation of a PINN and a DGM for predicting the velocity in a flow field from a coarse set of training data.

### heat_eqn

The `heat_eqn` example contains a few files which utilize PINNs and DGMs to predict the dissipation of heat on a finite plate.

### three_body

The `three_body` example includes a set of code that utilizes PINNs and DGMs to predict the position and velocity of a satellite moving in the constrained Earth-Moon gravitational system.

### example_dgm

The `example_dgm` folder contains the base code from Al-Aradi et al. which inspired the implementation in this repository.

## References

Code is inspired by and built from Al-Aradi, A., Correia, A., Naiff, D., Jardim, G., & Saporito, Y. (2018). Solving nonlinear and high-dimensional partial differential equations via deep learning. https://arxiv.org/abs/1811.08782

PINNs were first proposed by Raissi, M., Perdikaris, P., & Karniadakis, G. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045

The DGM was proposed by Sirignano, J., & Spiliopoulos, K. (2018). DGM: A deep learning algorithm for solving partial differential equations. Journal of Computational Physics, 375, 1339–1364. https://doi.org/10.1016/j.jcp.2018.08.029
