# linearizing-uncertainty-for-control
This code can be used to reproduce the results in our paper ``Extended Kalman filter---Koopman operator for tractable stochastic optimal control'', the preprint can be found at [https://arxiv.org/pdf/2402.18554](https://arxiv.org/pdf/2402.18554). It is written using the Julia language, which can be installed from here: [https://julialang.org/](https://julialang.org/), or paste this in your terminal (mac/linux) ```curl -fsSL https://install.julialang.org | sh```

If you're new to Julia, upon cloning the code, open a terminal within the corresponding directory, then visit the quick tutorial here [https://towardsdatascience.com/how-to-setup-project-environments-in-julia-ec8ae73afe9c](https://towardsdatascience.com/how-to-setup-project-environments-in-julia-ec8ae73afe9c).

## example.jl
Serves as the main file, and reproduces the results found in our paper above. It calls the eKF module, loads the example model, collects data, runs DMD, find the corresponding LQR control, and then run a closed-loop simulation.

## model4example.jl
Contains the definition of the dynamic model in the numerical example used in our paper.

## plotting_paper.jl
Solely used for generating figures.

## src/eKF.jl
Contains our eKF module, which runs the extended Kalman filter, using the automatic differentiation package Zygote. (It is a one-step ahead predictor extended Kalman filter).

## Other codes and modules
The remaining codes and modules play auxiliary roles: half-vectorizing the Cholesky factor, concatenating vectors, running simulations, ...etc.

