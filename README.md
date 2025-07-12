# HDM_Seminar

Code used for the POD-Galerkin project during the high dimensional methods seminar.

To run this code you will need the following packages: pytorch, FEniCS, numpy and matplotlib.

heatFenics solves the heat equation using FEniCS, and thus creates our reference solution.
heat_Galerkin solves the heat equation using (POD-)Galerkin. By changing some if statements, you can either use the standard Galerkin or POD-Galerkin. There is also an explicit and implicit Euler implementation which can also be played around with by changing an if statement in the solve function.

