## Installing julia
Install julia using the package manager of your operating system, or follow the instructions [here](https://julialang.org/downloads/).

## Running the simulation
First run a julia shell inside the project directory to instantiate the project environment:
`julia --project=. -ie 'using Pkg; Pkg.instantiate; import SlowGameSim;'`

Once the shell is up, you can run the following example including interactive visualization:
`SlowGameSim.experiment_and_plot(20, 20, 1, 10, 1, 10000, 0)`
