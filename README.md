# ProbabilisticCircuits
[![Build Status](https://travis-ci.org/trappmartin/ProbabilisticCircuits.jl.svg?branch=master)](https://travis-ci.org/trappmartin/ProbabilisticCircuits.jl)
[![codecov](https://codecov.io/gh/trappmartin/ProbabilisticCircuits.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/trappmartin/ProbabilisticCircuits.jl)

`ProbabilisticCircuits.jl` is a package for building probabilistic circuits and performing probabilistic reasoning.

Warning: This package is under heavy development and things might easily change!

## Installation
This package is currently not registered. To install `ProbabilisticCircuits.jl` either clone the package manually or add it using the Julia package manage using: `Pkg.add(url="https://github.com/trappmartin/ProbabilisticCircuits.jl")`.

## Getting Started

The following example illustrates the construction of a probabilistic circuit and its use for density estimation.

```julia
using ProbabilisticCircuits, MLDatasets

# download Iris dataset if necessary
# Iris.download()

X = Iris.features()

# manually define a simple probabilistic circuit

pc = Sum( 
        Prod( 
            Sum(Leaf(truncated(Normal(), 0, Inf), 1:2), Leaf(truncated(Normal(), 0, Inf), 1:2)), 
            Leaf(truncated(Normal(), 0, Inf), 3:4)), 
        Leaf(truncated(Normal(), 0, Inf), 1:4))
```
Once defined the probabilistic circuit, you should see the respective circuit in a more amendable form displayed in the REPL:

```julia
(+) (
0.902 × (x) (
    (+) (
    0.228 ×     Truncated{Normal{Float64},Continuous,Float64} - scope: [1, 2],
    0.772 ×     Truncated{Normal{Float64},Continuous,Float64} - scope: [1, 2]),
    Truncated{Normal{Float64},Continuous,Float64} - scope: [3, 4]),
0.098 × Truncated{Normal{Float64},Continuous,Float64} - scope: [1, 2, 3, 4])
```
Note that the colour coding of the nodes, e.g. `(+)` and `(x)`, indicate the nodes properties.

We can evaluate the log density of the dataset by calling the circuit on the data, i.e.

```julia
using Statistics

llh(model, x) = mean(model(x[:,i]) - partitionfunction(model) for i in 1:size(x,2))
@show llh(pc, X)
```
Alternatively we can also call: `pc(X)` to evaluate all observations at once.

We can optimise the paramters using Zygote as follows:

```julia
using Zygote, UnicodePlots

iters = 20
values = zeros(iters)

for i in 1:iters
    grads = Zygote.gradient(m -> llh(m, X), pc)[1]
    update!(pc, grads; η = 1.0)
    values[i] = llh(pc, X)
end

lineplot(values)
```
which should result in a result similar to:

```julia
         ┌────────────────────────────────────────┐
   -33.7 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⠤⠤⠤⠤⠤⠔⠒⠒⠒⠒⠒⠒⠒│
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠔⠒⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠀⠀⢀⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠀⢀⡎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⡠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⡠⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
   -34.6 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         └────────────────────────────────────────┘
         0                                       20

```

Note: This approach will currently only update internal node parameters.
