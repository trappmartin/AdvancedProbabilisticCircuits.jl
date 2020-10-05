# ProbabilisticCircuits
[![Build Status](https://travis-ci.org/trappmartin/ProbabilisticCircuits.jl.svg?branch=master)](https://travis-ci.org/trappmartin/ProbabilisticCircuits.jl)
[![codecov](https://codecov.io/gh/trappmartin/ProbabilisticCircuits.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/trappmartin/ProbabilisticCircuits.jl)

`ProbabilisticCircuits.jl` is a package for building probabilistic circuits and performing probabilistic reasoning.

## Example

```julia
using ProbabilisticCircuits
using Statistics

# Create some continuous data
D = 2
N = 1000
x = randn(D,N);

# Create a simple PC 
pc = Sum(Prod(Leaf(Normal(), 1), Leaf(Exponential(), 2)), Leaf(MvNormal(vec(mean(x, dims=2)), cov(x')), 1:2))

# Evaluate the test log likelihood

xtest = randn(D, 100);
llh = pc(xtest)

@show mean(llh)

```
