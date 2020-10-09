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

# we can create a leaf node by passing the required scope as an argument
l = Normal(1) # Normal with scope = 1

# we can create a product node with a specific partition using
p = Prod(TruncatedNormal(1, min=eps()), TruncatedNormal(2, min=eps()))

# or with the following shorthands 
partition = [1,2]
p = Prod(Normal, partition)
p = Prod((s)->TruncatedNormal(s, min=eps()), partition)

# we can create a sum nodes the same way as a product node, i.e.
s = Sum(TruncatedNormal(1, min=eps()), TruncatedNormal(1, min=eps()))

# or using
K, scope = 2, 1
p = Sum(Normal, scope, K)
s = Sum((k)->Prod((s)->TruncatedNormal(s, min=eps()), partition), K)

# now lets construct a simple circuit

pc = Sum( 
        Prod( 
            Sum(Prod((s)->TruncatedNormal(s, min=eps()), [1,2]), 
                Prod((s)->TruncatedNormal(s, max=0), [1,2])), 
            Prod(Normal, [3,4])), 
        Prod(Normal, [1,2,3,4]))
```

Once the probabilistic circuit is defined, you should see the circuit in a more amendable form displayed in the REPL, e.g.:

```julia
(+) (
0.761 × (×) (
        (+) (
        0.41 ×  (×) (
                TruncatedNormal[(μ = 0.0, σ = 1.0, min = 2.220446049250313e-16, max = Inf)] - scope: 1, 
                TruncatedNormal[(μ = 0.0, σ = 1.0, min = 2.220446049250313e-16, max = Inf)] - scope: 2  ), 
        0.254 ×         (×) (
                TruncatedNormal[(μ = 0.0, σ = 1.0, min = -Inf, max = 0)] - scope: 1, 
                TruncatedNormal[(μ = 0.0, σ = 1.0, min = -Inf, max = 0)] - scope: 2     )       ), 
        Normal[(μ = 0.0, σ = 1.0)] - scope: 3, 
        Normal[(μ = 0.0, σ = 1.0)] - scope: 4), 
0.33 × (×) (
        Normal[(μ = 0.0, σ = 1.0)] - scope: 1, 
        Normal[(μ = 0.0, σ = 1.0)] - scope: 2, 
        Normal[(μ = 0.0, σ = 1.0)] - scope: 3, 
        Normal[(μ = 0.0, σ = 1.0)] - scope: 4))
```

Note that the colour coding of the nodes indicate the nodes properties.
We can evaluate the (unnormalize) log density of the dataset by calling the `logpdf` of the circuit and compute the normalized log likelihood using `loglikelihood`, i.e.

```julia
using Statistics

lp = logpdf(pc, X)

llh(model, x) = mean(loglikelihood(model, x))
@show llh(pc, X)
```

We can optimise the paramters using Zygote as follows:

```julia
using Zygote, UnicodePlots

iters = 20 # number of iterations
η = 0.1 # learning rate

values = zeros(iters)

for i in 1:iters
    grads = Zygote.gradient(m -> llh(m, X), pc)[1]
    update!(pc, grads; η = η)
    values[i] = llh(pc, X)
end

lineplot(values)
```