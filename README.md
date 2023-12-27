# AdvancedProbabilisticCircuits.jl
[![Build Status](https://travis-ci.org/trappmartin/AdvancedProbabilisticCircuits.jl.svg?branch=master)](https://travis-ci.org/trappmartin/AdvancedProbabilisticCircuits.jl)
[![codecov](https://codecov.io/gh/trappmartin/AdvancedProbabilisticCircuits.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/trappmartin/AdvancedProbabilisticCircuits.jl)

`AdvancedProbabilisticCircuits.jl` is a package for probabilistic modelling and inference using probabilistic circuits.

Warning: This package is under heavy development and things might easily change!

## Installation
This package is currently not registered. To install `AdvancedProbabilisticCircuits.jl` either clone the package manually or add it using the Julia package manage using: `Pkg.add(url="https://github.com/trappmartin/AdvancedProbabilisticCircuits.jl")`.

## Getting Started

The following example illustrates the construction of a probabilistic circuit and its use for density estimation.

```julia
using AdvancedProbabilisticCircuits, MLDatasets, DataFrames

# download Iris dataset if necessary
# Iris()

X = Iris(as_df=false).features

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
using Zygote # used for AD 
using ProgressMeter, Plots # visualisation of progress & results

iters = 20 # number of iterations
η = 0.1 # learning rate

# results
values = zeros(iters)

@showprogress for i in 1:iters
    grads = Zygote.gradient(m -> llh(m, X), pc)[1]
    AdvancedProbabilisticCircuits.update!(pc, grads; η = η)
    values[i] = llh(pc, X)
end

plot(values)
```

After optimization, the resulting circuit should look similar to:

```julia
(+) (
0.43 × (×) (
        (+) (
        0.339 ×         (×) (
                TruncatedNormal[(μ = 0.2632989915560352, σ = 2.382310506673276, min = 2.220446049250313e-16, max = Inf)] - scope: 1, 
                TruncatedNormal[(μ = 0.18073477432822802, σ = 1.4408260826173522, min = 2.220446049250313e-16, max = Inf)] - scope: 2), 
        0.333 ×         (×) (
                TruncatedNormal[(μ = 0.0, σ = 1.0, min = -Inf, max = 0)] - scope: 1, 
                TruncatedNormal[(μ = 0.0, σ = 1.0, min = -Inf, max = 0)] - scope: 2     )       ), 
        Normal[(μ = 0.1738561801760675, σ = 1.654385423083228)] - scope: 3, 
        Normal[(μ = 0.06158775377035022, σ = 1.0158713531454664)] - scope: 4), 
1.106 × (×) (
        Normal[(μ = 1.1542622026997085, σ = 3.747453345484773)] - scope: 1, 
        Normal[(μ = 1.3191340405400875, σ = 2.0246776684705585)] - scope: 2, 
        Normal[(μ = 1.146697247351322, σ = 2.802238408757326)] - scope: 3, 
        Normal[(μ = 1.0225365827958446, σ = 0.848282907572307)] - scope: 4))
```

## Adding additional leaf nodes
This package provides a few standard leaf nodes, i.e.

```julia
Normal # univariate Gaussian
TruncatedNormal # truncated univariate Gaussian
Indicator # indicator function
```

and we can easily extend the set of leaf nodes using the `@leaf` macro:

```julia
import AdvancedProbabilisticCircuits.support 
using SpecialFunctions # required for logbeta

# define a Beta distribution with default values
@leaf Beta(α = 1.0, β = 1.0) RealInterval(0.0, 1.0)

# define the log density function
function logpdf(n::Beta{P}, x::Real) where {P<:NamedTuple{(:α, :β)}}
    return (n.params.α - 1) * log(x) + (n.params.β - 1) * log1p(-x) - logbeta(n.params.α, n.params.β)
end
```

If necessary, we can also call `@leaf Beta(α = 1.0, β = 1.0)` instead and define the support manually. See `? @leaf` for an example. Note that this package currently only supports univariate leaves.

Now we can use a Beta distribution as a leaf node in a probabilistic circuit, e.g.

```julia
pc = Sum(Beta(1), Beta(1, α = 0.5), Beta(1, α = 0.5, β = 0.5), Normal(1));
```

Note that we may additionally want to define the adjoint for Zygote, if necessary. We refer to the Zygote documentation on this.

## Adding additional internal nodes
The package proved sum and product nodes as internal nodes, but can easily be extended.
For example, one can implement a custom internal node as follows:

```julia
# sub-type NodeTypes to define a new node type
struct CustomProdNode <: NodeType end

# define custom node type constructor
function CPNode(::Type{T}, ns::AbstractNode...) where T<:Real
    parameters = rand(T, length(ns)) # every internal node has parameters (optional)
    return AdvancedProbabilisticCircuits._build_node(CustomProdNode, parameters, ns...)
end

# define custom log density function
logpdf(n::Node{T,V,S,P,CustomProdNode}, x) where {T,V,S,P} = sum(logpdf(children(n),Ref(x)))
```
