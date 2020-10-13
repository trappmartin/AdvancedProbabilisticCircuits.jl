module AdvancedProbabilisticCircuits

using MLStyle
using StatsFuns
using Crayons
using Crayons.Box

import Base.show

export Prod, Sum, Normal, TruncatedNormal, Indicator
export Node, SumNode, ProductNode, NodeType
export RealInterval
export Decomposability, Smoothness, Determinism, Invertability, NodeProperties
export support, scope, loglikelihood, params, descendants
export partitionfunction, logpdf, children, leaves


abstract type NodeType end
struct ProductNode <: NodeType end
struct SumNode <: NodeType end

printnodetype(t::Type{<:NodeType}) = string(t)
printnodetype(::Type{SumNode}) = "(+)"
printnodetype(::Type{ProductNode}) = "(×)"

abstract type NodeProperties end
struct Decomposability <: NodeProperties end
struct Smoothness <: NodeProperties end
struct Determinism <: NodeProperties end
struct Invertability <: NodeProperties end

abstract type AbstractNode end
abstract type AbstractLeaf <: AbstractNode end

struct RealInterval{T<:Real}
    lb::T
    ub::T
end
RealInterval(lb::T1, ub::T2) where {T1<:Real,T2<:Real} = RealInterval(lb, T1(ub))

struct Node{V<:Union{Vector{Int},Int},S<:Union{Vector,RealInterval},T<:AbstractVector{<:Real},P<:Tuple,N<:NodeType} <: AbstractNode
    scope::V
    support::S
    children::Vector{AbstractNode} # this might lead to inefficiencies
    params::T
    properties::P
end

children(n::Node) = n.children
scope(n::T) where {T<:AbstractNode} = n.scope
params(n::T) where {T<:AbstractNode} = n.params

partitionfunction(n::T) where {T<:AbstractLeaf} = 1.0
partitionfunction(n::N) where {N<:Node} = logpdf(n, missing)

support(n::Node{Int,S,T,P,N}) where {T,S,P,N} = [n.scope => n.support]
support(n::Node{Vector{Int},S,T,P,N}) where {T,S,P,N} = [s => sup for (s, sup) in zip(n.scope, n.support)]

logpdf(n::T, x::Missing) where {T<:AbstractLeaf} = 1.0
logpdf(n::T, x::AbstractVector{<:Real}) where {T<:AbstractLeaf} = logpdf(n, x[n.scope])
logpdf(n::T, x::AbstractMatrix{<:Real}) where {T<:AbstractLeaf} = @inbounds logpdf.(Ref(n), view(x,n.scope,:))

loglikelihood(n::T, x::X) where {T<:AbstractNode, X<:Real} = logpdf(n, x) - partitionfunction(n)
loglikelihood(n::T, x::X) where {T<:AbstractNode, X<:AbstractArray} = logpdf(n, x) .- partitionfunction(n)

# includes
include("macros.jl")
include("nodes.jl")
include("properties.jl")
include("optimize.jl")
include("util.jl")
include("print.jl")

end
