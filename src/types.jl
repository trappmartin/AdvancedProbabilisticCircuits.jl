export Prod, Sum, Leaf
export AbstractNode, Node, SumNode, ProductNode
export Decomposability, Smoothness, Determinism, Invertability, NodeProperties

abstract type NodeType end
struct ProductNode <: NodeType end
struct SumNode <: NodeType end

abstract type NodeProperties end
struct Decomposability <: NodeProperties end
struct Smoothness <: NodeProperties end
struct Determinism <: NodeProperties end
struct Invertability <: NodeProperties end

abstract type AbstractNode end

struct Node{T<:Number,V<:Union{Vector{Int},Int},N<:NodeType} <: AbstractNode
    scope::V
    children::Vector{AbstractNode}
    params::Vector{T}
    properties::Vector{NodeProperties}
end

function _build_node(n::Type{N}, params::Vector{T}, ns::AbstractNode...) where {T<:Number,N<:NodeType}
    children = collect(ns)
    s_ = reduce(union, scope.(children))
    scope_ = length(s_) == 1 ? first(s_) : s_
    prop_ = NodeProperties[]

    check_smooth(children) && push!(prop_, Smoothness())
    check_decomposable(children) && push!(prop_, Decomposability())

    return Node{T,typeof(scope_),N}(scope_, children, params, prop_)
end

Prod(ns::AbstractNode...) = Prod(Float32, ns...)
function Prod(t::Type{T}, ns::AbstractNode...) where {T<:Number}
    return _build_node(ProductNode, T[], ns...)
end
(n::Node{T,V,ProductNode})(x::AbstractVector) where {T<:Number,V} = sum(c(x) for c in n.children)
(n::Node{T,V,ProductNode})(x::AbstractMatrix) where {T<:Number,V} = sum(c(x) for c in n.children)

Sum(ns::AbstractNode...) = Sum(Float32, ns...)
function Sum(::Type{T}, ns::AbstractNode...; init = (y) -> log.(rand(Dirichlet(length(y), 1.0)))) where {T<:Number}
    return _build_node(SumNode, T.(init(ns)), ns...)
end
(n::Node{T,V,SumNode})(x::AbstractVector) where {T<:Number,V} = logsumexp(n.params[i]+c(x) for (i,c) in enumerate(n.children))
(n::Node{T,V,SumNode})(x::AbstractMatrix) where {T<:Number,V} = logsumexp(mapreduce(((w,c),) -> w.+c(x), hcat, zip(n.params, n.children)), dims=2)

struct Leaf{T<:Union{Vector{Int},Int},D<:Distribution} <: AbstractNode
    scope::T
    dist::D
end
Leaf(dist::UnivariateDistribution, scope::Int) = Leaf(scope, dist)
Leaf(dist::UnivariateDistribution, scope...) = Leaf(unique(scope), Product([dist for _ in unique(scope)]))
Leaf(dist::Distribution, scope...) = Leaf(unique(scope), dist)
Leaf(dist::Distribution, scope::AbstractRange) = Leaf(dist, collect(scope)...)

(n::Leaf{Int,<:UnivariateDistribution})(x::T) where {T<:AbstractVector} = logpdf(n.dist, scope(n))
function (n::Leaf{Int,<:UnivariateDistribution})(x::T) where {T<:AbstractArray}
    scope(n) > size(x,1) && throw(ErrorException())
    return logpdf(n.dist, view(x,scope(n),:))
end

function (n::Leaf{Vector{Int},<:MultivariateDistribution})(x::T) where {T<:AbstractArray}
    maximum(scope(n)) > size(x,1) && throw(ErrorException())
    return logpdf(n.dist, view(x,scope(n),:))
end

scope(n::AbstractNode) = n.scope
function check_decomposable(ns::AbstractVector{<:AbstractNode})
    d = true
    for i in eachindex(ns)
        l = ns[i]
        scope_l = scope(l)
        for j in (i+1):length(ns)
            r = ns[j]
            d &= isdisjoint(scope_l, scope(r))
        end
    end
    return d
end

function check_smooth(ns::AbstractVector{<:AbstractNode})
    s = true
    n1 = first(ns)
    for ni in ns[2:end]
        s &= scope(ni) == scope(n1)
    end
    return s
end

isdecomposable(n::Node{T,V,ProductNode}) where {T,V} = Decomposability() ∈ n.properties
isdecomposable(n::AbstractNode) = false

issmooth(n::Node{T,V,SumNode}) where {T,V} = Smoothness() ∈ n.properties
issmooth(n::AbstractNode) = false

isdeterministic(n::Node{T,V,SumNode}) where {T,V} = Determinism() ∈ n.properties
isdeterministic(n::AbstractNode) = false

export isdecomposable, issmooth, isdeterministic
