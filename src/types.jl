export Prod, Sum, Leaf
export AbstractNode, Node, SumNode, ProductNode
export Decomposability, Smoothness, Determinism, Invertability, NodeProperties

abstract type NodeType end
struct ProductNode <: NodeType end
struct SumNode <: NodeType end

printnodetype(::Type{SumNode}) = "(+)"
printnodetype(::Type{ProductNode}) = "(×)"

abstract type NodeProperties end
struct Decomposability <: NodeProperties end
struct Smoothness <: NodeProperties end
struct Determinism <: NodeProperties end
struct Invertability <: NodeProperties end

abstract type AbstractNode end

struct Node{T<:Number,V<:Union{Vector{Int},Int},S<:Union{Vector,Distributions.RealInterval},N<:NodeType} <: AbstractNode
    scope::V
    support::S
    children::Vector{AbstractNode}
    params::Vector{T}
    properties::Vector{NodeProperties}
end

# pretty print
function prettyprint(io::IO, node::Node{T,V,S,<:SumNode}, level::Int) where {T,V,S}
    s = issmooth(node) ? isdeterministic(node) ? NEGATIVE(BLUE_FG("(+)")) : BLUE_FG("(+)") : isdeterministic(node) ? NEGATIVE("(+)") : "(+)"
    println(io, string(repeat('\t', level), s, " ("))
    for (w, child) in zip(node.params[1:end-1], node.children[1:end-1])
        print(io, string(repeat('\t', level), "$(round(exp(w), digits=3)) × "))
        prettyprint(io, child, level)
        println(io, ", ")
    end
    w = node.params[end]
    print(io, string(repeat('\t', level), "$(round(exp(w), digits=3)) × "))
    prettyprint(io, node.children[end], level)
    print(io, string(repeat('\t', level), ")"))
end

function prettyprint(io::IO, node::Node{T,V,S,<:ProductNode}, level::Int) where {T,V,S}
    s = isdecomposable(node) ? RED_FG("(x)") : "(x)"

    println(io, string(repeat('\t', level), s, " ("))

    for child in node.children[1:end-1]
        prettyprint(io, child, level+1)
        println(io, ", ")
    end
    prettyprint(io, node.children[end], level+1)
    print(io, string(repeat('\t', level), ")"))
end

function Base.show(io::IO, node::Node)
    prettyprint(io, node, 0)
end

function descendants(n::Node{T,V,S,N}, N2::Type{<:NodeType}; level = Inf) where {T,V,S,N}
    l = N == N2 ? level-1 : level
    nodes = l > 0 ? mapreduce(c -> descendants(c, N2), vcat, n.children) : []
    N == N2 && push!(nodes, n)
    return unique(nodes)
end

function _build_node(n::Type{N}, params::Vector{T}, ns::AbstractNode...) where {T<:Number,N<:NodeType}
    children = collect(ns)
    s_ = reduce(union, scope.(children))
    scope_ = length(s_) == 1 ? first(s_) : s_
    prop_ = NodeProperties[]

    check_smooth(children) && push!(prop_, Smoothness())
    check_decomposable(children) && push!(prop_, Decomposability())
    check_determinism(children) && push!(prop_, Determinism())
    support_ = N == SumNode ? reduce(joinsupport, children) : reduce(vcat, scope.(children))

    return Node{T,typeof(scope_),typeof(support_),N}(scope_, support_, children, params, prop_)
end

Prod(ns::AbstractNode...) = Prod(Float32, ns...)
function Prod(t::Type{T}, ns::AbstractNode...) where {T<:Number}
    return _build_node(ProductNode, T[], ns...)
end
(n::Node{T,V,S,ProductNode})(x::AbstractVector) where {T<:Number,V,S} = sum(c(x) for c in n.children)
(n::Node{T,V,S,ProductNode})(x::AbstractMatrix) where {T<:Number,V,S} = sum(c(x) for c in n.children)

Sum(ns::AbstractNode...) = Sum(Float32, ns...)
function Sum(::Type{T}, ns::AbstractNode...; init = (y) -> log.(rand(Dirichlet(length(y), 1.0)))) where {T<:Number}
    return _build_node(SumNode, T.(init(ns)), ns...)
end
(n::Node{T,V,S,SumNode})(x::AbstractVector) where {T<:Number,V,S} = logsumexp(n.params[i]+c(x) for (i,c) in enumerate(n.children))
(n::Node{T,V,S,SumNode})(x::AbstractMatrix) where {T<:Number,V,S} = logsumexp(mapreduce(((w,c),) -> w.+c(x), hcat, zip(n.params, n.children)), dims=2)

struct Leaf{T<:Union{Vector{Int},Int},D<:Distribution} <: AbstractNode
    scope::T
    dist::D
end
Leaf(dist::UnivariateDistribution, scope::Int) = Leaf(scope, dist)
Leaf(dist::UnivariateDistribution, scope...) = Leaf(unique(scope), Product([dist for _ in unique(scope)]))
Leaf(dist::Distribution, scope...) = Leaf(unique(scope), dist)
Leaf(dist::Distribution, scope::AbstractRange) = Leaf(dist, collect(scope)...)

function prettyprint(io::IO, node::Leaf{T,D}, level::Int) where {T,D}
    print(io, string(repeat('\t', level), D, " - scope: $(node.scope)"))
end
descendants(n::Leaf, N2; level = Inf) = []

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

function check_determinism(ns::AbstractVector{<:AbstractNode})
    d = true
    for i in eachindex(ns)
        l = ns[i]
        sl = support(l)
        for j in (i+1):length(ns)
            r = ns[j]
            if scope(l) == scope(r)
                d &= isdisjointsupport(sl, support(r))
            else
                d &= false
            end
        end
    end
    return d
end


isdecomposable(n::Node{T,V,S,ProductNode}) where {T,V,S} = Decomposability() ∈ n.properties
isdecomposable(n::AbstractNode) = false

issmooth(n::Node{T,V,S,SumNode}) where {T,V,S} = Smoothness() ∈ n.properties
issmooth(n::AbstractNode) = false

isdeterministic(n::Node{T,V,S,SumNode}) where {T,V,S} = Determinism() ∈ n.properties
isdeterministic(n::AbstractNode) = false

export isdecomposable, issmooth, isdeterministic

support(n::Leaf) = Distributions.support(n.dist)
support(n::Node) = n.support

function joinsupport(l1::Leaf, l2::Leaf)
    if l1.scope != l2.scope
        if l1.scope isa Int || l2.scope isa Int
            return vcat(support(l1), support(l2))
        else
            throw(ErrorException("Joining non-matching scopes currently not supported."))
        end
    end
    return joinsupport(support(l1), support(l2))
end

joinsupport(r1::Vector{Int}, r2::Vector{Int}) = union(r1, r2)
function joinsupport(r1::Distributions.RealInterval, r2::Distributions.RealInterval)
    if r1 == r2
        return r1
    elseif isapprox(r1.ub, r2.lb, atol=eps())
        return Distributions.RealInterval(r1.lb, r2.ub)
    elseif isapprox(r2.ub, r1.lb, atol=eps())
        return Distributions.RealInterval(r2.lb, r1.ub)
    else
        throw(ErrorException("Joining non-adjecent real-supports currently not supported."))
    end
end

isdisjointsupport(r1::Vector{Int}, r2::Vector{Int}) = isdisjoint(r1,r2)
function isdisjointsupport(r1::Distributions.RealInterval, r2::Distributions.RealInterval)
    if (r1.lb == r2.lb) || (r1.lb == r2.ub) || (r1.ub == r2.lb) || (r1.ub == r2.ub)
        return false
    elseif (r1.lb <= r2.lb) && (r1.ub >= r2.lb)
        return false
    elseif (r2.lb <= r1.lb) && (r2.ub >= r1.lb)
        return false
    else
        return true
    end
end

export joinsupport, support
