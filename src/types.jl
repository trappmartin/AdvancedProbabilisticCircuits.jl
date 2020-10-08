export isdecomposable, issmooth, isdeterministic
export scope, support, joinsupport, isdisjointsupport
export partitionfunction

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

struct Node{T<:AbstractVector{<:Number},V<:Union{Vector{Int},Int},S<:Union{Vector,Distributions.RealInterval},N<:NodeType} <: AbstractNode
    scope::V
    support::S
    children::Vector{AbstractNode}
    params::T
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

function _build_node(n::Type{N}, params::T, ns::AbstractNode...) where {T<:AbstractVector{<:Number},N<:NodeType}
    children = collect(ns)
    s_ = reduce(union, scope.(children))
    scope_ = length(s_) == 1 ? first(s_) : s_
    prop_ = NodeProperties[]

    check_smooth(children) && push!(prop_, Smoothness())
    check_decomposable(children) && push!(prop_, Decomposability())
    check_determinism(children) && push!(prop_, Determinism())
    supports_ = N == SumNode ? Dict(reduce(joinsupport, children)) : Dict(reduce(vcat, support.(children)))

    support_ = scope_ isa Int ? supports_[1] : [supports_[s] for s in scope_]

    return Node{T,typeof(scope_),typeof(support_),N}(scope_, support_, children, params, prop_)
end

Prod(ns::AbstractNode...) = Prod(Float32, ns...)
function Prod(t::Type{T}, ns::AbstractNode...) where {T<:Number}
    return _build_node(ProductNode, T[], ns...)
end
(n::Node{T,V,S,ProductNode})(x::AbstractVector) where {T,V,S} = sum(c(x) for c in n.children)
(n::Node{T,V,S,ProductNode})(x::AbstractMatrix) where {T,V,S} = sum(c(x) for c in n.children)

partitionfunction(n::Node{T,V,S,ProductNode}) where {T,V,S} = sum(partitionfunction(c) for c in n.children)

Sum(ns::AbstractNode...) = Sum(Float32, ns...)
function Sum(::Type{T}, ns::AbstractNode...; init = (y) -> log.(rand(Dirichlet(length(y), 1.0)))) where {T<:Number}
    return _build_node(SumNode, T.(init(ns)), ns...)
end
(n::Node{T,V,S,SumNode})(x::AbstractVector) where {T,V,S} = logsumexp(n.params[i]+c(x) for (i,c) in enumerate(n.children))
(n::Node{T,V,S,SumNode})(x::AbstractMatrix) where {T,V,S} = logsumexp(mapreduce(((w,c),) -> w.+c(x), hcat, zip(n.params, n.children)), dims=2)

partitionfunction(n::Node{T,V,S,SumNode}) where {T,V,S} = logsumexp(n.params[i]+partitionfunction(c) for (i,c) in enumerate(n.children))

# --
# -- Leaf nodes --
# --

struct Leaf{V<:Union{Vector{Int},Int},D<:UnivariateDistribution,P<:AbstractVector{<:Real}} <: AbstractNode
    scope::V
    dist::D
    params::P
end
Leaf(dist::UnivariateDistribution, scope::Int) = Leaf(scope, dist, collect(params(dist)))
Leaf(dist::UnivariateDistribution, scope...) = Leaf(unique(scope), dist, repeat(collect(params(dist)), length(unique(scope))))
Leaf(dist::UnivariateDistribution, scope::AbstractRange) = Leaf(dist, collect(scope)...)

function prettyprint(io::IO, node::Leaf{T,D}, level::Int) where {T,D}
    print(io, string(repeat('\t', level), D, " - scope: $(node.scope)"))
end
descendants(n::Leaf, N2; level = Inf) = []

evaldist(dist::D, params, x::T) where {D<:UnivariateDistribution,T<:Real} = T(logpdf(convert(D, params...), x))
evaldist(dist::D, params, x::Missing) where {D<:UnivariateDistribution} = one(eltype(D))

(n::Leaf{Int,<:UnivariateDistribution,P})(x::T) where {T<:AbstractVector,V<:Real,P} = evaldist(n.dist, n.params, x[scope(n)])
function (n::Leaf{Int,D,P})(x::AbstractArray{T}) where {T<:Real,P,D<:UnivariateDistribution}
    scope(n) > size(x,1) && throw(ErrorException())
    return @inbounds logpdf(convert(D, n.params...), view(x,scope(n),:))
end

#evaldist(dist::T, params, x::D) where {T<:UnivariateDistribution,D<:Number} = logpdf(convert(T, params...), x)
#evaldist(dist::T, params, x::Missing) where {T<:UnivariateDistribution} = one(eltype(dist))

(n::Leaf{Vector{Int},<:UnivariateDistribution,P})(x::T) where {T<:AbstractVector,P} = sum(logpdf(n.dist, x[s]) for s in scope(n))
function (n::Leaf{Vector{Int},<:UnivariateDistribution,P})(x::T) where {T<:AbstractArray,P}
    maximum(scope(n)) > size(x,1) && throw(ErrorException())
    return sum(logpdf(n.dist, view(x,s,:)) for s in scope(n))
end

partitionfunction(l::Leaf{V,D,P}) where {V,D,P<:AbstractVector} = one(eltype(P))

scope(n::AbstractNode) = n.scope

# --
# -- Internal functions used to check node properties during construction
# --

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


"""
    isdecomposable(n)

Return true if the node is decomposable and false otherwise.
"""
isdecomposable(n::Node{T,V,S,ProductNode}) where {T,V,S} = Decomposability() ∈ n.properties
isdecomposable(n::AbstractNode) = false

"""
    issmooth(n)

Return true if the node is smooth and false otherwise.
"""
issmooth(n::Node{T,V,S,SumNode}) where {T,V,S} = Smoothness() ∈ n.properties
issmooth(n::AbstractNode) = false

"""
    isdeterministic(n)

Return true if the node is deterministic and false otherwise.
"""
isdeterministic(n::Node{T,V,S,SumNode}) where {T,V,S} = Determinism() ∈ n.properties
isdeterministic(n::AbstractNode) = false

"""
    support(n)

Return the support of the node
"""
support(n::Leaf{Int,D,P}) where {D,P} = n.scope => Distributions.support(n.dist)
support(n::Leaf{<:AbstractVector,D,P}) where {D,P} = [s => Distributions.support(n.dist) for s in n.scope]
support(n::Node{T,Int,S,N}) where {T,S<:Union{Distributions.RealInterval,Vector{Int}},N} = [n.scope => n.support]
support(n::Node{T,<:AbstractVector,S,N}) where {T,S,N} = [s => sup for (s, sup) in zip(n.scope, n.support)]

"""
    joinsupport(n, n)

Join the support of two nodes.
"""
function joinsupport(l1::AbstractNode, l2::AbstractNode)
    if l1.scope != l2.scope
        return joinsupport_(support(l1), support(l2))
    else
        return join(support(l1), support(l2))
    end
end

joinsupport_(l1::Pair{Int,T1}, l2::Pair{Int,T2}) where {T1,T2} = [l1, l2]
joinsupport_(l1::Vector{Pair{Int,T1}}, l2::Pair{Int,T2}) where {T1,T2} = joinsupport_(l2,l1)
function joinsupport_(l1::Pair{Int,T1}, supp::Vector{Pair{Int,T2}}) where {T1,T2}
    scope1, supp1 = l1
    scope = [first(s) for s in supp]
    if scope1 ∈ scope
        i = findfirst(scope1 .== scope)
        supp[i] = join(l1, supp[i])
    else
        push!(supp, l1)
    end
    return supp
end
function joinsupport_(supp1::Vector{Pair{Int,T1}}, supp::Vector{Pair{Int,T2}}) where {T1,T2}
    L = length(supp1)
    scope = [first(s) for s in supp]
    for i in 1:L
        s1, sup1 = supp1[i]
        if s1 ∈ scope
            j = findfirst(s1 .== scope)
            supp[j] = join(supp1[i], supp[j])
        else
            push!(supp, supp1[i])
        end
    end
    return supp
end

function join(p1::Pair{Int,T1}, p2::Vector{Pair{Int,T2}}) where {T1,T2}
    l1,r1 = p1
    p = copy(p2)
    scope = [first(s) for s in p2]
    j = findfirst(l1 .== scope)
    j == nothing && throw(ErrorException("Unexpected error"))
    p[j] = join(p1, p2[j])
    return p
end
function join(p1::Vector{Pair{Int,T1}}, p2::Vector{Pair{Int,T2}}) where {T1,T2}
    p = copy(p2)
    for i in 1:length(p1)
        l1 = p1[i]
        p = join(l1, p2)
    end
    return p
end


join(p1::Pair{Int,Vector{Int}}, p2::Pair{Int,Vector{Int}}) = first(p1) => union(last(p1), last(p2))
function join(p1::Pair{Int,Distributions.RealInterval}, p2::Pair{Int,Distributions.RealInterval})
    l1,r1 = p1
    l2,r2 = p2
    if r1 == r2
        return l1 => r1
    elseif isapprox(r1.ub, r2.lb, atol=eps())
        return l1 => Distributions.RealInterval(r1.lb, r2.ub)
    elseif isapprox(r2.ub, r1.lb, atol=eps())
        return l1 => Distributions.RealInterval(r2.lb, r1.ub)
    elseif !isdisjointsupport(r1, r2)
        return l1 => Distributions.RealInterval(min(r1.lb, r2.lb), max(r1.ub, r2.ub))
    else
        throw(ErrorException("Joining non-adjecent real-supports currently not supported."))
    end
end

isdisjointsupport(l1::Pair{Int,T},l2::Pair{Int,T}) where {T} = isdisjointsupport(l1[2], l2[2])
isdisjointsupport(l1::Vector{Pair{Int,T1}}, l2::Pair{Int,T2}) where {T1,T2} = isdisjointsupport(l2,l1)
function isdisjointsupport(l1::Pair{Int,T1}, l2::Vector{Pair{Int,T2}}) where {T1,T2}
    r = true
    for supp in l2
        s, v = supp
        if s == first(l1)
            r &= isdisjointsupport(last(l1), v)
        end
    end
    return r
end
function isdisjointsupport(l1::Vector{Pair{Int,T1}}, l2::Vector{Pair{Int,T2}}) where {T1,T2}
    r = true
    for supp in l1
        r &= isdisjointsupport(supp, l2)
    end
    return r
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

