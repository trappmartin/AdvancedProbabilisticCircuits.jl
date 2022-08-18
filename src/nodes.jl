"""
    _build_node

Function to construct an internal node of a probabilistic circuit.
"""
function _build_node(::Type{N}, params::T, ns::AbstractNode...) where {T<:AbstractVector{<:Real},N<:NodeType}
    children = collect(ns)
    s_ = reduce(union, scope.(children))
    scope_ = length(s_) == 1 ? first(s_) : s_
    prop_ = NodeProperties[]

    check_smooth(children) && push!(prop_, Smoothness())
    check_decomposable(children) && push!(prop_, Decomposability())
    check_determinism(children) && push!(prop_, Determinism())
    supports_ = N == SumNode ? Dict(reduce(joinsupport, children)) : Dict(reduce(vcat, support.(children)))

    support_ = scope_ isa Int ? supports_[scope_] : [supports_[s] for s in scope_]

    return Node{typeof(scope_),typeof(support_),T,typeof((prop_...,)),N}(scope_, support_, children, params, (prop_...,))
end

function _build_node(::Type{N}, params::T, child::AbstractNode) where {T<:AbstractVector{<:Real},N<:NodeType}
    s_ = scope(child)
    scope_ = length(s_) == 1 ? first(s_) : s_
    prop_ = NodeProperties[]

    push!(prop_, Smoothness())
    push!(prop_, Decomposability())
    push!(prop_, Determinism())
    supports_ = Dict(support(child))

    support_ = scope_ isa Int ? supports_[scope_] : [supports_[s] for s in scope_]

    return Node{typeof(scope_),typeof(support_),T,typeof((prop_...,)),N}(scope_, support_, [child], params, (prop_...,))
end

# --
# Default Internal Nodes
# --

# product node
Prod(t::Type{<:AbstractNode}, partition::Vector) = Prod([t(s) for s in partition]...)
Prod(f::Function, partition::Vector) = Prod([f(s) for s in partition]...)
Prod(ns::AbstractNode...) = Prod(Float32, ns...)
function Prod(::Type{T}, ns::AbstractNode...) where {T<:Real}
    nodes = mapreduce(n -> isproduct(n) ? children(n) : n, vcat, ns)
    return _build_node(ProductNode, T[], nodes...)
end

isproduct(n::Node{V,S,T,P,ProductNode}) where {V,T,S,P} = true
isproduct(n::Node{V,S,T,P,N}) where {V,T,S,P,N} = false
isproduct(n::T) where {T<:AbstractLeaf} = false

logpdf(n::Node{T,V,S,P,ProductNode}, x) where {T,V,S,P} = sum(logpdf.(children(n),Ref(x)))

# sum node
Sum(t::Type{<:AbstractNode}, scope, K::Int) = Sum([t(scope) for k in 1:K]...)
Sum(f::Function, K::Int) = Sum([f(k) for k in 1:K]...)

Sum(ns::AbstractNode...) = Sum(Float32, ns...)
Sum(n::AbstractNode) = Sum(Float32, n)
function Sum(::Type{T}, ns::AbstractNode...; init = (y) -> log.(rand(length(y)))) where {T<:Real}
    return _build_node(SumNode, T.(init(ns)), ns...)
end

"""
    _fast_logpdf(n, x)

More efficient evaluation of a sum node using log(sum(exp(x + log(w)))) based on:
[Sebastian Nowozin: Streaming Log-sum-exp Computation.](http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html)
"""
function _fast_logpdf(n::Node{T,V,S,P,SumNode}, x) where {T,V,S,P} 
    cs = children(n)
    xmax_r = mapreduce(i -> logpdf(cs[i], x) .+ n.params[i], _logsumexp_onepass_op, 1:length(cs))
    return first(xmax_r) .+ log1p.(last(xmax_r))
end
function logpdf(n::Node{T,V,S,P,SumNode}, x) where {T,V,S,P} 
    return logsumexp(n.params[i]+logpdf(c,x) for (i,c) in enumerate(n.children))
end
logpdf(n::Node{T,V,S,P,SumNode}, x::AbstractMatrix) where {T,V,S,P} = _fast_logpdf(n,x)

# --
# Default leaf nodes
# --

logpdfnormal(μ, σ, x) = -log(σ) - (log2π + ((x-μ) / σ)^2)/2

@leaf Normal(μ=0.0, σ=1.0) RealInterval(-Inf, Inf)
logpdf(n::Normal{P}, x::Real) where {P<:NamedTuple{(:μ, :σ)}} = logpdfnormal(n.params.μ, n.params.σ, x)

@leaf Indicator(v = 1)
logpdf(n::Indicator{P}, x::Real) where {P<:Real} = n.params == x ? 0 : -Inf
support(n::Indicator{P}) where {P<:Real} = (n.scope => [n.params])

@leaf TruncatedNormal(μ = 0.0, σ = 1.0, min = -Inf, max = Inf)
function logpdf(n::TruncatedNormal{P}, x::Real) where {P<:NamedTuple{(:μ, :σ, :min, :max)}}
    return (x >= n.params.min) && (x < n.params.max) ? logpdfnormal(n.params.μ, n.params.σ, x) : -Inf
end
function support(n::TruncatedNormal{P}) where {P<:NamedTuple{(:μ, :σ, :min, :max)}}
    return (n.scope => RealInterval(n.params.min, n.params.max))
end

# --
# Helper functions
# --

function descendants(n::Node{T,V,S,P,N}, N2::Type{<:NodeType}; level = Inf) where {T,V,S,P,N}
    l = N == N2 ? level-1 : level
    nodes = l > 0 ? mapreduce(c -> descendants(c, N2), vcat, n.children) : []
    N == N2 && push!(nodes, n)
    return unique(nodes)
end
descendants(::AbstractLeaf, ::Type{<:NodeType}; level = Inf) = []

leaves(n::Node) = unique(mapreduce(c -> leaves(c), vcat, n.children))
leaves(n::AbstractLeaf) = [n]
