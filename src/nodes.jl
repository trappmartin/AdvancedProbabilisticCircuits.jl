"""
    _build_node

Internal function to construct an internal node of a probabilistic circuit.
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

# --
# Default Internal Nodes
# --

Prod(t::Type{<:AbstractNode}, partition::Vector) = Prod([t(s) for s in partition]...)
Prod(f::Function, partition::Vector) = Prod([f(s) for s in partition]...)

Prod(ns::AbstractNode...) = Prod(Float32, ns...)
function Prod(::Type{T}, ns::AbstractNode...) where {T<:Number}
    nodes = mapreduce(n -> isproduct(n) ? children(n) : n, vcat, ns)
    return _build_node(ProductNode, T[], nodes...)
end
logpdf(n::Node{T,V,S,P,ProductNode}, x) where {T,V,S,P} = sum(logpdf(c,x) for c in n.children)
(n::Node{T,V,S,P,ProductNode})(x) where {T,V,S,P} = logpdf(n,x)

partitionfunction(n::Node{V,S,T,P,ProductNode}) where {T,V,S,P} = sum(partitionfunction(c) for c in n.children)

Sum(t::Type{<:AbstractNode}, scope, K::Int) = Sum([t(scope) for k in 1:K]...)
Sum(f::Function, K::Int) = Sum([f(k) for k in 1:K]...)

Sum(ns::AbstractNode...) = Sum(Float32, ns...)
function Sum(::Type{T}, ns::AbstractNode...; init = (y) -> log.(rand(length(y)))) where {T<:Number}
    return _build_node(SumNode, T.(init(ns)), ns...)
end
function logpdf(n::Node{T,V,S,P,SumNode}, x) where {T,V,S,P} 
    return logsumexp(n.params[i]+logpdf(c,x) for (i,c) in enumerate(n.children))
end
function logpdf(n::Node{T,V,S,P,SumNode}, x::AbstractMatrix) where {T,V,S,P} 
    return map(j -> logsumexp(n.params[i]+logpdf(c,view(x,:,j)) for (i,c) in enumerate(n.children)), 1:size(x,2))
end
(n::Node{T,V,S,P,SumNode})(x) where {T,V,S,P} = logpdf(n, x)

partitionfunction(n::Node{T,V,S,P,SumNode}) where {T,V,S,P} = logsumexp(n.params[i]+partitionfunction(c) for (i,c) in enumerate(n.children))

# --
# Default leaf nodes
# --

logpdf(n::T, x::AbstractVector{<:Real}) where {T<:AbstractLeaf} = logpdf(n, x[n.scope])
logpdf(n::T, x::AbstractMatrix{<:Real}) where {T<:AbstractLeaf} = @inbounds logpdf.(Ref(n), view(x,n.scope,:))

@leaf Normal(μ=0.0, σ=1.0)
logpdf(n::Normal{P}, x::Real) where {P<:NamedTuple{(:μ, :σ)}} = -log(n.params.σ) - (x-n.params.μ)^2 / (2 * n.params.σ^2)
support(n::Normal) = (n.scope => RealInterval(-Inf, Inf))

@leaf Indicator(v = 1)
logpdf(n::Indicator{P}, x::Real) where {P<:NamedTuple{(:v,)}} = n.params.v == x ? 0 : -Inf
support(n::Indicator{P}) where {P<:NamedTuple{(:v,)}} = (n.scope => [n.params.v])

@leaf TruncatedNormal(μ = 0.0, σ = 1.0, min = -Inf, max = Inf)
function logpdf(n::TruncatedNormal{P}, x::Real) where {P<:NamedTuple{(:μ, :σ, :min, :max)}}
    return (x >= n.params.min) && (x < n.params.max) ? -log(n.params.σ) - (x-n.params.μ)^2 / (2 * n.params.σ^2) : -Inf
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
