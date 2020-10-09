export isdecomposable, issmooth, isdeterministic

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
isdecomposable(n::Node{V,S,T,P,ProductNode}) where {V,S,T,P<:Tuple{Decomposability}} = true
isdecomposable(n::AbstractNode) = false

"""
    issmooth(n)

Return true if the node is smooth and false otherwise.
"""
issmooth(n::Node{V,S,T,P,SumNode}) where {V,S,T,P<:Tuple{Smoothness}} = true
issmooth(n::Node{V,S,T,P,SumNode}) where {V,S,T,P<:Tuple{Smoothness,Determinism}} = true
issmooth(n::AbstractNode) = false

"""
    isdeterministic(n)

Return true if the node is deterministic and false otherwise.
"""
isdeterministic(n::Node{V,S,T,P,SumNode}) where {V,S,T,P<:Tuple{Determinism}} = true
isdeterministic(n::Node{V,S,T,P,SumNode}) where {V,S,T,P<:Tuple{Smoothness,Determinism}} = true
isdeterministic(n::AbstractNode) = false
