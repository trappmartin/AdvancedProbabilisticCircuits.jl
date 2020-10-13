"""
    joinsupport(n, n)

Join the support of two nodes.
"""
function joinsupport(l1::AbstractNode, l2::AbstractNode)
    if l1.scope != l2.scope
        return joinsupport_(support(l1), support(l2))
    else
        return join_(support(l1), support(l2))
    end
end

function joinsupport(l1::Pair{Int,T}, l2::AbstractNode) where {T}
    scope, _ = l1
    if scope != l2.scope
        return joinsupport_(l1, support(l2))
    else
        return join_(l1, support(l2))
    end
end

joinsupport_(l1::Pair{Int,T1}, l2::Pair{Int,T2}) where {T1,T2} = [l1, l2]
joinsupport_(l1::Vector{Pair{Int,T1}}, l2::Pair{Int,T2}) where {T1,T2} = joinsupport_(l2,l1)
function joinsupport_(l1::Pair{Int,T1}, supp::Vector{Pair{Int,T2}}) where {T1,T2}
    scope1, supp1 = l1
    scope = [first(s) for s in supp]
    if scope1 ∈ scope
        i = findfirst(scope1 .== scope)
        supp[i] = join_(l1, supp[i])
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
            supp[j] = join_(supp1[i], supp[j])
        else
            push!(supp, supp1[i])
        end
    end
    return supp
end

function join_(p1::Pair{Int,T1}, p2::Vector{Pair{Int,T2}}) where {T1,T2}
    l1,r1 = p1
    p = copy(p2)
    scope = [first(s) for s in p2]
    j = findfirst(l1 .== scope)
    j === nothing && throw(ErrorException("Unexpected error"))
    p[j] = join_(p1, p2[j])
    return p
end
function join_(p1::Vector{Pair{Int,T1}}, p2::Vector{Pair{Int,T2}}) where {T1,T2}
    p = copy(p2)
    for i in 1:length(p1)
        l1 = p1[i]
        p = join_(l1, p2)
    end
    return p
end
join_(p1::Pair{Int,Vector{Int}}, p2::Pair{Int,Vector{Int}}) = first(p1) => union(last(p1), last(p2))
function join_(p1::Pair{Int,RealInterval{T1}}, p2::Pair{Int,RealInterval{T2}}) where {T1,T2}
    l1,r1 = p1
    l2,r2 = p2
    if r1 == r2
        return l1 => r1
    elseif isapprox(r1.ub, r2.lb, atol=eps())
        return l1 => RealInterval(r1.lb, r2.ub)
    elseif isapprox(r2.ub, r1.lb, atol=eps())
        return l1 => RealInterval(r2.lb, r1.ub)
    elseif !isdisjointsupport(r1, r2)
        return l1 => RealInterval(min(r1.lb, r2.lb), max(r1.ub, r2.ub))
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
function isdisjointsupport(r1::RealInterval, r2::RealInterval)
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