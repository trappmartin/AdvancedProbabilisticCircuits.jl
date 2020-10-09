export update!

function update!(n::Node, grads::NamedTuple; η = 0.01, uparams = [])
    if !isnothing(grads.params)
        n.params .+= η * grads.params
    end

    for (i, child) in enumerate(n.children)
        if child isa AbstractLeaf
            n.children[i] = update!(child, grads.children[i]; η = η, uparams = uparams)
        else
            if grads.children[i] === nothing # why is this happening?
                nothing
            else
                update!(child, grads.children[i]; η = η, uparams = uparams)
            end
        end
    end
end

function update!(n::T, grads::NamedTuple; η = 0.01, uparams = []) where {T<:AbstractLeaf}
    
    v = collect(params(n))
    for (i,k) in enumerate(keys(params(n)))
        if (k ∈ uparams) || isempty(uparams)
            v[i] += grads.params[k] === nothing ? zero(eltype(v)) : η * grads.params[k]
        end
    end

    return T(n.scope, (;zip(keys(params(n)), v)...))
end

