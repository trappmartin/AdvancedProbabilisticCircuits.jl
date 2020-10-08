export update!

function update!(n::Node, grads::NamedTuple; η = 0.01)
    if !isnothing(grads.params)
        n.params .+= η * grads.params
    end

    for (i, child) in enumerate(n.children)
        update!(child, grads.children[i]; η = η)
    end
end

function update!(l::Leaf, grads::NamedTuple; η = 0.01)
    # not implemented yet
end
