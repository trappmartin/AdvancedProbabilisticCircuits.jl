function printstyle(node::Node{T,V,S,P,N}) where {T,V,S,P,N<:NodeType}
    s = printnodetype(N)
    s = issmooth(node) ? BLUE_FG(s) : isdecomposable(node) ? RED_FG(s) : s
    s = isdeterministic(node) ? NEGATIVE(s) : s
    return s
end

function prettyprint(io::IO, node::Node{T,V,S,P,SumNode}, level::Int) where {T,V,S,P}
    println(io, string(repeat('\t', level), printstyle(node), " ("))
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

function prettyprint(io::IO, node::Node{T,V,S,P,N}, level::Int) where {T,V,S,P,N}
    println(io, string(repeat('\t', level), printstyle(node), " ("))
    for child in node.children[1:end-1]
        prettyprint(io, child, level+1)
        println(io, ", ")
    end
    prettyprint(io, node.children[end], level+1)
    print(io, string(repeat('\t', level), ")"))
end

function prettyprint(io::IO, node::T, level::Int) where {T<:AbstractLeaf}
    print(io, string(repeat('\t', level), printnodetype(T), "[",params(node),"] - scope: $(node.scope)"))
end

function Base.show(io::IO, node::AbstractNode)
    prettyprint(io, node, 0)
end