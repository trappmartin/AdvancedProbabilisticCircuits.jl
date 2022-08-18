using AdvancedProbabilisticCircuits
using Random

X1 = collect(1:5)
X2 = collect(1:3)
X3 = collect(1:3)

X = [X1,X2,X3]

function generateProdNode(X::Vector{Vector{Int}}, dims, Ksum, Kprod)
    D = length(X)

    children = Node[]
    if Kprod < D
        y = rand(1:Kprod, D)

        for k in 1:Kprod
            push!(children, generateSumNode(X[y .== k], dims[y .== k], Ksum, Kprod))
        end
    else
        for k in 1:D
            push!(children, generateSumNode(X[k], dims[k], Ksum, Kprod))
        end
    end
    return Prod(children...)
end

function generateSumNode(X::Vector{Vector{Int}}, dims::Vector{Int}, Ksum, Kprod)
    D = length(X)

    children = Vector{Node}(undef, Ksum)
    for k in 1:Ksum
        # randomly select values
        M = map(d -> rand(Bool, length(X[d])), 1:D)

        while any(m -> all(m .== true) || all(m .== false), M)
            M = map(d -> rand(Bool, length(X[d])), 1:D)
        end

        X_ = map(d -> X[d][M[d]], 1:D)

        children[k] = generateProdNode(X_, dims, Ksum, Kprod)        
    end
    return Sum(children...)
end

function generateSumNode(X::Vector{Int}, dim::Int, Ksum, Kprod)
    K = length(X)
    children = map(x -> Indicator(x, dim), X)
    return Sum(children...)
end

function generate(X; Ksum = 4, Kprod = 2)
    dims = collect(1:length(X))
    return generateSumNode(X, dims, Ksum, Kprod)
end

c = generate(X)

print(c)