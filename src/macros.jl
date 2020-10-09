export @leaf

# from https://thautwarm.github.io/MLStyle.jl/latest/tutorials/capture.html
function capture(template, ex, action)
    let template = Expr(:quote, template)
        quote
            @match $ex begin 
                $template => $action
                _         => nothing
            end
        end 
    end
end

macro capture(template, ex, action)
    capture(template, ex, action) |> esc
end

"""
    _leaf(expr)
 
This function is called by the `@leaf` macro.
The function turns:

```julia
@leaf RNorm(m=1,s=0)
```

into

```julia
struct RNorm{P} <: ProbabilisticCircuits.AbstractLeaf
    scope::Int
    params::P
end
function RNorm(scope::Int, nt::NamedTuple)
    P = typeof(nt)
    return RNorm{P}(scope, nt)
end
RNorm(scope, m, s) = RNorm(scope, (; m, s))
RNorm(scope::Int; m = 1, s = 1) = RNorm(scope, m, s)
````

"""
function _leaf(expr)
    @capture $n($(p...)) expr begin

        err1str = "Field \'"
        err2str = "\' is missing a default value."

        kwargs_ = Any[]
        args_ = Any[]
        for k in filter(pi -> pi isa Expr, p)
            push!(args_, k.args[1])
            push!(kwargs_, k)
        end

        for k in filter(pi -> pi isa Symbol, p)
            push!(args_, k)
            push!(kwargs_, Expr(:kw, k, :(error($err1str * $k * $err2str))))
        end

        q = quote
            struct $n{P} <: ProbabilisticCircuits.AbstractLeaf
                scope::Int
                params::P
            end

            function $n(scope::Int, nt::NamedTuple)
                P = typeof(nt)
                return $n{P}(scope, nt)
            end
            $n(scope, $(args_...)) = $n(scope, (;$(args_...)))
            $n(scope::Int; $(kwargs_...)) = $n(scope, $(args_...))

            printnodetype(::Type{<:$n}) = $n
        end
        return q
    end
end

"""
Macro which generates a leaf node.
Example usage to create a Gaussian leaf:

```julia
@leaf Normal(μ = 0.0, σ = 1.0)

# define the logpdf and the support of the new leaf
function logpdf(n::Normal{P}, x::Real) where {P<:NamedTuple{(:μ, :σ)}}
    return -log(n.params.σ) - (x-n.params.μ)^2 / (2 * n.params.σ^2)
end

support(::Normal) = RealInterval(-Inf, Inf)
```

"""
macro leaf(expr)
    esc(_leaf(expr))
end