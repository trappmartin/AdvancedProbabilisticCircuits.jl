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
@leaf RNorm(m=1,s=0) RealInterval(0.0, 1.0)
```

into

```julia
struct RNorm{P} <: AdvancedProbabilisticCircuits.AbstractLeaf
    scope::Int
    params::P
end
function RNorm(scope::Int, nt::NamedTuple)
    P = typeof(nt)
    return RNorm{P}(scope, nt)
end
RNorm(scope, m, s) = RNorm(scope, (; m, s))
RNorm(scope::Int; m = 1, s = 1) = RNorm(scope, m, s)
(printnodetype(::Type{T}) where T <: RNorm) = RNorm
support(n::RNorm) = n.scope => RealInterval(0.0, 1.0)
````

"""
function _leaf(expr1, expr2)
    @capture $n($(p...)) expr1 begin

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

        # sanity check
        if !(eval(expr2) isa RealInterval) && !(eval(expr2) isa Vector{Int})
            throw(ErrorException("Unknown support specification!"))
        end

        q = quote
            struct $n{P} <: AdvancedProbabilisticCircuits.AbstractLeaf
                scope::Int
                params::P
            end

            function $n(scope::Int, nt::NamedTuple)
                P = typeof(nt)
                return $n{P}(scope, nt)
            end
            $n(scope, $(args_...)) = $n(scope, (;$(args_...)))
            $n(scope::Int; $(kwargs_...)) = $n(scope, $(args_...))

            printnodetype(::Type{T}) where {T<:$n} = $n

            support(n::$n) = (n.scope => $expr2)
        end
        return q
    end
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
struct RNorm{P} <: AdvancedProbabilisticCircuits.AbstractLeaf
    scope::Int
    params::P
end
function RNorm(scope::Int, nt::NamedTuple)
    P = typeof(nt)
    return RNorm{P}(scope, nt)
end
RNorm(scope, m, s) = RNorm(scope, (; m, s))
RNorm(scope::Int; m = 1, s = 1) = RNorm(scope, m, s)
(printnodetype(::Type{T}) where T <: RNorm) = RNorm
````

"""
function _leaf(expr1)
    @capture $n($(p...)) expr1 begin

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
            struct $n{P} <: AdvancedProbabilisticCircuits.AbstractLeaf
                scope::Int
                params::P
            end

            function $n(scope::Int, nt::NamedTuple)
                P = typeof(nt)
                return $n{P}(scope, nt)
            end
            $n(scope, $(args_...)) = $n(scope, (;$(args_...)))
            $n(scope::Int; $(kwargs_...)) = $n(scope, $(args_...))

            printnodetype(::Type{T}) where {T<:$n} = $n
        end
        return q
    end
end

"""
Macro which generates a leaf node.
Example usage to create a Gaussian leaf with support on the whole real line:

```julia
@leaf Normal(μ = 0.0, σ = 1.0) RealInterval(-Inf, Inf)

# define the logpdf
function logpdf(n::Normal{P}, x::Real) where {P<:NamedTuple{(:μ, :σ)}}
    return -log(n.params.σ) - (x-n.params.μ)^2 / (2 * n.params.σ^2)
end
```

Example usage to create a Bernoulli leaf with support on [0,1]:

```julia
@leaf Bernoulli(p = 0.5) [0, 1]

# define the logpdf
function logpdf(n::Bernoulli{P}, x::Real) where {P<:NamedTuple{(:p)}}
    !(isone(x) | iszero(x)) && return -Inf
    return isone(x) ? log(n.params.p) : log(1-n.params.p)
end
```

If necessary, we can also define the support manually.
Example usage to create an Indicator leaf:

```julia
@leaf Indicator(v = 1)

# define the logpdf
function logpdf(n::Indicator{P}, x::Real) where {P<:NamedTuple{(:v)}}
    return x == n.params.v ? 0 : -Inf
end

# define support manually
support(n::Indicator{P}) where {P<:NamedTuple{(:v,)}} = n.scope => [n.params.v]
```

"""
macro leaf(expr1)
    esc(_leaf(expr1))
end

macro leaf(expr1, expr2)
    esc(_leaf(expr1, expr2))
end
