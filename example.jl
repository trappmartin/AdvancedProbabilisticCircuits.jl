using ProbabilisticCircuits

x = randn(3, 100); # some dummy data

n = Sum(Leaf(truncated(Normal(), 0+eps(), Inf), 1), Leaf(truncated(Normal(), -Inf, 0), 1))

@show n
println()
println("Check node properties...")
@show issmooth(n);
@show isdeterministic(n);
@show isdecomposable(n);

println()
println("---")
println()

n = Sum(Leaf(truncated(Normal(), 0+eps(), Inf), 2), Leaf(truncated(Normal(), -Inf, 0), 1))

@show n
println()
println("Check node properties...")
@show issmooth(n);
@show isdeterministic(n);
@show isdecomposable(n);

println()
println("---")
println()

n = Sum(Leaf(Normal(), 1), Leaf(Normal(), 1))

@show n
println()
println("Check node properties...")
@show issmooth(n);
@show isdeterministic(n);
@show isdecomposable(n);

println()
println("---")
println()

n = Prod(Leaf(Normal(), 1), Leaf(Normal(), 2))

@show n
println()
println("Check node properties...")
@show issmooth(n);
@show isdeterministic(n);
@show isdecomposable(n);
