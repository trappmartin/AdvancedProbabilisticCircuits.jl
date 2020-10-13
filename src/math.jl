
# Function to compute log(sum(exp.(x))) using [Sebastian Nowozin: Streaming Log-sum-exp Computation.](http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html)

_f1(x1, x2) = x1 == x2 ? zero(x1 - x2) : -abs(x1 - x2)
_f2(x1, x2, a) = x1 > x2 ? oftype(a, x1) : oftype(a, x2)
_f3(x1, x2, r, a) = x1 > x2 ? (r + one(r)) * exp(a) : r + exp(a)
_f4(x1, x2, r1, r2, a) = x1 > x2 ? r1 + (r2 + one(r2)) * exp(a) : r2 + (r1 + one(r1)) * exp(a)

function _logsumexp_onepass_op(x1::AbstractVector, x2::AbstractVector)
    a = _f1.(x1,x2)
    xmax = _f2.(x1,x2,a)
    r = exp.(a)
    return xmax, r
end

function _logsumexp_onepass_op(x, (xmax, r)::Tuple)
    a = _f1.(x, xmax)
    _xmax = _f2.(x,xmax,a)
    _r = _f3.(x, xmax, r, a)
    return _xmax, _r
end
_logsumexp_onepass_op(xmax_r::Tuple, x) = _logsumexp_onepass_op(x, xmax_r)

function _logsumexp_onepass_op((xmax1, r1)::Tuple, (xmax2, r2)::Tuple)
    a = _f1.(xmax1, xmax2)
    xmax = _f2.(xmax1, xmax2)
    r = _f4.(xmax1, xmax2, r1, r2, a)
    return xmax, r
end

