export hs5

"""
    nlp = HS5()

## Problem 5 in the Hock-Schittkowski suite

```math
\\begin{aligned}
\\min \\quad & \\sin(x_1 + x_2) + (x_1 - x_2)^2 - \\tfrac{3}{2}x_1 + \\tfrac{5}{2}x_2 + 1 \\\\
\\text{s. to} \\quad & -1.5 \\leq x_1 \\leq 4 \\\\
& -3 \\leq x_2 \\leq 3
\\end{aligned}
```

Starting point: `[0.0; 0.0]`.
"""
hs5() = hs5(Float64)
hs5(::Type{T}) where {T <: Number} = hs5(Vector{T})
function hs5(::Type{V}) where {V}
  T = eltype(V)
  hprod(hv, x, v; obj_weight = one(T)) = (
    hv .=
      (-sin(x[1] + x[2]) * (v[1] + v[2]) .+ 2 * V([v[1] - v[2]; v[2] - v[1]])) *
      obj_weight
  )
  hess_coord(vals, x; obj_weight = one(T)) = begin
    vals[1] = vals[3] = -sin(x[1] + x[2]) + 2
    vals[2] = -sin(x[1] + x[2]) - 2
    vals .*= obj_weight
  end
  f(x) = sin(x[1] + x[2]) + (x[1] - x[2])^2 - 3x[1] / 2 + 5x[2] / 2 + 1
  grad(gx, x) = (gx .= cos(x[1] + x[2]) .+ 2 * (x[1] - x[2]) * V([1; -1]) + V([-15 // 10; 25 // 10]))
  objgrad(gx, x) = f(x), grad(gx, x)
  return NLPModel(
    fill!(V(undef, 2), 0),
    V([-15 // 10; -3]),
    V([4; 3]),
    f,
    grad = grad,
    objgrad = objgrad,
    hprod = hprod,
    hess_coord = ([1, 2, 2], [1, 1, 2], hess_coord),
  )
end
