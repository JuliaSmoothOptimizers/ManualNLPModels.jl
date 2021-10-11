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
function hs5(::Type{T} = Float64) where {T}
  hprod(hv, x, v; obj_weight=one(T)) = (hv .= (-sin(x[1] + x[2]) * (v[1] + v[2]) * ones(T, 2) + 2 * [v[1] - v[2]; v[2] - v[1]]) * obj_weight)
  hess_coord(vals, x; obj_weight=one(T)) = begin
    vals[1] = vals[3] = -sin(x[1] + x[2]) + 2
    vals[2] = -sin(x[1] + x[2]) - 2
    vals .*= obj_weight
  end
  return NLPModel(
    zeros(T, 2),
    T[-1.5; -3],
    T[4; 3],
    x -> sin(x[1] + x[2]) + (x[1] - x[2])^2 - 3x[1] / 2 + 5x[2] / 2 + 1,
    grad = (gx, x) -> (gx .= cos(x[1] + x[2]) * ones(T, 2) + 2 * (x[1] - x[2]) * T[1; -1] + T[-1.5; 2.5]),
    hprod = hprod,
    hess_coord = ([1, 2, 2], [1, 1, 2], hess_coord),
  )
end
