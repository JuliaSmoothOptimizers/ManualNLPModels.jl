export hs6

"""
    nlp = hs6()

## Problem 6 in the Hock-Schittkowski suite

```math
\\begin{aligned}
\\min \\quad & (1 - x_1)^2 \\\\
\\text{s. to} \\quad & 10 (x_2 - x_1^2) = 0
\\end{aligned}
```

Starting point: `[-1.2; 1.0]`.
"""
function hs6(::Type{T} = Float64) where {T}
  hprod(hv, x, v; obj_weight = one(T)) = (hv .= [2obj_weight * v[1]; 0])
  hprod(hv, x, y, v; obj_weight = one(T)) = (hv .= [(2obj_weight - 20y[1]) * v[1]; 0])
  hess_coord(vals, x; obj_weight = one(T)) = (vals .= 2obj_weight)
  hess_coord(vals, x, y; obj_weight = one(T)) = (vals .= 2obj_weight - 20y[1])
  return NLPModel(
    T[-1.2; 1],
    x -> (1 - x[1])^2;
    grad = (gx, x) -> gx .= [2 * (x[1] - 1); 0],
    hprod = hprod,
    hess_coord = ([1], [1], hess_coord),
    cons = ((cx, x) -> (cx[1] = 10 * (x[2] - x[1]^2); cx), T[0], T[0]),
    jac_coord = ([1, 1], [1, 2], (vals, x) -> vals .= T[-20x[1]; 10]),
    jprod = (jv, x, v) -> jv .= [-20x[1] * v[1] + 10v[2]],
    jtprod = (jtv, x, v) -> jtv .= [-20x[1]; 10] * v[1],
  )
end
