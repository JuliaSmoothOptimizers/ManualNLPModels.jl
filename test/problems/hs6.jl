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
hs6() = hs6(Float64)
hs6(::Type{T}) where {T <: Number} = hs6(Vector{T})
function hs6(::Type{V}) where {V}
  T = eltype(V)
  hprod(hv, x, v; obj_weight = one(T)) = (hv[1] = 2obj_weight * v[1]; hv[2] = 0; hv)
  hprod(hv, x, y, v; obj_weight = one(T)) = (hv[1] = (2obj_weight - 20y[1]) * v[1]; hv[2] = 0; hv)
  hess_coord(vals, x; obj_weight = one(T)) = (vals .= 2obj_weight)
  hess_coord(vals, x, y; obj_weight = one(T)) = (vals .= 2obj_weight - 20y[1])
  return NLPModel(
    V([-12 // 10; 1]),
    x -> (1 - x[1])^2;
    grad = (gx, x) -> (gx[1] = 2 * (x[1] - 1); gx[2] = 0; gx),
    # objgrad explicitly not implemented
    hprod = hprod,
    hess_coord = ([1], [1], hess_coord),
    cons = (
      (cx, x) -> (cx[1] = 10 * (x[2] - x[1]^2); cx),
      fill!(V(undef, 1), 0),
      fill!(V(undef, 1), 0),
    ),
    jac_coord = ([1, 1], [1, 2], (vals, x) -> (vals[1] = -20x[1]; vals[2] = 10; vals)),
    jprod = (jv, x, v) -> (jv[1] = -20x[1] * v[1] + 10v[2]; jv),
    jtprod = (jtv, x, v) -> (jtv[1] = -20x[1] * v[1]; jtv[2] = 10 * v[1]; jtv),
  )
end
