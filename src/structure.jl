export NLPModel

"""
    nlp = NLPModel(x, f; kwargs...)
    nlp = NLPModel(x, lvar, uvar, f; kwargs...)

Creates a nonlinear optimization model with objective function `f`,
starting point `x`, and variables bounds `lvar` and `uvar` (if provided).
You can provide additional functions by keyword arguments.
Here is the list of accepted function names and their signatures:

Unconstrained:
- `grad = (gx, x) -> ...`: gradient of `f` at `x`. Stores in `gx`.
- `hprod = (hv, x, v; obj_weight=1) -> ...`: Hessian at `x` times vector `v`. Stores in `hv`.
- `hess_coord = (rows, cols, (vals, x; obj_weight=1) -> ...)`: sparse Hessian at `x` in triplet format.

Constrained:
- `cons = ((cx, x) -> ..., lcon, ucon)`: constraints at `x`. Stores in `cx`. `lcon` and `ucon` are the constraint bounds.
- `jprod = (jv, x, v) -> ...`: Jacobian at `x` times vector `v`. Stores in `jv`.
- `jtprod = (jtv, x, v) -> ...`: transposed Jacobian at `x` times vector `v`. Stores in `jtv`.
- `jac_coord = (rows, cols, (vals, x) -> ....)`: sparse Jacobian at `x` in triplet format.
- `hprod = (hv, x, y, v; obj_weight=1) -> ...`: Lagrangian Hessian at `(x, y)` times vector `v`. Stores in `hv`.
- `hess_coord = (rows, cols, (vals, x, y; obj_weight=1) -> ...)`: sparse Lagrangian Hessian at `(x,y)` in triplet format.
"""
struct NLPModel{T, V} <: AbstractNLPModel{T, V}
  meta::NLPModelMeta{T, V}
  counters::Counters
  obj # obj(x)
  grad # grad(gx, x)
  hprod # hprod(hv, x, v; obj_weight::Real=1) or hprod(hv, x, y, v; obj_weight::Real=1)
  Hrows
  Hcols
  Hvals # Hvals(vals, x; obj_weight::Real=1) or Hvals(vals, x, y; obj_weight::Real=1)
  cons # cons(cx, x)
  jprod # jprod(jv, x, v)
  jtprod # jtprod(jtv, x, v)
  Jrows
  Jcols
  Jvals # Jvals(vals, x)
end

function notimplemented(args...; kwargs...)
  error("The function you called was not implemented.")
end

function NLPModel(
  x::V,
  obj;
  grad = notimplemented,
  hprod = notimplemented,
  hess_coord = (Int[], Int[], notimplemented),
  cons = (notimplemented, T[], T[]),
  jprod = notimplemented,
  jtprod = notimplemented,
  jac_coord = (Int[], Int[], notimplemented),
  meta_args = (),
) where {T, V <: AbstractVector{T}}
  Hrows, Hcols, Hvals = hess_coord
  Jrows, Jcols, Jvals = jac_coord
  c, lcon, ucon = cons
  nnzh, nnzj = length(Hrows), length(Jrows)
  meta = NLPModelMeta{T, V}(
    length(x),
    x0 = x,
    nnzj = nnzj,
    nnzh = nnzh,
    ncon = length(lcon),
    lcon = lcon,
    ucon = ucon;
    meta_args...,
  )
  return NLPModel{T, V}(
    meta,
    Counters(),
    obj,
    grad,
    hprod,
    Hrows,
    Hcols,
    Hvals,
    c,
    jprod,
    jtprod,
    Jrows,
    Jcols,
    Jvals,
  )
end

function NLPModel(
  x::V,
  ℓ::V,
  u::V,
  obj;
  grad = notimplemented,
  hprod = notimplemented,
  hess_coord = (Int[], Int[], notimplemented),
  cons = (notimplemented, T[], T[]),
  jprod = notimplemented,
  jtprod = notimplemented,
  jac_coord = (Int[], Int[], notimplemented),
  meta_args = (),
) where {T, V <: AbstractVector{T}}
  Hrows, Hcols, Hvals = hess_coord
  Jrows, Jcols, Jvals = jac_coord
  c, lcon, ucon = cons
  nnzh, nnzj = length(Hrows), length(Jrows)
  meta = NLPModelMeta{T, V}(
    length(x),
    x0 = x,
    lvar = ℓ,
    uvar = u,
    nnzj = nnzj,
    nnzh = nnzh,
    ncon = length(lcon),
    lcon = lcon,
    ucon = ucon;
    meta_args...,
  )
  return NLPModel{T, V}(
    meta,
    Counters(),
    obj,
    grad,
    hprod,
    Hrows,
    Hcols,
    Hvals,
    c,
    jprod,
    jtprod,
    Jrows,
    Jcols,
    Jvals,
  )
end
