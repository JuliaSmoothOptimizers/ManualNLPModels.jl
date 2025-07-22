export NLPModel

"""
    nlp = NLPModel(x, f; kwargs...)

Creates a nonlinear optimization model with objective function `f`, and
starting point `x`.
You can provide bounds and additional functions by keyword arguments.
Here is the list of accepted kwyword arguments and their default value:

Unconstrained:
- `grad = (gx, x) -> gx`: gradient of `f` at `x`. Stores in `gx`.
- `objgrad = (gx, x) -> (f, gx)`: `f` and gradient of `f` at `x`. Stores in `gx`.
- `hprod = (hv, x, v; obj_weight=1) -> ...`: Hessian at `x` times vector `v`. Stores in `hv`.
- `hess_coord = (rows, cols, (vals, x; obj_weight=1) -> ...)`: sparse Hessian at `x` in triplet format.

Constrained:
- `lvar = -Inf * ones(nvar)`: vecteur of lower bounds on `x`. 
- `uvar = Inf * ones(nvar)`: vecteur of upper bounds on `x`. 
- `cons = ((cx, x) -> ..., lcon, ucon)`: constraints at `x`. Stores in `cx`. `lcon` and `ucon` are the constraint bounds.
- `jprod = (jv, x, v) -> ...`: Jacobian at `x` times vector `v`. Stores in `jv`.
- `jtprod = (jtv, x, v) -> ...`: transposed Jacobian at `x` times vector `v`. Stores in `jtv`.
- `jac_coord = (rows, cols, (vals, x) -> ....)`: sparse Jacobian at `x` in triplet format.
- `hprod = (hv, x, y, v; obj_weight=1) -> ...`: Lagrangian Hessian at `(x, y)` times vector `v`. Stores in `hv`.
- `hess_coord = (rows, cols, (vals, x, y; obj_weight=1) -> ...)`: sparse Lagrangian Hessian at `(x,y)` in triplet format.
"""
struct NLPModel{T, V, F, G, FG, Hv, Vi, H, C, Jv, Jtu, J} <: AbstractNLPModel{T, V}
  meta::NLPModelMeta{T, V}
  counters::Counters
  obj::F # obj(x)
  grad::G # grad(gx, x)
  objgrad::FG # objgrad(gx, x) -> (f, gx)
  hprod::Hv # hprod(hv, x, v; obj_weight::Real=1) or hprod(hv, x, y, v; obj_weight::Real=1)
  Hrows::Vi
  Hcols::Vi
  Hvals::H # Hvals(vals, x; obj_weight::Real=1) or Hvals(vals, x, y; obj_weight::Real=1)
  cons::C # cons(cx, x)
  jprod::Jv # jprod(jv, x, v)
  jtprod::Jtu # jtprod(jtv, x, v)
  Jrows::Vi
  Jcols::Vi
  Jvals::J # Jvals(vals, x)
end

function notimplemented(args...; kwargs...)
  error("The function you called was not implemented.")
end

function NLPModel(
  x::V,
  obj;
  lvar::V = fill!(V(undef, length(x)), -Inf),
  uvar::V = fill!(V(undef, length(x)), Inf),
  grad = notimplemented,
  objgrad = notimplemented,
  hprod = notimplemented,
  hess_coord = (Int[], Int[], notimplemented),
  cons = (notimplemented, V(undef, 0), V(undef, 0)),
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
    lvar = lvar,
    uvar = uvar,
    nnzj = nnzj,
    nnzh = nnzh,
    ncon = length(lcon),
    lcon = lcon,
    ucon = ucon;
    meta_args...,
  )
  grad = grad == notimplemented ? (gx, x) -> objgrad(gx, x)[2] : grad
  F = typeof(obj)
  G = typeof(grad)
  FG = typeof(objgrad)
  Hv = typeof(hprod)
  Vi = typeof(Hrows)
  H = typeof(Hvals)
  C = typeof(c)
  Jv = typeof(jprod)
  Jtu = typeof(jtprod)
  J = typeof(Jvals)
  return NLPModel{T, V, F, G, FG, Hv, Vi, H, C, Jv, Jtu, J}(
    meta,
    Counters(),
    obj,
    grad,
    objgrad,
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

@deprecate NLPModel(x, ℓ, u, args...; kwargs...) NLPModel(x, args...; lvar = ℓ, uvar = u, kwargs...)
