module ManualNLPModels

using NLPModels

export NLPModel

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

function NLPModels.obj(nlp::NLPModel, x::AbstractVector)
  increment!(nlp, :neval_obj)
  nlp.obj(x)
end

function NLPModels.grad!(nlp::NLPModel, x::AbstractVector, gx::AbstractVector)
  increment!(nlp, :neval_grad)
  nlp.grad(gx, x)
end

function NLPModels.hprod!(
  nlp::NLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = 1.0,
)
  increment!(nlp, :neval_hprod)
  nlp.hprod(Hv, x, v; obj_weight = obj_weight)
end

function NLPModels.hprod!(
  nlp::NLPModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = 1.0,
)
  increment!(nlp, :neval_hprod)
  nlp.hprod(Hv, x, y, v; obj_weight = obj_weight)
end

function NLPModels.hess_structure!(
  nlp::NLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= nlp.Hrows
  cols .= nlp.Hcols
  rows, cols
end

function NLPModels.hess_coord!(
  nlp::NLPModel,
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = 1.0,
)
  increment!(nlp, :neval_hess)
  nlp.Hvals(vals, x; obj_weight = obj_weight)
end

function NLPModels.hess_coord!(
  nlp::NLPModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = 1.0,
)
  increment!(nlp, :neval_hess)
  nlp.Hvals(vals, x, y; obj_weight = obj_weight)
end

function NLPModels.cons!(nlp::NLPModel, x::AbstractVector, cx::AbstractVector)
  increment!(nlp, :neval_cons)
  nlp.cons(cx, x)
end

function NLPModels.jprod!(nlp::NLPModel, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
  increment!(nlp, :neval_jprod)
  nlp.jprod(jv, x, v)
end

function NLPModels.jtprod!(nlp::NLPModel, x::AbstractVector, v::AbstractVector, jtv::AbstractVector)
  increment!(nlp, :neval_jtprod)
  nlp.jtprod(jtv, x, v)
end

function NLPModels.jac_coord!(nlp::NLPModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac)
  nlp.Jvals(vals, x)
end

function NLPModels.jac_structure!(
  nlp::NLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= nlp.Jrows
  cols .= nlp.Jcols
  rows, cols
end

end
