function NLPModels.obj(nlp::NLPModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_obj)
  nlp.obj(x)
end

function NLPModels.grad!(nlp::NLPModel, x::AbstractVector, gx::AbstractVector)
  @lencheck nlp.meta.nvar x gx
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
  @lencheck nlp.meta.nvar x v Hv
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
  @lencheck nlp.meta.nvar x v Hv
  @lencheck nlp.meta.ncon y
  increment!(nlp, :neval_hprod)
  nlp.hprod(Hv, x, y, v; obj_weight = obj_weight)
end

function NLPModels.hess_structure!(
  nlp::NLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.meta.nnzh rows cols
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
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzh vals
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
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  nlp.Hvals(vals, x, y; obj_weight = obj_weight)
end

function NLPModels.cons_nln!(nlp::NLPModel, x::AbstractVector, cx::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnln cx
  increment!(nlp, :neval_cons_nln)
  nlp.cons(cx, x)
end

function NLPModels.jprod_nln!(
  nlp::NLPModel,
  x::AbstractVector,
  v::AbstractVector,
  jv::AbstractVector,
)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.nnln jv
  increment!(nlp, :neval_jprod_nln)
  nlp.jprod(jv, x, v)
end

function NLPModels.jtprod_nln!(
  nlp::NLPModel,
  x::AbstractVector,
  v::AbstractVector,
  jtv::AbstractVector,
)
  @lencheck nlp.meta.nvar x jtv
  @lencheck nlp.meta.nnln v
  increment!(nlp, :neval_jtprod_nln)
  nlp.jtprod(jtv, x, v)
end

function NLPModels.jac_nln_structure!(
  nlp::NLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck nlp.meta.nln_nnzj rows cols
  rows .= nlp.Jrows
  cols .= nlp.Jcols
  rows, cols
end

function NLPModels.jac_nln_coord!(nlp::NLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nln_nnzj vals
  increment!(nlp, :neval_jac_nln)
  nlp.Jvals(vals, x)
end
