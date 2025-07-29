function NLPModels.residual!(nls::NLSModel, x::AbstractVector, Fx::AbstractVector)
  NLPModels.@lencheck nls.meta.nvar x
  NLPModels.@lencheck nls.nls_meta.nequ Fx
  increment!(nls, :neval_residual)
  nls.resid!(Fx, x)
  Fx
end

function NLPModels.jprod_residual!(
  nls::NLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  NLPModels.@lencheck nls.meta.nvar x v
  NLPModels.@lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
  nls.jprod_resid!(Jv, x, v)
  Jv
end

function NLPModels.jtprod_residual!(
  nls::NLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  NLPModels.@lencheck nls.meta.nvar x Jtv
  NLPModels.@lencheck nls.nls_meta.nequ v
  increment!(nls, :neval_jtprod_residual)
  nls.jtprod_resid!(Jtv, x, v)
  Jtv
end

function NLPModels.jac_structure_residual!(
  nls::NLSModel,
  rows::AbstractVector,
  cols::AbstractVector,
)
  NLPModels.@lencheck nls.nls_meta.nnzj rows cols
  rows .= nls.jrows
  cols .= nls.jcols
  return (rows, cols)
end

function NLPModels.jac_coord_residual!(nls::NLSModel, x::AbstractVector, vals::AbstractVector)
  NLPModels.@lencheck nls.meta.nvar x
  NLPModels.@lencheck nls.nls_meta.nnzj vals
  increment!(nls, :neval_jac_residual)
  nls.jvals!(vals, x)
  vals
end
