mgh01() = mgh01(Float64)
mgh01(::Type{T}) where {T <: Number} = mgh01(Vector{T})
function mgh01(::Type{V}) where {V}
  function residual!(Fx::AbstractVector, x::AbstractVector)
    Fx[1] = 1 - x[1]
    Fx[2] = 10 * (x[2] - x[1]^2)
    return Fx
  end

  # Jx = [-1  0;
  #       -20x₁  10]
  rows = [1, 2, 2]
  cols = [1, 1, 2]
  function jac_coord_residual!(vals::AbstractVector, x::AbstractVector)
    vals[1] = -1
    vals[2] = -20x[1]
    vals[3] = 10
    return vals
  end

  function jprod_residual!(Jv::AbstractVector, x::AbstractVector, v::AbstractVector)
    Jv[1] = -v[1]
    Jv[2] = -20 * x[1] * v[1] + 10 * v[2]
    return Jv
  end

  function jtprod_residual!(Jtv::AbstractVector, x::AbstractVector, v::AbstractVector)
    Jtv[1] = -v[1] - 20 * x[1] * v[2]
    Jtv[2] = 10 * v[2]
    return Jtv
  end

  NLSModel(
    V([-12 // 10, 1]),
    residual!,
    2,
    jprod = jprod_residual!,
    jtprod = jtprod_residual!,
    jac_coord = (rows, cols, jac_coord_residual!),
  )
end
