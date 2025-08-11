mgh04() = mgh04(Float64)
mgh04(::Type{T}) where {T <: Number} = mgh04(Vector{T})
function mgh04(::Type{V}) where {V}
  T = eltype(V)

  function residual!(Fx::AbstractVector, x::AbstractVector)
    Fx[1] = x[1] - T(1e6)
    Fx[2] = x[2] - T(2e-6)
    Fx[3] = x[1] * x[2] - 2
    return Fx
  end

  # Jx = [1    0;
  #       1    0;
  #       x[2] x[1]]
  rows = [1, 1, 1, 2]
  cols = [1, 1, 1, 2]
  function jac_coord_residual!(vals::AbstractVector, x::AbstractVector)
    vals[1] = 1
    vals[2] = 1
    vals[3] = x[2]
    vals[4] = x[1]
    return vals
  end

  function jprod_residual!(Jv::AbstractVector, x::AbstractVector, v::AbstractVector)
    Jv[1] = v[1]
    Jv[2] = v[2]
    Jv[3] = x[2] * v[1] + x[1] * v[2]
    return Jv
  end

  function jtprod_residual!(Jtv::AbstractVector, x::AbstractVector, v::AbstractVector)
    Jtv[1] = v[1] + v[2] + x[2] * v[3]
    Jtv[2] = x[1] * v[3]
    return Jtv
  end

  NLSModel(
    V([1, 1]),
    residual!,
    3,
    jprod = jprod_residual!,
    jtprod = jtprod_residual!,
    jac_coord = (rows, cols, jac_coord_residual!),
  )
end
