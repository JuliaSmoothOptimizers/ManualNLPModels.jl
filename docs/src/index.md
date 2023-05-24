```@meta
CurrentModule = ManualNLPModels
```

# ManualNLPModels

This package exists to simplify a little the process of creating an user-defined NLPModel.
The advantages of using this package is that you avoid dealing with the internals of a JSO-compliant model.
The disadvantages are that you do not have as much flexibility.

## Usage

We provide one structure/constructor: [`NLPModel`](@ref).
You have to provide the starting point and the objective function.
The additional functions are passed through keyword arguments.
You can also pass bounds on the variables.

- `nlp = NLPModel(x ,f)`: Only the objective.
- `nlp = NLPModel(x, f, grad=my_grad)`: Also the gradient (in place).
- `nlp = NLPModel(x, f, grad=my_grad, hprod=my_hprod)`: Also the Hessian-vector product (in place).

Check the details in the reference of `NLPModel`.

### Objective and gradient of a function

```@example 1
using ManualNLPModels, JSOSolvers

f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
g!(gx, x) = begin
  gx[1] = 2 * (x[1] - 1) - 16 * x[1] * (x[2] - x[1]^2)
  gx[2] = 8 * (x[2] - x[1]^2)
  gx
end

nlp = NLPModel(
  [-1.2; 1.0],
  f,
  grad = g!,
)

output = lbfgs(nlp)
println(output)
```

### Objective and gradient at the same time

```@example 2
using ManualNLPModels, JSOSolvers

f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
fg!(gx, x) = begin
  y1, y2 = x[1] - 1, x[2] - x[1]^2
  f = y1^2 + 4 * y2^2
  gx[1] = 2 * y1 - 16 * x[1] * y2
  gx[2] = 8 * y2
  return f, gx
end

nlp = NLPModel(
  [-1.2; 1.0],
  f,
  objgrad = fg!,
)

output = lbfgs(nlp)
println(output)
```

### Objective, gradient, and Hessian-vector product

```@example 3
using ManualNLPModels, JSOSolvers

f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
fg!(gx, x) = begin
  y1, y2 = x[1] - 1, x[2] - x[1]^2
  f = y1^2 + 4 * y2^2
  gx[1] = 2 * y1 - 16 * x[1] * y2
  gx[2] = 8 * y2
  return f, gx
end
hv!(hv, x, v; obj_weight = 1.0) = begin
  h11 = 2 - 16 * x[2] + 48 * x[1]^2
  h12 = -16 * x[1]
  h22 = 8.0
  hv[1] = (h11 * v[1] + h12 * v[2]) * obj_weight
  hv[2] = (h12 * v[1] + h22 * v[2]) * obj_weight
  return hv
end

nlp = NLPModel(
  [-1.2; 1.0],
  f,
  objgrad = fg!,
  hprod = hv!,
)

output = trunk(nlp)
println(output)
```
