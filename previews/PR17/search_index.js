var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [ManualNLPModels]","category":"page"},{"location":"reference/#ManualNLPModels.NLPModel","page":"Reference","title":"ManualNLPModels.NLPModel","text":"nlp = NLPModel(x, f; kwargs...)\nnlp = NLPModel(x, lvar, uvar, f; kwargs...)\n\nCreates a nonlinear optimization model with objective function f, starting point x, and variables bounds lvar and uvar (if provided). You can provide additional functions by keyword arguments. Here is the list of accepted function names and their signatures:\n\nUnconstrained:\n\ngrad = (gx, x) -> gx: gradient of f at x. Stores in gx.\nobjgrad = (gx, x) -> (f, gx): f and gradient of f at x. Stores in gx.\nhprod = (hv, x, v; obj_weight=1) -> ...: Hessian at x times vector v. Stores in hv.\nhess_coord = (rows, cols, (vals, x; obj_weight=1) -> ...): sparse Hessian at x in triplet format.\n\nConstrained:\n\ncons = ((cx, x) -> ..., lcon, ucon): constraints at x. Stores in cx. lcon and ucon are the constraint bounds.\njprod = (jv, x, v) -> ...: Jacobian at x times vector v. Stores in jv.\njtprod = (jtv, x, v) -> ...: transposed Jacobian at x times vector v. Stores in jtv.\njac_coord = (rows, cols, (vals, x) -> ....): sparse Jacobian at x in triplet format.\nhprod = (hv, x, y, v; obj_weight=1) -> ...: Lagrangian Hessian at (x, y) times vector v. Stores in hv.\nhess_coord = (rows, cols, (vals, x, y; obj_weight=1) -> ...): sparse Lagrangian Hessian at (x,y) in triplet format.\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = ManualNLPModels","category":"page"},{"location":"#ManualNLPModels","page":"Home","title":"ManualNLPModels","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package exists to simplify a little the process of creating an user-defined NLPModel. The advantages of using this package is that you avoid dealing with the internals of a JSO-compliant model. The disadvantages are that you do not have as much flexibility.","category":"page"},{"location":"#Usage","page":"Home","title":"Usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"We provide one structure/constructor: NLPModel. You have to provide the starting point and the objective function. The additional functions are passed through keyword arguments. You can also pass bounds on the variables.","category":"page"},{"location":"","page":"Home","title":"Home","text":"nlp = NLPModel(x ,f): Only the objective.\nnlp = NLPModel(x, f, grad=my_grad): Also the gradient (in place).\nnlp = NLPModel(x, f, grad=my_grad, hprod=my_hprod): Also the Hessian-vector product (in place).","category":"page"},{"location":"","page":"Home","title":"Home","text":"Check the details in the reference of NLPModel.","category":"page"},{"location":"#Objective-and-gradient-of-a-function","page":"Home","title":"Objective and gradient of a function","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using ManualNLPModels, JSOSolvers\n\nf(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2\ng!(gx, x) = begin\n  gx[1] = 2 * (x[1] - 1) - 16 * x[1] * (x[2] - x[1]^2)\n  gx[2] = 8 * (x[2] - x[1]^2)\n  gx\nend\n\nnlp = NLPModel(\n  [-1.2; 1.0],\n  f,\n  grad = g!,\n)\n\noutput = lbfgs(nlp)\nprintln(output)","category":"page"},{"location":"#Objective-and-gradient-at-the-same-time","page":"Home","title":"Objective and gradient at the same time","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using ManualNLPModels, JSOSolvers\n\nf(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2\nfg!(gx, x) = begin\n  y1, y2 = x[1] - 1, x[2] - x[1]^2\n  f = y1^2 + 4 * y2^2\n  gx[1] = 2 * y1 - 16 * x[1] * y2\n  gx[2] = 8 * y2\n  return f, gx\nend\n\nnlp = NLPModel(\n  [-1.2; 1.0],\n  f,\n  objgrad = fg!,\n)\n\noutput = lbfgs(nlp)\nprintln(output)","category":"page"},{"location":"#Objective,-gradient,-and-Hessian-vector-product","page":"Home","title":"Objective, gradient, and Hessian-vector product","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using ManualNLPModels, JSOSolvers\n\nf(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2\nfg!(gx, x) = begin\n  y1, y2 = x[1] - 1, x[2] - x[1]^2\n  f = y1^2 + 4 * y2^2\n  gx[1] = 2 * y1 - 16 * x[1] * y2\n  gx[2] = 8 * y2\n  return f, gx\nend\nhv!(hv, x, v; obj_weight = 1.0) = begin\n  h11 = 2 - 16 * x[2] + 48 * x[1]^2\n  h12 = -16 * x[1]\n  h22 = 8.0\n  hv[1] = h11 * v[1] + h12 * v[2]\n  hv[2] = h12 * v[1] + h22 * v[2]\n  return hv * obj_weight\nend\n\nnlp = NLPModel(\n  [-1.2; 1.0],\n  f,\n  objgrad = fg!,\n  hprod = hv!,\n)\n\noutput = trunk(nlp)\nprintln(output)","category":"page"}]
}
