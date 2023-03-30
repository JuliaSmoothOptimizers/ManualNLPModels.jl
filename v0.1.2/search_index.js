var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [ManualNLPModels]","category":"page"},{"location":"reference/#ManualNLPModels.NLPModel","page":"Reference","title":"ManualNLPModels.NLPModel","text":"nlp = NLPModel(x, f; kwargs...)\nnlp = NLPModel(x, lvar, uvar, f; kwargs...)\n\nCreates a nonlinear optimization model with objective function f, starting point x, and variables bounds lvar and uvar (if provided). You can provide additional functions by keyword arguments. Here is the list of accepted function names and their signatures:\n\nUnconstrained:\n\ngrad = (gx, x) -> ...: gradient of f at x. Stores in gx.\nhprod = (hv, x, v; obj_weight=1) -> ...: Hessian at x times vector v. Stores in hv.\nhess_coord = (rows, cols, (vals, x; obj_weight=1) -> ...): sparse Hessian at x in triplet format.\n\nConstrained:\n\ncons = ((cx, x) -> ..., lcon, ucon): constraints at x. Stores in cx. lcon and ucon are the constraint bounds.\njprod = (jv, x, v) -> ...: Jacobian at x times vector v. Stores in jv.\njtprod = (jtv, x, v) -> ...: transposed Jacobian at x times vector v. Stores in jtv.\njac_coord = (rows, cols, (vals, x) -> ....): sparse Jacobian at x in triplet format.\nhprod = (hv, x, y, v; obj_weight=1) -> ...: Lagrangian Hessian at (x, y) times vector v. Stores in hv.\nhess_coord = (rows, cols, (vals, x, y; obj_weight=1) -> ...): sparse Lagrangian Hessian at (x,y) in triplet format.\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = ManualNLPModels","category":"page"},{"location":"#ManualNLPModels","page":"Home","title":"ManualNLPModels","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ManualNLPModels.","category":"page"}]
}