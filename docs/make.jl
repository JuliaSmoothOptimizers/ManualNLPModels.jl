using ManualNLPModels
using Documenter

DocMeta.setdocmeta!(ManualNLPModels, :DocTestSetup, :(using ManualNLPModels); recursive = true)

makedocs(;
  modules = [ManualNLPModels],
  doctest = true,
  linkcheck = false,
  strict = true,
  authors = "Abel Soares Siqueira <abel.s.siqueira@gmail.com> and contributors",
  repo = "https://github.com/JuliaSmoothOptimizers/ManualNLPModels.jl/blob/{commit}{path}#{line}",
  sitename = "ManualNLPModels.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://JuliaSmoothOptimizers.github.io/ManualNLPModels.jl",
    assets = ["assets/style.css"],
  ),
  pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo = "github.com/JuliaSmoothOptimizers/ManualNLPModels.jl",
  push_preview = true,
  devbranch = "main",
)
