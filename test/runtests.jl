using NLPModelsTest
using ManualNLPModels
using Test

for problem in ["hs6"]
  include("problems/$problem.jl")

  @testset "Problem $problem" begin
    nlp1 = eval(Symbol(problem))()
    nlp2 = eval(Symbol(uppercase(problem)))()
    nlps = [nlp1; nlp2]
    consistent_nlps(nlps)
  end
end
