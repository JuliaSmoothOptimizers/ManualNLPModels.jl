using NLPModelsTest
using ManualNLPModels
using Test

for problem in ["hs6"]
  include("problems/$problem.jl")

  @testset "Problem $problem" begin
    nlp_this = eval(Symbol(problem))()
    @testset "Check consistency" begin
      nlp_test = eval(Symbol(uppercase(problem)))()
      nlps = [nlp_this; nlp_test]
      consistent_nlps(nlps)
    end
    @testset "Check multiple precision" begin
      nlp_from_T = eval(Symbol(problem))
      multiple_precision_nlp(nlp_from_T)
    end
  end
end
