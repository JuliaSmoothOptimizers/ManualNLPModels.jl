for problem in ["hs5", "hs6"]
  include("problems/$problem.jl")

  @testset "Problem $problem" begin
    nlp_from_T = eval(Symbol(problem))
    nlp_this = nlp_from_T()
    nlp_test = eval(Symbol(uppercase(problem)))()
    @testset "Check consistency" begin
      nlps = [nlp_this; nlp_test]
      consistent_nlps(nlps, linear_api = true)
    end
    @testset "Check multiple precision" begin
      multiple_precision_nlp(nlp_from_T, linear_api = true)
    end
    @testset "Check dimensions" begin
      check_nlp_dimensions(nlp_this, linear_api = true)
    end
  end
end
