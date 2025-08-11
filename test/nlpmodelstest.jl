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
    if CUDA.functional()
      @testset "Check GPU multiple precision" begin
        CUDA.allowscalar() do
          multiple_precision_nlp_array(nlp_from_T, CuArray, linear_api = true)
        end
      end
    end
    @testset "Check dimensions" begin
      check_nlp_dimensions(nlp_this, linear_api = true)
    end
  end
end

for nlsproblem in ["mgh01", "mgh04"]
  include("problems/$nlsproblem.jl")

  @testset "NLS problem $nlsproblem" begin
    nls_from_T = eval(Symbol(nlsproblem))
    nls_this = nls_from_T()
    exclude = [
      hess_residual,
      hess_structure_residual,
      hess_coord_residual,
      # jth_hess_residual,
      # jth_hess_residual_coord,
      hprod_residual,
      hess_op_residual,
    ]
    @testset "Check multiple precision" begin
      multiple_precision_nls(nls_from_T, exclude = exclude)
    end
    if CUDA.functional()
      @testset "Check GPU multiple precision" begin
        CUDA.allowscalar() do
          multiple_precision_nls_array(nls_from_T, CuArray, exclude = exclude)
        end
      end
    end
    @testset "Check dimensions" begin
      check_nls_dimensions(nls_this, exclude = exclude)
    end
  end
end
