@testset "Allocations" begin
  @testset "Allocations for hs5" begin
    @testset "hs5" begin
      include("problems/hs5.jl")
      test_zero_allocations(hs5())
    end
  end

  @testset "Allocations for hs6" begin
    @testset "hs6" begin
      include("problems/hs6.jl")
      test_zero_allocations(hs6())
    end
  end
end
