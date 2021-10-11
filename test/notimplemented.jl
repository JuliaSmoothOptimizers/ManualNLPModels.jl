@testset "Not implemented tests" begin
  blah(args...; kwargs...) = args[1]
  x = zeros(2)
  y = zeros(0)

  @testset "Minimum NLPModel throws for everything else" begin
    for nlp in [
      NLPModel(x, blah),
      NLPModel(x, -ones(2), ones(2), blah),
    ]
      @test_throws ErrorException grad(nlp, x)
      @test_throws ErrorException hprod(nlp, x, x)
      @test_throws ErrorException hess(nlp, x)
      @test_throws ErrorException hess_coord(nlp, x)
      @test_throws ErrorException cons(nlp, x)
      @test_throws ErrorException jac(nlp, x)
      @test_throws ErrorException jprod(nlp, x, x)
      @test_throws ErrorException jtprod(nlp, x, y)
      @test_throws ErrorException jac_coord(nlp, x)
      @test_throws ErrorException hprod(nlp, x, y, x)
      @test_throws ErrorException hess(nlp, x, y)
      @test_throws ErrorException hess_coord(nlp, x, y)
    end
  end

  @testset "NLPModel with single argument fails for everything else" begin
    for (funsym, funarg) in [
      (:grad, blah),
      (:hprod, blah),
      (:hess_coord, (Int[], Int[], blah)),
      (:cons, (blah, y, y)),
      (:jprod, blah),
      (:jtprod, blah),
      (:jac_coord, (Int[], Int[], blah)),
    ]
      for nlp in [
        NLPModel(x, blah; funsym => funarg),
        NLPModel(x, -ones(2), ones(2), blah; funsym => funarg),
      ]
        obj(nlp, x)
        if funsym != :grad @test_throws ErrorException grad(nlp, x) else grad(nlp, x) end
        if funsym != :hprod @test_throws ErrorException hprod(nlp, x, x) else hprod(nlp, x, x) end
        if funsym != :hprod @test_throws ErrorException hprod(nlp, x, y, x) else hprod(nlp, x, y, x) end
        if funsym != :hess_coord @test_throws ErrorException hess(nlp, x) else hess(nlp, x) end
        if funsym != :hess_coord @test_throws ErrorException hess(nlp, x, y) else hess(nlp, x, y) end
        if funsym != :cons @test_throws ErrorException cons(nlp, x) else cons(nlp, x) end
        if funsym != :jprod @test_throws ErrorException jprod(nlp, x, x) else jprod(nlp, x, x) end
        if funsym != :jtprod @test_throws ErrorException jtprod(nlp, x, y) else jtprod(nlp, x, y) end
        if funsym != :jac_coord @test_throws ErrorException jac(nlp, x) else jac(nlp, x) end
      end
    end
  end
end