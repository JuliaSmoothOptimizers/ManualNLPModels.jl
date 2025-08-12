@testset "Not implemented tests" begin
  blah(args...; kwargs...) = args[1]
  x = zeros(2)
  y = zeros(0)

  @testset "Minimum NLPModel throws for everything else" begin
    for nlp in [NLPModel(x, blah), NLPModel(x, -ones(2), ones(2), blah)]
      @test_throws ErrorException grad(nlp, x)
      @test_throws ErrorException hprod(nlp, x, x)
      @test_throws ErrorException hess(nlp, x)
      @test_throws ErrorException hess_coord(nlp, x)
      @test_throws ErrorException cons_nln(nlp, x)
      @test_throws ErrorException jac_nln(nlp, x)
      @test_throws ErrorException jprod_nln(nlp, x, x)
      @test_throws ErrorException jtprod_nln(nlp, x, y)
      @test_throws ErrorException jac_nln_coord(nlp, x)
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
        #! format: off
        if funsym != :grad        @test_throws ErrorException  grad(nlp, x)          else  grad(nlp, x)          end
        if funsym != :hprod       @test_throws ErrorException  hprod(nlp, x, x)      else  hprod(nlp, x, x)      end
        if funsym != :hprod       @test_throws ErrorException  hprod(nlp, x, y, x)   else  hprod(nlp, x, y, x)   end
        if funsym != :hess_coord  @test_throws ErrorException  hess(nlp, x)          else  hess(nlp, x)          end
        if funsym != :hess_coord  @test_throws ErrorException  hess(nlp, x, y)       else  hess(nlp, x, y)       end
        if funsym != :cons        @test_throws ErrorException  cons_nln(nlp, x)      else  cons_nln(nlp, x)      end
        if funsym != :jprod       @test_throws ErrorException  jprod_nln(nlp, x, x)  else  jprod_nln(nlp, x, x)  end
        if funsym != :jtprod      @test_throws ErrorException  jtprod_nln(nlp, x, y) else  jtprod_nln(nlp, x, y) end
        if funsym != :jac_coord   @test_throws ErrorException  jac_nln(nlp, x)       else  jac_nln(nlp, x)       end
        #! format: on
      end
    end
  end

  blah_resid(r, x) = r .= x

  @testset "Minimum NLSModel throws for everything else" begin
    for nls in [NLSModel(x, blah_resid, 2)]
      r = residual(nls, nls.meta.x0)
      @test all(r .== nls.meta.x0)

      # grad is not implemented because jtprod_residual is not implemented by default
      @test_throws ErrorException grad(nls, x)

      # NLS method supported but not implemented by default
      @test_throws ErrorException jac_residual(nls, x)
      @test_throws ErrorException jac_coord_residual(nls, x)
      @test_throws ErrorException jprod_residual(nls, x, x)
      @test_throws ErrorException jtprod_residual(nls, x, x)

      # NLP methods not currently implemented
      @test_throws MethodError hprod(nls, x, x)
      @test_throws MethodError hess(nls, x)
      @test_throws MethodError hess_coord(nls, x)
      @test_throws MethodError cons_nln(nls, x)
      @test_throws MethodError jac_nln(nls, x)
      @test_throws MethodError jprod_nln(nls, x, x)
      @test_throws MethodError jtprod_nln(nls, x, y)
      @test_throws MethodError jac_nln_coord(nls, x)
      @test_throws MethodError hprod(nls, x, y, x)
      @test_throws MethodError hess(nls, x, y)
      @test_throws MethodError hess_coord(nls, x, y)

      # NLS methods not currently implemented
      @test_throws MethodError hess_residual(nls, x)
      @test_throws MethodError hess_structure_residual(nls)
      @test_throws MethodError hess_coord_residual(nls, x)
      @test_throws MethodError hprod_residual(nls, x, x)
      @test_throws MethodError jth_hess_residual(nls, x)

      # TODO: Enable this test when jth_hess_residual_coord is implemented in NLPModels.
      # @test_throws MethodError jth_hess_residual_coord(nls, x, 2)
    end
  end
end
