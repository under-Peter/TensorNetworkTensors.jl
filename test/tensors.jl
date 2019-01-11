# using TNTensors
using Test
using TensorOperations: @tensor
using LinearAlgebra

@testset "U1Tensor" begin
    u1chs, u1ds, io = U1Charges(-2:2), fill(2,5), InOut(-1,1)
    @testset "Initialization" begin
        a = DASTensor{Float64,2}(U1(), (u1chs, u1chs), (u1ds, u1ds), io)

        @test symmetry(a) == U1()
        @test chargetype(a) == chargetype(U1())
        @test chargestype(a) == chargestype(U1())
        @test eltype(a) == Float64
        @test isempty(tensor(a))


        #Initialize with zeros
        b0 = initwithzero!(deepcopy(a))
        @test isinvariant(b0)
        @tensor bs = b0[1,1]

        @test charge(b0) == zero(U1Charge)
        @test bs == zero(Float64)


        #Initialize with rand
        br = initwithrand!(deepcopy(a))
        bd = todense(br)
        @tensor begin
            sc = br[1,1]
            scd = bd[1,1]
        end

        @test sc isa Float64
        @test scd isa Float64
        @test sc ≈ scd
    end

    @testset "TensorOperations" begin
        f = DASTensor{Complex{Float64},3}(U1(), ntuple(x -> u1chs, 3),
                 ntuple(x -> u1ds, 3), InOut(-1,-1,1))
        fr = initwithrand!(deepcopy(f))
        ##inner product direct
        @tensor ref = fr[1,2,3] * fr'[1,2,3]
        ##inner product direct with permutation
        @test ref ≈ @tensor fr'[2,1,3] * fr[2,1,3]
        ##outer product then trace
        @test ref ≈ @tensor begin
            fr2[1,2,3,4,5,6] := fr[1,2,3] * fr'[4,5,6]
            fr2[1,2,3,1,2,3]
        end
        ##partial contraction then partial trace
        @test ref ≈ @tensor begin
            fr3[1,2,3,4] := fr[1,2,-1] * fr'[3,4,-1]
            fr3[1,2,1,2]
        end
    end

    @testset "KrylovKit" begin
        #compare operations from krylovkit
        f = DASTensor{Complex{Float64},3}(U1(), ntuple(x -> u1chs, 3),
                 ntuple(x -> u1ds, 3), InOut(-1,-1,1))
        fr = initwithrand!(deepcopy(f))

        ##fill!
        fr0 = fill!(deepcopy(fr), 0)
        @test all(iszero(v) for v in values(fr0))

        ##mul! -> w = a .* v
        fr2 = deepcopy(fr)
        mul!(fr2, fr, 2)
        @test fr2 ≈ fr * 2
        mul!(fr2, fr, 1/2)
        @test fr2 ≈ fr/2
        mul!(fr2, fr, 1im)
        @test fr2 ≈ fr * 1im

        ##rmul! w = a .* w
        fr2 = deepcopy(fr)
        rmul!(fr2, 2)
        @test fr2 ≈ fr * 2
        rmul!(fr2, 0)
        @test all(iszero(v) for v in values(fr2))

        ##axpy! a,v,w -> w = a * v + w
        fr2 = deepcopy(fr)
        axpy!(1,fr, fr2)
        @test fr2 ≈ 2 * fr
        axpy!(0,fr, fr2)
        @test fr2 ≈ 2 * fr

        ##axpby! a,v,b,w -> w = a * v + b * w
        fr2 = deepcopy(fr)
        axpby!(1,fr,1,fr2)
        @test fr2 ≈ 2 * fr
        initwithrand!(fr2)
        axpby!(1,fr,0,fr2)
        @test fr2 ≈ fr
        initwithrand!(fr2)
        @tensor frsum[1,2,3] := fr[1,2,3] + fr2[1,2,3]
        axpby!(1,fr,1,fr2)
        @test fr2 ≈ frsum

        ##dot, norm
        f = DASTensor{Complex{Float64},3}(U1(), ntuple(x -> u1chs, 3),
                 ntuple(x -> u1ds, 3), InOut(-1,-1,1))
        fr = initwithrand!(deepcopy(f))
        @tensor ref = fr[1,2,3] * fr'[1,2,3]
        @test ref ≈ dot(fr,fr)
        @test ref ≈ norm(fr)^2
        nfr = fr / norm(fr)
        @test norm(nfr) ≈ 1
        @tensor ref = nfr[1,2,3] * conj(nfr)[1,2,3]
    end

    @testset "TensorFactorization" begin
        ##w/o splitting and fusing
        ar = DASTensor{Float64,2}(U1(), (u1chs, u1chs), (u1ds, u1ds), io)
        initwithrand!(ar)
        ua, sa, vda = tensorsvd(ar)
        @tensor ar2[1,2] := ua[1,-1] * sa[-1,-2] * vda[-2,2]
        @test ar2 ≈ ar
        @test sum(diag(sa)) ≈ @tensor sa[1,1]
        @tensor sa2[1,2] := ua'[-1,1] * ar[-1,-2] * vda'[2,-2]
        @test sa2 ≈ sa

        ##with splitting and fusing
        f = DASTensor{Complex{Float64},3}(U1(), ntuple(x -> u1chs, 3),
                 ntuple(x -> u1ds, 3), InOut(-1,-1,1))
        fr = initwithrand!(deepcopy(f))
        uf, sf, vdf = tensorsvd(fr,((1,2),3))
        @tensor fr2[1,2,3] := uf[1,2,-1] * sf[-1,-2] * vdf[-2,3]
        @test fr2 ≈ fr

        ## with charge
        ar = DASTensor{Float64,2}(U1(), (u1chs, u1chs), (u1ds, u1ds), io)
        initwithrand!(ar, U1Charge(1))
        @test charge(ar) == U1Charge(1)
        ua, sa, vda = tensorsvd(ar)
        @tensor ar2[1,2] := ua[1,-1] * sa[-1,-2] * vda[-2,2]
        @test ar2 ≈ ar
        @test sum(diag(sa)) ≈ @tensor sa[1,1]
        @tensor sa2[1,2] := ua'[-1,1] * ar[-1,-2] * vda'[2,-2]
        @test sa2 ≈ sa

        ## with truncation
        ar = DASTensor{Float64,2}(U1(), (U1Charges(0:0), U1Charges(0:0)), ([200], [200]), io)
        initwithrand!(ar)
        ua, sa, vda = tensorsvd(ar, svdtrunc = svdtrunc_maxerror(0.001))
        @test all(sizes(sa,1) .<= 200)

        @tensor ar2[1,2] := ua[1,-1] * sa[-1,-2] * vda[-2,2]
        ar2 /= norm(ar2)
        ar /= norm(ar)
        @test dot(ar2,ar) > 1-0.001

        ua, sa, vda = tensorsvd(ar, svdtrunc = svdtrunc_maxcumerror(0.001))
        @test all(sizes(sa,1) .<= 200)

        @tensor ar2[1,2] := ua[1,-1] * sa[-1,-2] * vda[-2,2]
        ar2 /= norm(ar2)
        ar /= norm(ar)
        @test dot(ar2,ar) > 1-0.001
    end

    @testset "Reshaping" begin
        ##without permutation
        f = DASTensor{Complex{Float64},6}(U1(), ntuple(x -> u1chs, 6),
                 ntuple(x -> u1ds, 6), InOut(-1,-1,-1,1,1,1))
        fr = initwithrand!(deepcopy(f))
        frf, rs2 = fuselegs(fr, ((1,2),(3,4),(5,6)))
        frff, rs3 = fuselegs(frf, ((1,2,3),))
        @test abs(norm(frf)  - norm(fr)) < 1e-10
        @test abs(norm(frff) - norm(fr)) < 1e-10

        frffs = initwithzero!(deepcopy(frf))
        frffss = initwithzero!(deepcopy(fr))
        splitlegs!(frffs,  frff,  ((1,1,1),(1,1,2),(1,1,3)), rs3...)
        splitlegs!(frffss, frffs, ((1,1,1),(1,1,2),(2,2,1),(2,2,2),(3,3,1),(3,3,3)), rs2...)
        @test frffss ≈ fr

        ##with permutation
        f = DASTensor{Complex{Float64},4}(U1(), ntuple(x -> u1chs, 4),
                 ntuple(x -> u1ds, 4), InOut(-1,-1,1,1))
        fr = initwithrand!(deepcopy(f))
        frf, rs = fuselegs(fr, ((1,3),(2,4)))
        @test abs(norm(frf)  - norm(fr)) < 1e-10

        frfs = initwithzero!(deepcopy(fr))
        splitlegs!(frfs,  frf,  ((1,1,1),(2,2,1),(1,1,2),(2,2,2)), rs...)
        @test frfs ≈ fr
    end
end

@testset "Z2" begin
    z2chs, z2ds, io = Z2Charges(), fill(2,2), InOut(-1,1)
    @testset "Initialization" begin
        a = DASTensor{Float64,2}(Z2(), (z2chs, z2chs), (z2ds, z2ds), io)
        @test symmetry(a) == Z2()
        @test chargetype(a) == chargetype(Z2())
        @test chargestype(a) == chargestype(Z2())
        @test eltype(a) == Float64
        @test isempty(tensor(a))


        #Initialize with zeros
        b0 = initwithzero!(deepcopy(a))
        @tensor bs = b0[1,1]
        @test isinvariant(b0)

        @test charge(b0) == zero(Z2Charge)
        @test bs == zero(Float64)


        #Initialize with rand
        br = initwithrand!(deepcopy(a))
        bd = todense(br)
        @tensor begin
            sc = br[1,1]
            scd = bd[1,1]
        end

        @test sc isa Float64
        @test scd isa Float64
        @test sc ≈ scd
    end

    @testset "TensorOperations" begin
        f = DASTensor{Complex{Float64},3}(Z2(), ntuple(x -> z2chs, 3),
                 ntuple(x -> z2ds, 3), InOut(-1,-1,1))
        fr = initwithrand!(deepcopy(f))
        ##inner product direct
        @tensor ref = fr[1,2,3] * fr'[1,2,3]
        ##inner product direct with permutation
        @test ref ≈ @tensor fr'[2,1,3] * fr[2,1,3]
        ##outer product then trace
        @test ref ≈ @tensor begin
            fr2[1,2,3,4,5,6] := fr[1,2,3] * fr'[4,5,6]
            fr2[1,2,3,1,2,3]
        end
        ##partial contraction then partial trace
        @test ref ≈ @tensor begin
            fr3[1,2,3,4] := fr[1,2,-1] * fr'[3,4,-1]
            fr3[1,2,1,2]
        end
    end

    @testset "KrylovKit" begin
        #compare operations from krylovkit
        f = DASTensor{Complex{Float64},3}(Z2(), ntuple(x -> z2chs, 3),
                 ntuple(x -> z2ds, 3), InOut(-1,-1,1))
        fr = initwithrand!(deepcopy(f))

        ##fill!
        fr0 = fill!(deepcopy(fr), 0)
        @test all(iszero(v) for v in values(fr0))

        ##mul! -> w = a .* v
        fr2 = deepcopy(fr)
        mul!(fr2, fr, 2)
        @test fr2 ≈ fr * 2
        mul!(fr2, fr, 1/2)
        @test fr2 ≈ fr/2
        mul!(fr2, fr, 1im)
        @test fr2 ≈ fr * 1im

        ##rmul! w = a .* w
        fr2 = deepcopy(fr)
        rmul!(fr2, 2)
        @test fr2 ≈ fr * 2
        rmul!(fr2, 0)
        @test all(iszero(v) for v in values(fr2))

        ##axpy! a,v,w -> w = a * v + w
        fr2 = deepcopy(fr)
        axpy!(1,fr, fr2)
        @test fr2 ≈ 2 * fr
        axpy!(0,fr, fr2)
        @test fr2 ≈ 2 * fr

        ##axpby! a,v,b,w -> w = a * v + b * w
        fr2 = deepcopy(fr)
        axpby!(1,fr,1,fr2)
        @test fr2 ≈ 2 * fr
        initwithrand!(fr2)
        axpby!(1,fr,0,fr2)
        @test fr2 ≈ fr
        initwithrand!(fr2)
        @tensor frsum[1,2,3] := fr[1,2,3] + fr2[1,2,3]
        axpby!(1,fr,1,fr2)
        @test fr2 ≈ frsum

        ##dot, norm
        f = DASTensor{Complex{Float64},3}(Z2(), ntuple(x -> z2chs, 3),
                 ntuple(x -> z2ds, 3), InOut(-1,-1,1))
        fr = initwithrand!(deepcopy(f))
        @tensor ref = fr[1,2,3] * fr'[1,2,3]
        @test ref ≈ dot(fr,fr)
        @test ref ≈ norm(fr)^2
        nfr = fr / norm(fr)
        @test norm(nfr) ≈ 1
    end

    @testset "TensorFactorization" begin
        ##w/o splitting and fusing
        ar = DASTensor{Float64,2}(Z2(), (z2chs, z2chs), (z2ds, z2ds), io)
        initwithrand!(ar)
        ua, sa, vda = tensorsvd(ar)
        @tensor ar2[1,2] := ua[1,-1] * sa[-1,-2] * vda[-2,2]
        @test ar2 ≈ ar
        @test sum(diag(sa)) ≈ @tensor sa[1,1]
        @tensor sa2[1,2] := ua'[-1,1] * ar[-1,-2] * vda'[2,-2]
        @test sa2 ≈ sa

        ##with splitting and fusing
        f = DASTensor{Complex{Float64},3}(Z2(), ntuple(x -> z2chs, 3),
                 ntuple(x -> z2ds, 3), InOut(-1,-1,1))
        fr = initwithrand!(deepcopy(f))
        uf, sf, vdf = tensorsvd(fr,((1,2),3))
        @tensor fr2[1,2,3] := uf[1,2,-1] * sf[-1,-2] * vdf[-2,3]
        @test fr2 ≈ fr

        ## with charge
        ar = DASTensor{Float64,2}(Z2(), (z2chs, z2chs), (z2ds, z2ds), io)
        initwithrand!(ar, Z2Charge(1))
        @test charge(ar) == Z2Charge(1)
        ua, sa, vda = tensorsvd(ar)
        @tensor ar2[1,2] := ua[1,-1] * sa[-1,-2] * vda[-2,2]
        @test ar2 ≈ ar
        @test sum(diag(sa)) ≈ @tensor sa[1,1]
        @tensor sa2[1,2] := ua'[-1,1] * ar[-1,-2] * vda'[2,-2]
        @test sa2 ≈ sa

        ## with truncation
        ar = DASTensor{Float64,2}(Z2(), (Z2Charges(), Z2Charges()), ([200,200], [200,200]), io)
        initwithrand!(ar)
        ua, sa, vda = tensorsvd(ar, svdtrunc = svdtrunc_maxerror(0.001))
        @test all(sizes(sa,1) .<= 200)

        @tensor ar2[1,2] := ua[1,-1] * sa[-1,-2] * vda[-2,2]
        ar2 /= norm(ar2)
        ar /= norm(ar)
        @test dot(ar2,ar) > 1-0.001

        ua, sa, vda = tensorsvd(ar, svdtrunc = svdtrunc_maxcumerror(0.001))
        @test all(sizes(sa,1) .<= 200)

        @tensor ar2[1,2] := ua[1,-1] * sa[-1,-2] * vda[-2,2]
        ar2 /= norm(ar2)
        ar /= norm(ar)
        @test dot(ar2,ar) > 1-0.001
    end

    @testset "Reshaping" begin
        ##without permutation
        f = DASTensor{Complex{Float64},6}(Z2(), ntuple(x -> z2chs, 6),
                 ntuple(x -> z2ds, 6), InOut(-1,-1,-1,1,1,1))
        fr = initwithrand!(deepcopy(f))
        frf, rs2 = fuselegs(fr, ((1,2),(3,4),(5,6)))
        frff, rs3 = fuselegs(frf, ((1,2,3),))
        @test abs(norm(frf)  - norm(fr)) < 1e-10
        @test abs(norm(frff) - norm(fr)) < 1e-10

        frffs = initwithzero!(deepcopy(frf))
        frffss = initwithzero!(deepcopy(fr))
        splitlegs!(frffs,  frff,  ((1,1,1),(1,1,2),(1,1,3)), rs3...)
        splitlegs!(frffss, frffs, ((1,1,1),(1,1,2),(2,2,1),(2,2,2),(3,3,1),(3,3,3)), rs2...)
        @test frffss ≈ fr

        ##with permutation
        f = DASTensor{Complex{Float64},4}(Z2(), ntuple(x -> z2chs, 4),
                 ntuple(x -> z2ds, 4), InOut(-1,-1,1,1))
        fr = initwithrand!(deepcopy(f))
        frf, rs = fuselegs(fr, ((1,3),(2,4)))
        @test abs(norm(frf)  - norm(fr)) < 1e-10

        frfs = initwithzero!(deepcopy(fr))
        splitlegs!(frfs,  frf,  ((1,1,1),(2,2,1),(1,1,2),(2,2,2)), rs...)
        @test frfs ≈ fr
    end
end

@testset "NDAS" begin
    ndaschs, ndasds, io = NDASCharges(Z2Charges(), U1Charges(-2:2)), fill(2,10), InOut(-1,1)
    @testset "Initialization" begin
        a = DASTensor{Float64,2}(NDAS(Z2(),U1()), (ndaschs, ndaschs), (ndasds, ndasds), io)
        @test symmetry(a) == NDAS(Z2(),U1())
        @test chargetype(a) == chargetype(a)
        @test chargestype(a) == chargestype(a)
        @test eltype(a) == Float64
        @test isempty(tensor(a))

        #Initialize with zeros
        b0 = initwithzero!(deepcopy(a))
        @test isinvariant(b0)
        @tensor bs = b0[1,1]

        @test charge(b0) == zero(chargetype(a))
        @test bs == zero(Float64)

        #Initialize with rand
        br = initwithrand!(deepcopy(a))
        bd = todense(br)
        @tensor begin
            sc = br[1,1]
            scd = bd[1,1]
        end

        @test sc isa Float64
        @test scd isa Float64
        @test sc ≈ scd
    end

    @testset "TensorOperations" begin
        f = DASTensor{Complex{Float64},3}(NDAS(Z2(),U1()), ntuple(x -> ndaschs, 3),
                 ntuple(x -> ndasds, 3), InOut(-1,-1,1))
        fr = initwithrand!(deepcopy(f))
        ##inner product direct
        @tensor ref = fr[1,2,3] * fr'[1,2,3]
        ##inner product direct with permutation
        @test ref ≈ @tensor fr'[2,1,3] * fr[2,1,3]
        ##outer product then trace
        @test ref ≈ @tensor begin
            fr2[1,2,3,4,5,6] := fr[1,2,3] * fr'[4,5,6]
            fr2[1,2,3,1,2,3]
        end
        ##partial contraction then partial trace
        @test ref ≈ @tensor begin
            fr3[1,2,3,4] := fr[1,2,-1] * fr'[3,4,-1]
            fr3[1,2,1,2]
        end
    end

    @testset "KrylovKit" begin
        #compare operations from krylovkit
        f = DASTensor{Complex{Float64},3}(NDAS(Z2(),U1()), ntuple(x -> ndaschs, 3),
                 ntuple(x -> ndasds, 3), InOut(-1,-1,1))
        fr = initwithrand!(deepcopy(f))

        ##fill!
        fr0 = fill!(deepcopy(fr), 0)
        @test all(iszero(v) for v in values(fr0))

        ##mul! -> w = a .* v
        fr2 = deepcopy(fr)
        mul!(fr2, fr, 2)
        @test fr2 ≈ fr * 2
        mul!(fr2, fr, 1/2)
        @test fr2 ≈ fr/2
        mul!(fr2, fr, 1im)
        @test fr2 ≈ fr * 1im

        ##rmul! w = a .* w
        fr2 = deepcopy(fr)
        rmul!(fr2, 2)
        @test fr2 ≈ fr * 2
        rmul!(fr2, 0)
        @test all(iszero(v) for v in values(fr2))

        ##axpy! a,v,w -> w = a * v + w
        fr2 = deepcopy(fr)
        axpy!(1,fr, fr2)
        @test fr2 ≈ 2 * fr
        axpy!(0,fr, fr2)
        @test fr2 ≈ 2 * fr

        ##axpby! a,v,b,w -> w = a * v + b * w
        fr2 = deepcopy(fr)
        axpby!(1,fr,1,fr2)
        @test fr2 ≈ 2 * fr
        initwithrand!(fr2)
        axpby!(1,fr,0,fr2)
        @test fr2 ≈ fr
        initwithrand!(fr2)
        @tensor frsum[1,2,3] := fr[1,2,3] + fr2[1,2,3]
        axpby!(1,fr,1,fr2)
        @test fr2 ≈ frsum

        ##dot, norm
        f = DASTensor{Complex{Float64},3}(NDAS(Z2(),U1()), ntuple(x -> ndaschs, 3),
                 ntuple(x -> ndasds, 3), InOut(-1,-1,1))
        fr = initwithrand!(deepcopy(f))
        @tensor ref = fr[1,2,3] * fr'[1,2,3]
        @test ref ≈ dot(fr,fr)
        @test ref ≈ norm(fr)^2
        nfr = fr / norm(fr)
        @test norm(nfr) ≈ 1
    end

    @testset "TensorFactorization" begin
        ##w/o splitting and fusing
        ar = DASTensor{Float64,2}(NDAS(Z2(),U1()), (ndaschs, ndaschs), (ndasds, ndasds), io)
        initwithrand!(ar)
        ua, sa, vda = tensorsvd(ar)
        @tensor ar2[1,2] := ua[1,-1] * sa[-1,-2] * vda[-2,2]
        @test ar2 ≈ ar
        @test sum(diag(sa)) ≈ @tensor sa[1,1]
        @tensor sa2[1,2] := ua'[-1,1] * ar[-1,-2] * vda'[2,-2]
        @test sa2 ≈ sa

        ##with splitting and fusing
        f = DASTensor{Complex{Float64},3}(NDAS(Z2(),U1()), ntuple(x -> ndaschs, 3),
                 ntuple(x -> ndasds, 3), InOut(-1,-1,1))
        fr = initwithrand!(deepcopy(f))
        uf, sf, vdf = tensorsvd(fr,((1,2),3))
        @tensor fr2[1,2,3] := uf[1,2,-1] * sf[-1,-2] * vdf[-2,3]
        @test fr2 ≈ fr

        ## with charge
        ar = DASTensor{Float64,2}(NDAS(Z2(),U1()), (ndaschs, ndaschs), (ndasds, ndasds), io)
        initwithrand!(ar, NDASCharge(Z2Charge(1), U1Charge(1)))
        @test charge(ar) == NDASCharge(Z2Charge(1), U1Charge(1))
        ua, sa, vda = tensorsvd(ar)
        @tensor ar2[1,2] := ua[1,-1] * sa[-1,-2] * vda[-2,2]
        @test ar2 ≈ ar
        @test sum(diag(sa)) ≈ @tensor sa[1,1]
        @tensor sa2[1,2] := ua'[-1,1] * ar[-1,-2] * vda'[2,-2]
        @test sa2 ≈ sa

        ## with truncation
        ar = DASTensor{Float64,2}(NDAS(Z2(),U1()),
            (NDASCharges(Z2Charges(), U1Charges(0:0)),
             NDASCharges(Z2Charges(), U1Charges(0:0))),
             ([200,200], [200,200]), io)
        initwithrand!(ar)
        ua, sa, vda = tensorsvd(ar, svdtrunc = svdtrunc_maxerror(0.001))
        @test all(sizes(sa,1) .<= 200)

        @tensor ar2[1,2] := ua[1,-1] * sa[-1,-2] * vda[-2,2]
        ar2 /= norm(ar2)
        ar /= norm(ar)
        @test dot(ar2,ar) > 1-0.001

        ua, sa, vda = tensorsvd(ar, svdtrunc = svdtrunc_maxcumerror(0.001))
        @test all(sizes(sa,1) .<= 200)

        @tensor ar2[1,2] := ua[1,-1] * sa[-1,-2] * vda[-2,2]
        ar2 /= norm(ar2)
        ar /= norm(ar)
        @test dot(ar2,ar) > 1-0.001
    end

    @testset "Reshaping" begin
        ##without permutation
        f = DASTensor{Complex{Float64},6}(NDAS(Z2(),U1()), ntuple(x -> ndaschs, 6),
                 ntuple(x -> ndasds, 6), InOut(-1,-1,-1,1,1,1))
        fr = initwithrand!(deepcopy(f))
        frf, rs2 = fuselegs(fr, ((1,2),(3,4),(5,6)))
        frff, rs3 = fuselegs(frf, ((1,2,3),))
        @test abs(norm(frf)  - norm(fr)) < 1e-10
        @test abs(norm(frff) - norm(fr)) < 1e-10

        frffs = initwithzero!(deepcopy(frf))
        frffss = initwithzero!(deepcopy(fr))
        splitlegs!(frffs,  frff,  ((1,1,1),(1,1,2),(1,1,3)), rs3...)
        splitlegs!(frffss, frffs, ((1,1,1),(1,1,2),(2,2,1),(2,2,2),(3,3,1),(3,3,3)), rs2...)
        @test frffss ≈ fr

        ##with permutation
        f = DASTensor{Complex{Float64},4}(NDAS(Z2(),U1()), ntuple(x -> ndaschs, 4),
                 ntuple(x -> ndasds, 4), InOut(-1,-1,1,1))
        fr = initwithrand!(deepcopy(f))
        frf, rs = fuselegs(fr, ((1,3),(2,4)))
        @test abs(norm(frf)  - norm(fr)) < 1e-10

        frfs = initwithzero!(deepcopy(fr))
        splitlegs!(frfs,  frf,  ((1,1,1),(2,2,1),(1,1,2),(2,2,2)), rs...)
        @test frfs ≈ fr
    end
end

@testset "DTensor" begin
    @testset "Initialization" begin
        #Initialize with zeros
        a = DTensor{Complex{Float64}}((8,8))
        b0 = initwithzero!(deepcopy(a))
        @tensor bs = b0[1,1]

        @test bs == zero(Complex{Float64})

        #Initialize with rand
        br = initwithrand!(deepcopy(a))
        @tensor sc = br[1,1]
        @test sc isa Complex{Float64}
    end

    @testset "TensorOperations" begin
        f = DTensor{Complex{Float64},3}((4,4,4))
        fr = initwithrand!(deepcopy(f))
        ##inner product direct
        @tensor ref = fr[1,2,3] * fr'[1,2,3]
        ##inner product direct with permutation
        @test ref ≈ @tensor fr'[2,1,3] * fr[2,1,3]
        ##outer product then trace
        @test ref ≈ @tensor begin
            fr2[1,2,3,4,5,6] := fr[1,2,3] * fr'[4,5,6]
            fr2[1,2,3,1,2,3]
        end
        ##partial contraction then partial trace
        @test ref ≈ @tensor begin
            fr3[1,2,3,4] := fr[1,2,-1] * fr'[3,4,-1]
            fr3[1,2,1,2]
        end
    end

    @testset "KrylovKit" begin
        #compare operations from krylovkit
        f = DTensor{Complex{Float64},3}((8,8,8))
        fr = initwithrand!(deepcopy(f))

        ##fill!
        fr0 = fill!(deepcopy(fr), 0)
        @test iszero(fr0.array)

        ##mul! -> w = a .* v
        fr2 = deepcopy(fr)
        mul!(fr2, fr, 2)
        @test fr2 ≈ fr * 2
        mul!(fr2, fr, 1/2)
        @test fr2 ≈ fr/2
        mul!(fr2, fr, 1im)
        @test fr2 ≈ fr * 1im

        ##rmul! w = a .* w
        fr2 = deepcopy(fr)
        rmul!(fr2, 2)
        @test fr2 ≈ fr * 2
        rmul!(fr2, 0)
        @test iszero(fr2.array)

        ##axpy! a,v,w -> w = a * v + w
        fr2 = deepcopy(fr)
        axpy!(1,fr, fr2)
        @test fr2 ≈ 2 * fr
        axpy!(0,fr, fr2)
        @test fr2 ≈ 2 * fr

        ##axpby! a,v,b,w -> w = a * v + b * w
        fr2 = deepcopy(fr)
        axpby!(1,fr,1,fr2)
        @test fr2 ≈ 2 * fr
        initwithrand!(fr2)
        axpby!(1,fr,0,fr2)
        @test fr2 ≈ fr
        initwithrand!(fr2)
        @tensor frsum[1,2,3] := fr[1,2,3] + fr2[1,2,3]
        axpby!(1,fr,1,fr2)
        @test fr2 ≈ frsum

        ##dot, norm
        f = DTensor{Complex{Float64},3}((8,8,8))
        fr = initwithrand!(deepcopy(f))
        @tensor ref = fr[1,2,3] * fr'[1,2,3]
        @test ref ≈ dot(fr,fr)
        @test ref ≈ norm(fr)^2
        nfr = fr / norm(fr)
        @test norm(nfr) ≈ 1
    end

    @testset "TensorFactorization" begin
        ##w/o splitting and fusing
        ar = DTensor{Float64,2}((20,20))
        initwithrand!(ar)
        ua, sa, vda = tensorsvd(ar)
        @tensor ar2[1,2] := ua[1,-1] * sa[-1,-2] * vda[-2,2]
        @test ar2 ≈ ar
        @test sum(diag(sa)) ≈ @tensor sa[1,1]
        @tensor sa2[1,2] := ua'[-1,1] * ar[-1,-2] * vda'[2,-2]
        @test sa2 ≈ sa

        ##with splitting and fusing
        f = DTensor{Complex{Float64},3}((5,5,5))
        fr = initwithrand!(deepcopy(f))
        uf, sf, vdf = tensorsvd(fr,((1,2),3))
        @tensor fr2[1,2,3] := uf[1,2,-1] * sf[-1,-2] * vdf[-2,3]
        @test fr2 ≈ fr

        ## with truncation
        ar = DTensor{Float64,2}((200,200))
        initwithrand!(ar)
        ua, sa, vda = tensorsvd(ar, svdtrunc = svdtrunc_maxerror(0.001))
        @test all(size(sa) .<= 200)

        @tensor ar2[1,2] := ua[1,-1] * sa[-1,-2] * vda[-2,2]
        ar2 /= norm(ar2)
        ar /= norm(ar)
        @test dot(ar2,ar) > 1-0.001

        ua, sa, vda = tensorsvd(ar, svdtrunc = svdtrunc_maxcumerror(0.001))
        @test all(size(sa) .<= 200)

        @tensor ar2[1,2] := ua[1,-1] * sa[-1,-2] * vda[-2,2]
        ar2 /= norm(ar2)
        ar /= norm(ar)
        @test dot(ar2,ar) > 1-0.001
    end

    @testset "Reshaping" begin
        ##without permutation
        f = DTensor{Complex{Float64},6}(ntuple(i -> 2, 6))
        fr = initwithrand!(deepcopy(f))
        frf, rs2 = fuselegs(fr, ((1,2),(3,4),(5,6)))
        frff, rs3 = fuselegs(frf, ((1,2,3),))
        @test abs(norm(frf)  - norm(fr)) < 1e-10
        @test abs(norm(frff) - norm(fr)) < 1e-10

        frffs = initwithzero!(deepcopy(frf))
        frffss = initwithzero!(deepcopy(fr))
        splitlegs!(frffs,  frff,  ((1,1,1),(1,1,2),(1,1,3)), rs3...)
        splitlegs!(frffss, frffs, ((1,1,1),(1,1,2),(2,2,1),(2,2,2),(3,3,1),(3,3,3)), rs2...)
        @test frffss ≈ fr

        ##with permutation
        f = DTensor{Complex{Float64},4}(ntuple(i -> 2, 4))
        fr = initwithrand!(deepcopy(f))
        frf, rs = fuselegs(fr, ((1,3),(2,4)))
        @test abs(norm(frf)  - norm(fr)) < 1e-10

        frfs = initwithzero!(deepcopy(fr))
        splitlegs!(frfs,  frf,  ((1,1,1),(2,2,1),(1,1,2),(2,2,2)), rs...)
        @test frfs ≈ fr
    end
end
