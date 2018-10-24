using TNTensors, Test
using TensorOperations: @tensor, tensoradd, tensortrace, tensorcontract, scalar
using Random: randperm
const ntype = ComplexF64
const u1chs = -1:1
const u1ds = [2,1,2]
testequality(fun, tensor) = @test todense(fun(tensor)) ≈ fun(todense(tensor))
testequality(fun) = x -> testequality(fun, x)
randu(io::NTuple{N}) where N = rand(U1Tensor{ntype,N}, ntuple(x -> u1chs, N), ntuple(x -> u1ds, N), io)
randz(io::NTuple{N}) where N = rand(ZNTensor{ntype,N,2}, ntuple(x -> [1,2], N), io)
u1tensors = map(randu, (fill(1,i)...,fill(-1,i)...) for i in 1:4)
zntensors = map(randz, (fill(1,i)...,fill(-1,i)...) for i in 1:4)

testaddition(x::AbstractTensor{T,N}) where {T,N} = tensoradd(x, ntuple(identity,N), x, ntuple(identity,N))
testtrace(x::AbstractTensor{T,N}) where {T,N} =
    tensortrace(x, (ntuple(identity, div(N,2))...,
                    ntuple(identity, div(N,2))...))
testptrace(x::AbstractTensor{T,N}) where {T,N} =
    tensortrace(x, (1,(2:N-1)...,1))
testcontractblas(x::AbstractTensor{T,N}) where {T,N} =
    tensorcontract(x,collect(1:N),x,collect(N:-1:1))
testcontractnative(x::AbstractTensor{T,N}) where {T,N} =
    tensorcontract(x,collect(1:N),x,collect(N:-1:1),method=:native)

@testset "U1Tensor" begin
    map(testequality(testaddition), u1tensors)
    map(testequality(testtrace), u1tensors)
    map(testequality(testptrace), u1tensors)
    map(testequality(testcontractblas), u1tensors)
    map(testequality(testcontractnative), u1tensors)
end

@testset "ZNTensor" begin
    map(testequality(testaddition), zntensors)
    map(testequality(testtrace), zntensors)
    map(testequality(testptrace), zntensors)
    map(testequality(testcontractblas), zntensors)
    map(testequality(testcontractnative), zntensors)
end

@testset "SplitFuse" begin
    au1 = rand(U1Tensor{ComplexF64,3}, (-2:2,-1:1,-2:2),
        ([3,2,1,4,5],[2,1,3],[3,2,1,2,3]),(1,1,-1))
    bu1, = fuselegs(au1,((1,2),3),(1,-1));
    cu1, = fuselegs(au1,(1,(2,3)),(-1,-1));
    du1, = fuselegs(au1,((3,2),1),(1,-1));
    eu1, = fuselegs(du1,((2,1),),(1,));
    @tensor begin
         ra[] := au1[1,2,3] * au1'[1,2,3]
         rb[] := bu1[1,2] * bu1'[1,2]
         rc[] := cu1[1,2] * cu1'[1,2]
         rd[] := du1[1,2] * du1'[1,2]
         re[] := eu1[1] * eu1'[1]
    end;
    @test scalar(ra) ≈ scalar(rb) ≈ scalar(rc) ≈ scalar(rd) ≈ scalar(re)

    az2 = rand(ZNTensor{ComplexF64,3,2}, (0:1,0:1,0:1),
        ([1,3],[4,2],[5,2]),(1,1,-1))
    bz2, = fuselegs(az2,((1,2),3),(1,1));
    cz2, = fuselegs(az2,(1,(2,3)),(1,1));
    dz2, = fuselegs(az2,((3,2),1),(1,-1));
    ez2, = fuselegs(dz2,((2,1),),(1,));
    @tensor begin
         ra[] := az2[1,2,3] * az2'[1,2,3]
         rb[] := bz2[1,2] * bz2'[1,2]
         rc[] := cz2[1,2] * cz2'[1,2]
         rd[] := dz2[1,2] * dz2'[1,2]
         re[] := ez2[1] * ez2'[1]
    end;
    @test scalar(ra) ≈ scalar(rb) ≈ scalar(rc) ≈ scalar(rd) ≈ scalar(re)

    az3 = rand(ZNTensor{ComplexF64,3,3}, (0:2,0:2,0:2),
        ([1,3,6],[4,2,2],[5,2,3]),(1,1,-1))
    bz3, = fuselegs(az3,((1,2),3),(1,1));
    cz3, = fuselegs(az3,(1,(2,3)),(1,1));
    dz3, = fuselegs(az3,((3,2),1),(1,-1));
    ez3, = fuselegs(dz3,((2,1),),(1,));
    @tensor begin
         ra[] := az3[1,2,3] * az3'[1,2,3]
         rb[] := bz3[1,2] * bz3'[1,2]
         rc[] := cz3[1,2] * cz3'[1,2]
         rd[] := dz3[1,2] * dz3'[1,2]
         re[] := ez3[1] * ez3'[1]
    end;
    @test scalar(ra) ≈ scalar(rb) ≈ scalar(rc) ≈ scalar(rd) ≈ scalar(re)
end

@testset "splitting" begin
    a = rand(U1Tensor{ComplexF64,3},
            (-1:1,-1:1,-1:1),
            ([2,1,3],[4,1,2],[2,5,2]),
            (-1,1,-1));
    b, inverter = fuselegs(a, (1,(2,3)), (-1,-1));
    a2 = splitlegs(b, (1,(2,2,1),(2,2,2)), inverter)
    @test a2 == a
    b, inverter = fuselegs(a, (1, (3,2)), (1,-1));
    @test a == splitlegs(b, ((1,1,1),(2,2,2),(2,2,1)), inverter)

    a = rand(U1Tensor{ComplexF64,2}, (-2:2,-1:1), ([4,5,1,2,3],[6,1,2]), (1,-1));
    b, inverter = fuselegs(a, (1,2), (1,1));
    @test a == splitlegs(b, ((1,1,1),(2,2,1)), inverter)
    b, inverter = fuselegs(a, (2,1), (1,1));
    @test a == splitlegs(b, ((2,2,1),(1,1,1)), inverter)
    splitlegs(splitlegs(b, (2,1), inverter) , (2,1), inverter) == b

    a = rand(ZNTensor{ComplexF64,3,3},
            (0:2,0:2,0:2),
            ([1,8,3],[3,2,3],[1,2,6]),
            (-1,1,-1));
    b, inverter = fuselegs(a, ((1,2),3), (1,-1));
    @test a == splitlegs(b, ((1,1,1),(1,1,2),2), inverter)
    b, inverter = fuselegs(a, (1,(2,3)), (-1,-1));
    @test a == splitlegs(b, (1,(2,2,1),(2,2,2)), inverter)
    b, inverter = fuselegs(a, (1, (3,2)), (-1,-1));
    @test a == splitlegs(b, (1,(2,2,2),(2,2,1)), inverter)

    a = rand(U1Tensor{ComplexF64,2}, (-2:2,-1:1), ([3,2,1,2,3],[2,1,2]), (1,-1));
    b, inverter = fuselegs(a, (1,2), (1,1));
    @test splitlegs(splitlegs(b, (2,1), inverter) , (2,1), inverter) == b


    chs = -30:30
    ds = [5 for c in chs]
    a = rand(U1Tensor{ComplexF64,3},
            (chs,chs,chs),
            (ds,ds,ds),
            (-1,1,-1));
    b, inverter = fuselegs(a, (1,(2,3)), (-1,-1));
    a2 = splitlegs(b, (1,(2,2,1),(2,2,2)), inverter)
    @test a == a2
end

@testset "TensorSVD" begin
    chs = -3:3
    ds = [4 for i in chs]
    ntype = ComplexF64
    au1= rand(U1Tensor{ntype,2}, (chs, chs), (ds, ds), (1,-1))
    u, s, vt = tensorsvd(au1)
    @tensor au12[1,2] := u[1,-1] * s[-1, -2] * vt[-2,2]
    @test au12 ≈ au1

    au1= rand(U1Tensor{ntype,2}, (chs, chs), (ds, ds), (-1,-1))
    u, s, vt = tensorsvd(au1)
    @tensor au12[1,2] := u[1,-1] * s[-1, -2] * vt[-2,2]
    @test au12 ≈ au1

    az2 = rand(ZNTensor{ntype,2,2}, ntuple(x -> [1,2], 2), (1,-1))
    u, s, vt = tensorsvd(az2)
    @tensor az22[1,2] := u[1,-1] * s[-1, -2] * vt[-2,2]
    @test az22 ≈ az2

    az2 = rand(ZNTensor{ntype,2,2}, ntuple(x -> [1,2], 2), (-1,-1))
    u, s, vt = tensorsvd(az2)
    @tensor az22[1,2] := u[1,-1] * s[-1, -2] * vt[-2,2]
    @test az22 ≈ az2

    az3= rand(ZNTensor{ntype,2,3}, ntuple(x -> [1,2,3], 2), (1,1))
    u, s, vt = tensorsvd(az3)
    @tensor az32[1,2] := u[1,-1] * s[-1, -2] * vt[-2,2]
    @test az32 ≈ az3
end

@testset "in_out agnosticism" begin
    for io in [(1,1),(1,-1),(-1,1),(-1,-1)]
        a = rand(U1Tensor{ComplexF64,4}, (-1:1,-1:1,-1:1,-1:1), ([2,1,2],[2,1,2],[2,1,2],[2,1,2]), (1,1,-1,-1));
        @tensor ref[] := a[1,2,1,2]
        io = (1,-1)
        b = fuselegs(a, ((1,2),(3,4)), io)[1]
        @tensor c[] := b[1,1]
        @test c ≈ ref
    end

    N = 6
    a = rand(U1Tensor{ComplexF64,N}, tuple([-1:1 for i in 1:N]...), tuple([[2,1,2] for i in 1:N]...), tuple([1 for i in 1:N]...));
    b = rand(U1Tensor{ComplexF64,N}, tuple([-1:1 for i in 1:N]...), tuple([[2,1,2] for i in 1:N]...), tuple([-1 for i in 1:N]...));

    ref = tensorcontract(a,1:N,b,1:N,())
    for i in rand(2:N-1,1)
        for ioa in [(1,1),(1,-1),(-1,1),(-1,-1)], iob in [(1,1),(1,-1),(-1,1),(-1,-1)]
            inds = randperm(N)
            i1 = (inds[1:i]...,)
            i2 = (inds[i+1:end]...,)
            af = fuselegs(a,(i1,i2),ioa)[1]
            bf = fuselegs(b,(i1,i2),iob)[1]
            @test scalar(tensorcontract(af,1:2,bf,1:2,())) ≈ scalar(ref)
        end
    end

    N = 3
    a = rand(U1Tensor{ComplexF64,N}, tuple([-1:1 for i in 1:N]...), tuple([[2,1,2] for i in 1:N]...), tuple([1 for i in 1:N]...));
    b = rand(U1Tensor{ComplexF64,N}, tuple([-1:1 for i in 1:N]...), tuple([[2,1,2] for i in 1:N]...), tuple([-1 for i in 1:N]...));

    ref = tensorcontract(a,1:N,b,1:N,())
    for ioa in Iterators.product([(1,-1) for i in 1:N]...),
        iob in Iterators.product([(1,-1) for i in 1:N]...)
        inds = randperm(N)
        af = fuselegs(a,inds,ioa)[1]
        bf = fuselegs(b,inds,iob)[1]
        @test scalar(tensorcontract(af,1:N,bf,1:N,())) ≈ scalar(ref)
    end
    for io in [(1,1),(1,-1),(-1,1),(-1,-1)]
        a = rand(ZNTensor{ComplexF64,4,2}, (0:1,0:1,0:1,0:1), ([2,2],[2,2],[2,2],[2,2]), (1,1,-1,-1));
        @tensor ref[] := a[1,2,1,2]
        io = (1,-1)
        b = fuselegs(a, ((1,2),(3,4)), io)[1]
        @tensor c[] := b[1,1]
        @test c ≈ ref
    end

    N = 6
    a = rand(ZNTensor{ComplexF64,N,2}, tuple([0:1 for i in 1:N]...), tuple([[2,2] for i in 1:N]...), tuple([1 for i in 1:N]...));
    b = rand(ZNTensor{ComplexF64,N,2}, tuple([0:1 for i in 1:N]...), tuple([[2,2] for i in 1:N]...), tuple([-1 for i in 1:N]...));

    ref = tensorcontract(a,1:N,b,1:N,())
    for i in rand(2:N-1,1)
        for ioa in [(1,1),(1,-1),(-1,1),(-1,-1)], iob in [(1,1),(1,-1),(-1,1),(-1,-1)]
            inds = randperm(N)
            i1 = (inds[1:i]...,)
            i2 = (inds[i+1:end]...,)
            af = fuselegs(a,(i1,i2),ioa)[1]
            bf = fuselegs(b,(i1,i2),iob)[1]
            @test scalar(tensorcontract(af,1:2,bf,1:2,())) ≈ scalar(ref)
        end
    end

    N = 3
    a = rand(ZNTensor{ComplexF64,N,2}, tuple([0:1 for i in 1:N]...), tuple([[2,2] for i in 1:N]...), tuple([1 for i in 1:N]...));
    b = rand(ZNTensor{ComplexF64,N,2}, tuple([0:1 for i in 1:N]...), tuple([[2,2] for i in 1:N]...), tuple([-1 for i in 1:N]...));

    ref = tensorcontract(a,1:N,b,1:N,())
    for ioa in Iterators.product([(1,-1) for i in 1:N]...),
        iob in Iterators.product([(1,-1) for i in 1:N]...)
        inds = randperm(N)
        af = fuselegs(a,inds,ioa)[1]
        bf = fuselegs(b,inds,iob)[1]
        @test scalar(tensorcontract(af,1:N,bf,1:N,())) ≈ scalar(ref)
    end
end
