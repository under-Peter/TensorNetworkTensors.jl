"""
    tensorsvd(A::AbstractTensor{T,2}; svdtrunc)
returns the `svd` of an AbstractTensor. `svdtrunc` is a function that has the
singular values as input and returns a number specifying how many of them to keep.
The default `svdtrunc` is `svdtrunc_default` and keeps all of them.

Other options are:
svdtrunc_discardzero
svdtrunc_maxχ
svdtrunc_maxcumerror
svdtrunc_maxerror
"""
function tensorsvd end

function _tensorsvd(A::Array{<:Any,2}; svdtrunc = svdtrunc_default,
        helper::Bool = false)
    F = LA.svd(A)
    svals = F.S
    cutoff = svdtrunc(svals)
    U = F.U[:, 1:cutoff]
    S = LA.diagm(0=>svals[1:cutoff])
    Vt = F.Vt[1:cutoff, :]
    helper && return (U, S, Vt, cutoff)
    return (U, S, Vt)
end

"""
    tensorsvd(A::AbstractTensor, indexes; svdtrunc)
works like `tensorsvd` except that `A` can have arbitrary rank.
`indexes` specifies which indices the fuse for `A` to be a rank-2 tensor
as in `fuselegs`. The tensor is then fused, `tensorsvd` applied and split again.
"""
function tensorsvd(A::AbstractTensor, indexes; svdtrunc = svdtrunc_default)
    fA, rs = fuselegs(A, indexes)
    U, S, Vt = tensorsvd(fA, svdtrunc=svdtrunc)
    li1 = length(indexes[1])
    li2 = length(indexes[2])
    if indexes[1] isa Tuple
        indxs = (ntuple(x -> (1,1,x), li1)..., 2)
        U = splitlegs(U, indxs, rs...)
    end
    if indexes[2] isa Tuple
        indxs = (1, ntuple(x -> (2,2,x), li2)...)
        Vt = splitlegs(Vt, indxs, rs...)
    end
    return (U,S,Vt)
end

svdtrunc_default(x) = length(x)
"""
    svdtrunc_discardzero(s)
return the number of non-zero values in s
"""
svdtrunc_discardzero(x) = length(filter(!iszero, x))

"""
    svdtrunc_maxχ(χ)
return a function that given singular values `s` returns the min of the
length of `s` and χ.
"""
svdtrunc_maxχ(χ) = x -> min(length(x), χ)

"""
    svdtrunc_maxcumerror(ϵ::Real; χ::Int = typemax(Int))
return a function that returns then min of χ and l where l is the number
of singular values that need to be kept to have the truncated sum up to not
more than ϵ.
"""
svdtrunc_maxcumerror(ϵ::Real; χ::Int = typemax(Int)) = x -> _maxcumerror(x, ϵ, χ)

"""
    svdtrunc_maxerror(ϵ::Real; χ::Int = typemax(Int))
return a function that returns then min of χ and l where l is the number
of singular values that need to be kept such that the largest discarded singular
value is below ϵ.
"""
svdtrunc_maxerror(ϵ::Real; χ::Int = typemax(Int)) = x -> _maxerror(x, ϵ, χ)

function _maxcumerror(xs, ϵ, χ)
    acc, lxs = 0, length(xs)
    for i in lxs:-1:1
        acc += xs[i]/xs[1]
        acc >= ϵ && return min(i+1,χ)
    end
    return min(lxs, χ)
end

function _maxerror(xs, ϵ, χ)
    index = findfirst(x -> x/xs[1] < ϵ, xs)
    index == nothing && return min(length(xs), χ)
    return min(index-1, χ)
end

#DTensor
function tensorsvd(A::DTensor{T,2}; svdtrunc = svdtrunc_default) where T
    U, S, Vd = _tensorsvd(A.array, svdtrunc = svdtrunc)
    TT = promote_type(eltype.((U,S,Vd))...)
    return (DTensor(convert(Array{TT}, U)),
            DTensor(convert(Array{TT}, S)),
            DTensor(convert(Array{TT}, Vd)))
end

#DASTensor
function tensorsvd(A::DASTensor{T,N,SYM,CHARGES,SIZES,CHARGE};
                   svdtrunc = svdtrunc_default) where
                   {T,N,SYM,CHARGES,SIZES,CHARGE}
    N == 2 || throw(ArgumentError("SVD only works on rank 2 tensors"))
    lch = connectingcharge(A)
    ld =  sizes(A,2)[[chargeindex(c,charges(A,2)) for c in lch]]

    U  = DASTensor{T,N}(SYM, (charges(A,1), lch),
            deepcopy.((sizes(A,1), ld)),
            in_out(A), charge(A))
    S  = DASTensor{T,N}(SYM, (lch, lch),
            deepcopy.((ld, ld)),
            vcat(inv(in_out(A,2)), in_out(A,2)))
    Vd = DASTensor{T,N}(SYM, (lch, charges(A,2)),
            deepcopy((ld,sizes(A,2))),
            vcat(inv(in_out(A,2)), in_out(A,2)))

    for (k, degen) in tensor(A)
        in, out = k[1], k[2]
        U[DASSector(in, out)], S[DASSector(out, out)], Vd[DASSector(out, out)], cutoff =
                _tensorsvd(degen, svdtrunc = svdtrunc, helper = true)
        U.dims[2][chargeindex(out,charges(U,2))] = cutoff
        S.dims[1][chargeindex(out,charges(S,1))] = cutoff
        S.dims[2][chargeindex(out,charges(S,2))] = cutoff
        Vd.dims[1][chargeindex(out,charges(Vd,1))] = cutoff
    end
    return (U, S, Vd)
end

"""
    connectingcharge(A)
where A is a rank-2 DASTensor,
returns the charges on the second index that can be realized
given the charges on the first index and the charge of A.
"""
function connectingcharge(A::DASTensor{<:Any,2})
    chs1, chs2 = charges(A)
    io = in_out(A)
    io[1] == InOut(-1) && (chs1 = inv(chs1))
    io[2] == InOut( 1) && (chs2 = inv(chs2))
    #                         (1)ch
    #                          ↓
    #canonical form: chs1 →(1)→A→(-1)→ lch = chs1 + ch ∩ chs2
    chs1 += charge(A)
    lch = chs1 ∩ chs2
    io[2] == InOut(1) && return inv(lch)
    return lch
end


"""
    tensorqr(A::AbstractTensor)

returns tensor Q,R such that A = QR and Q is an orthogonal/unitary matrix
and R is an upper triangular matrix.
"""
function tensorqr end

"""
    tensorqr(A::AbstractTensor, inds)

returns the `tensorqr` of `A` fused according to `inds`.
"""
function tensorqr(A::AbstractTensor, inds)
    fA, rs = fuselegs(A, inds)
    Q,R = tensorqr(fA)
    li1, li2 = length.(inds)
    if inds[1] isa Tuple
        indxs = (ntuple(x -> (1,1,x), li1)..., 2)
        Q = splitlegs(Q, indxs, rs...)
    end
    if inds[2] isa Tuple
        indxs = (1, ntuple(x -> (2,2,x), li2)...)
        R = splitlegs(R, indxs, rs...)
    end
    return (Q,R)
end

function tensorqr(a::DTensor{<:Any,2})
    f = LA.qr(a.array)
    return DTensor(Matrix(f.Q)), DTensor(Matrix(f.R))
end

function tensorqr(a::DASTensor{T,2,SYM}) where {T,SYM}
    lch = connectingcharge(a)
    ld =  sizes(a,2)[[chargeindex(c,charges(a,2)) for c in lch]]

    Q = DASTensor{T,2}(SYM, (charges(a,1), lch),
            deepcopy.((sizes(a,1), ld)),
            in_out(a), charge(a))
    R = DASTensor{T,2}(SYM, (lch, charges(a,2)),
            deepcopy((ld,sizes(a,2))),
            vcat(inv(in_out(a,2)), in_out(a,2)))

    for (k, degen) in tensor(a)
        in, out = k[1], k[2]
        Q[DASSector(in, out)], R[DASSector(out, out)] = LA.qr(degen)
    end
    return (Q,R)
end

"""
    tensorrq(A::AbstractTensor)

returns tensor R,Q such that A = RQ and Q is
obeys Q*Q' = 1 and R is triangular.
This is simply a wrapper for `tensorqr`, useful for e.g. MPS canonicalization.
"""
function tensorrq end

"""
    tensorrq(A::AbstractTensor, inds)

returns the `tensorrq` of `A` fused according to `inds`.
"""
function tensorrq(c, inds = ((1,),(2,)))
    q, r = tensorqr(c, reverse(inds))
    nq, nr = ndims.((q,r))
    q = TO.tensorcopy(q, circshift(1:nq,-1),  1:nq)
    r = TO.tensorcopy(r, circshift(1:nr,1), 1:nr)
    return r, q
end

function tensoreig end

function tensoreig(A::AbstractTensor, indexes; truncfun = svdtrunc_default)
    fA, rs = fuselegs(A, indexes)
    E, D, Ep = tensoreig(fA, truncfun=svdtrunc)
    li1, li2 = length.(indexes)

    if indexes[1] isa Tuple
        indxs = (ntuple(x -> (1,1,x), li1)..., 2)
        E = splitlegs(E, indxs, rs...)
    end
    if indexes[2] isa Tuple
        indxs = (1, ntuple(x -> (2,2,x), li2)...)
        Ep = splitlegs(Ep, indxs, rs...)
    end
    return (E, D, Ep)
end

function _tensoreig(a::Matrix; truncfun = length, helper=false)
    if LA.ishermitian(a)
        evals, evecs = LA.eigen(LA.Hermitian(a))
    else
        evals, evecs = LA.eigen(a)
    end

    p = sortperm(evals, by=abs, rev=true)
    evals = evals[p]
    evecs = evecs[:,p]
    cutoff = truncfun(evals)
    E = evecs[:,1:cutoff]
    d = LA.diagm(0 => evals[1:cutoff])
    helper && return E, d, E', cutoff
    return E, d, E'
end

function tensoreig(a::DTensor{T,2}; truncfun = length) where T
    E, d, Ep = _tensoreig(a.array, truncfun = truncfun)
    DTensor(E), DTensor(d), DTensor(collect(Ep))
end

function tensoreig(A::DASTensor{T,N,SYM,CHARGES,SIZES,CHARGE};
                   truncfun = svdtrunc_default) where
                   {T,N,SYM,CHARGES,SIZES,CHARGE}
    N == 2 || throw(ArgumentError("EIG only works on rank 2 tensors"))
    lch = connectingcharge(A)
    ld =  sizes(A,2)[[chargeindex(c,charges(A,2)) for c in lch]]

    E  = DASTensor{T,N}(SYM, (charges(A,1), lch),
            deepcopy.((sizes(A,1), ld)),
            in_out(A), charge(A))
    D  = DASTensor{T,N}(SYM, (lch, lch),
            deepcopy.((ld, ld)),
            vcat(inv(in_out(A,2)), in_out(A,2)))
    Ep = DASTensor{T,N}(SYM, (lch, charges(A,2)),
            deepcopy((ld,sizes(A,2))),
            vcat(inv(in_out(A,2)), in_out(A,2)))

    for (k, degen) in tensor(A)
        in, out = k[1], k[2]
        E[DASSector(in, out)], D[DASSector(out, out)], Ep[DASSector(out, out)], cutoff =
                _tensoreig(degen, truncfun = truncfun, helper = true)
        E.dims[2][chargeindex(out,charges(E,2))] = cutoff
        D.dims[1][chargeindex(out,charges(D,1))] = cutoff
        D.dims[2][chargeindex(out,charges(D,2))] = cutoff
        Ep.dims[1][chargeindex(out,charges(Ep,1))] = cutoff
    end
    return (E, D, Ep)
end
