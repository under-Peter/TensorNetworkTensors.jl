function tensorsvd(A::DASTensor{T,N,SYM,CHARGES,SIZES,CHARGE};
                   svdcutfunction = svdcutfun_default) where
                   {T,N,SYM,CHARGES,SIZES,CHARGE}
    N == 2 || throw(ArgumentError("SVD only works on rank 2 tensors"))
    lch = connectingcharge(A)
    ld =  sizes(A,2)[[chargeindex(c,charges(A,2)) for c in lch]]

    U  = DASTensor{T,N}(SYM, (charges(A,1), lch),
            deepcopy.((sizes(A,1), ld)),
            in_out(A))
    S  = DASTensor{T,N}(SYM, (lch, lch),
            deepcopy.((ld, ld)),
            vcat(inv(in_out(A,2)), in_out(A,2)))
    Vd = DASTensor{T,N}(SYM, (lch, charges(A,2)),
            deepcopy((ld,sizes(A,2))),
            vcat(inv(in_out(A,2)), in_out(A,2)))

    for (k, degen) in tensor(A)
        in, out = k[1], k[2]
        U[DASSector(in, out)], S[DASSector(out, out)], Vd[DASSector(out, out)], cutoff =
                _tensorsvd(degen, svdcutfunction = svdcutfunction, helper = true)
        U.dims[2][chargeindex(out,charges(U,2))] = cutoff
        S.dims[1][chargeindex(out,charges(S,1))] = cutoff
        S.dims[2][chargeindex(out,charges(S,2))] = cutoff
        Vd.dims[1][chargeindex(out,charges(Vd,1))] = cutoff
    end
    return (U, S, Vd)
end

function tensorsvd(A::DASTensor, indexes; svdcutfunction = svdcutfun_default)
    fA, rs = fuselegs(A, indexes)
    U, S, Vt = tensorsvd(fA, svdcutfunction=svdcutfunction)
    li1 = length(indexes[1])
    if !iszero(li1)
        indxs = (ntuple(x -> (1,(1,x)), li1)..., (2,))
        U = splitlegs(U, indxs, rs...)
    end
    li2 = length(indexes[2])
    if !iszero(li2)
        indxs = ((1,), ntuple(x -> (2,(2,x)), li2)...)
        Vt = splitlegs(Vt, indxs, rs...)
    end
    return (U,S,Vt)
end

function _tensorsvd(A::AbstractArray; svdcutfunction = svdcutfun_default,
        helper::Bool = false)
    F = svd(A)
    svals = F.S
    cutoff = svdcutfunction(svals)
    U = F.U[:, 1:cutoff]
    S = diagm(0=>svals[1:cutoff])
    Vt = F.Vt[1:cutoff, :]
    helper && return (U, S, Vt, cutoff)
    return (U, S, Vt)
end

function connectingcharge(A)
    chs1, chs2 = in_out(A) ⊗ charges(A)
    chs1 += charge(A)
    return in_out(A,2) ⊗ (chs1 ∩ chs2)
end

svdcutfun_default(x) = length(x)
svdcutfun_discardzero(x) = length(filter(!iszero, x))
svdcutfun_maxχ(χ) = x -> min(length(x), χ)
svdcutfun_maxcumerror(ϵ; χ::Int = Inf64) = x -> _maxcumerror(x, ϵ, χ)
svdcutfun_maxerror(ϵ; χ::Int = Inf64) = x -> _maxerror(x, ϵ, χ)

function _maxcumerror(xs, ϵ, χ)
    acc, lxs = 0, length(xs)
    for i in lxs:-1:1
        acc += xs[i]
        acc >= ϵ && return min(i+1,χ)
    end
    return min(lxs, χ)
end

function _maxerror(xs, ϵ, χ)
    index = findfirst(x -> x < ϵ, xs)
    index == nothing && return min(length(xs), χ)
    return min(index-1, χ)
end
