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
#AbstractTensor
function _tensorsvd(A::Array{<:Any,2}; svdtrunc = svdtrunc_default,
        helper::Bool = false)
    F = svd(A)
    svals = F.S
    cutoff = svdtrunc(svals)
    U = F.U[:, 1:cutoff]
    S = diagm(0=>svals[1:cutoff])
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
    if li1 != 1
        indxs = (ntuple(x -> (1,1,x), li1)..., 2)
        U = splitlegs(U, indxs, rs...)
    end
    if li2 != 1
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
tensorsvd(A::DTensor{T,2}; svdtrunc = svdtrunc_default) where T =
    DTensor.(_tensorsvd(A.array, svdtrunc = svdtrunc))

#DASTensor
function tensorsvd(A::DASTensor{T,N,SYM,CHARGES,SIZES,CHARGE};
                   svdtrunc = svdtrunc_default) where
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
                _tensorsvd(degen, svdtrunc = svdtrunc, helper = true)
        U.dims[2][chargeindex(out,charges(U,2))] = cutoff
        S.dims[1][chargeindex(out,charges(S,1))] = cutoff
        S.dims[2][chargeindex(out,charges(S,2))] = cutoff
        Vd.dims[1][chargeindex(out,charges(Vd,1))] = cutoff
    end
    return (U, S, Vd)
end

function connectingcharge(A)
    chs1, chs2 = in_out(A) ⊗ charges(A)
    chs1 += charge(A)
    return in_out(A,2) ⊗ (chs1 ∩ chs2)
end
