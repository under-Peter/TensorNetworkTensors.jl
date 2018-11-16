import TensorOperations: ndims, similar_from_indices, add!, trace!, contract!, scalar, numind
struct DTensor{T,N} <: AbstractTensor{T,N}
    array::Array{T,N}
end

DTensor(T::Type = ComplexF64) = DTensor{T,0}(Array{T,0}(undef))
DTensor{T,N}(ds::NTuple{N,Int}) where {T,N} = DTensor{T,N}(zeros(T,ds...))
tnttype(::Type{<:DTensor}) = DTensor

function Base.show(io::IO, A::DTensor)
    print(io, typeof(A))
    println(io)
    print(io, A.array)
end


TensorOperations.numind(A::DTensor)  = TensorOperations.numind(A.array)
Base.ndims(A::DTensor) = ndims(A.array)
scalar(A::DTensor)  = scalar(A.array)
Base.getindex(A::DTensor, i...) = getindex(A.array, i...)
Base.setindex!(A::DTensor, x, i...) = setindex!(A.array, x, i...)
Base.size(A) = size(A.array)
Base.size(A::DTensor, i::Int) where N = size(A.array,i)
Base.isapprox(A::DTensor, B::DTensor) = (A.array ≈ B.array)
Base.isapprox(A, B::DTensor) = false
Base.isapprox(A::DTensor, B) = false
Base.:(==)(A::DTensor{T,N}, B::DTensor{T,N}) where {T,N} = A.array == B.array
Base.:(==)(A, B::DTensor) = false
Base.:(==)(A::DTensor, B) = false
apply!(A::DTensor, f) = (A.array[:] = f(A.array); A)
Base.adjoint(A::DTensor{T,N}) where {T,N} = DTensor{T,N}(conj(A.array))
diag(a::DTensor{T,2}) where T = diag(a.array)

Base.conj!(A::DTensor) = DTensor(conj!(A.array))
Base.:(*)(α::Number, A::DTensor) = DTensor( α * A.array)

similar_from_indices(T::Type, indices, A::DTensor, ::Type{<:Val}=Val{:N}) =
    DTensor(similar_from_indices(T, indices, A.array, Val{:N}))

similar_from_indices(T::Type, poA, poB, p1, p2, A::DTensor, B::DTensor, CA, CB) =
    DTensor(similar_from_indices(T, poA, poB, p1, p2, A.array, B.array, CA, CB))

similar_from_indices(T::Type, index, A::DTensor, B::DTensor,
    ::Type{Val{CA}} = Val{:N}, ::Type{Val{CB}} = Val{:N}) where
    {CA,CB} = DTensor(similar_from_indices(T, index, A.array, B.array, Val{CA}, Val{CB}))

similar_from_indices(T::Type, p1::Tuple, p2, A::DTensor, CA::Type{<:Val}) =
 DTensor(similar_from_indices(T, (p1...,p2...), A.array, CA))


Base.similar(A::DTensor{T,N}, ::Type{S}) where {T,N,S} = DTensor(Base.similar(A.array, S))
Base.similar(A::DTensor{T,N}) where {T,N,S} = DTensor(Base.similar(A.array, T))
Base.copy(A::DTensor) = DTensor(copy(A.array))
Base.copy!(dest::DTensor, source::DTensor) = DTensor(copy!(dest.array, source.array))
Base.copyto!(dest::DTensor, source::DTensor) = Base.copyto!(dest.array, source.array)
Base.eltype(A::DTensor{T,N}) where {T,N} = T

add!(α, A::DTensor{T,N}, CA::Type{<:Val}, β, C::DTensor{S,M},
    p1, p2) where {T,N,S,M} = add!(α, A, CA, β, C, (p1..., p2...))

add!(α, A::DTensor{T,N}, ::Type{Val{CA}}, β, C::DTensor{S,M}, indCinA) where
    {CA,T,N,S,M} = DTensor(add!(α, A.array, Val{CA}, β, C.array, indCinA))

add!(α, A::DTensor, CA::Type{<:Val}, β, C::DTensor, p1::Tuple, p2::Tuple) =
    add!(α, A, CA, β, C, (p1...,p2...))

trace!(α, A::DTensor, CA::Type{<:Val}, β, C::DTensor, p1, p2, cindA1, cindA2) =
    trace!(α, A, CA, β, C, (p1..., p2...), cindA1, cindA2)
trace!(α, A::DTensor, ::Type{Val{CA}}, β, C::DTensor, indCinA, cindA1, cindA2) where
    {CA} = DTensor( trace!(α, A.array, Val{CA}, β, C.array, indCinA, cindA1, cindA2))

contract!(α, A::DTensor, ::Type{Val{CA}}, B::DTensor, ::Type{Val{CB}}, β,
    C::DTensor, oindA, cindA, oindB, cindB, indCinoAB, ::Type{Val{M}} = Val{:BLAS}) where {CA,CB,M} =
    DTensor(contract!(α, A.array, Val{CA}, B.array, Val{CB},
                             β, C.array, oindA, cindA, oindB, cindB,
                             indCinoAB, Val{M}))

contract!(α, A::DTensor, CA::Type{<:Val}, B::DTensor, CB::Type{<:Val}, β,
                C::DTensor, oindA, cindA, oindB, cindB, p1, p2, method::Type{<:Val} = Val{:BLAS}) =
    contract!(α, A, CA, B, CB, β, C, oindA, cindA, oindB, cindB, (p1..., p2...), method)

#= RESHAPING W/O CONTRACTION=#
function fuselegs(A::DTensor, indexes)
    _pick(i::Int) = size(A,i)
    _pick(i::NTuple) = tuple((size(A,j) for j in i)...)
    perm = TT.vcat(indexes...)
    dims = map(prod ∘ _pick, indexes)
    inverter = tuple(map(_pick, indexes)...)
    return  (DTensor(reshape(permutedims(A.array, perm), dims...)), inverter)
end

function splitlegs(A::DTensor{T}, indexes::NTuple{N,Union{Int,NTuple{3,Int}}}, inverter) where {N,T}
    tindexes = map(i -> ifelse( i isa Int, (i,), i), indexes)
	perm  = TT.sortperm(tindexes)
	iperm = TT.invperm(perm)
	indexes = TT.getindices(indexes, perm)
    _pick(i::Int) = size(A, i)
    _pick((l,m,n)) = inverter[m][n]
    dims = [_pick(i) for i in indexes]
	return DTensor{T,N}(permutedims(reshape(A.array, dims...), iperm))
end

function tensorsvd(A::DTensor{T,2}; svdcutfunction = svdcutfun_default, helper::Bool = false) where T
    F   = svd(A.array)
    cutoff = svdcutfunction(F.S)
    U = DTensor{T,2}(F.U[:, 1:cutoff])
    S = DTensor{T,2}(diagm(0=>F.S)[1:cutoff, 1:cutoff])
    Vt = DTensor{T,2}(F.Vt[1:cutoff, :])
    helper && return (U, S, Vt, cutoff)
    return (U, S, Vt)
end

function tensorsvd(A::DTensor{T}, indexes; svdcutfunction = svdcutfun_default) where T
    fA, inverter = fuselegs(A, indexes)
    U, S, Vt = tensorsvd(fA, svdcutfunction=svdcutfunction)
    li1 = length(indexes[1])
    if !iszero(li1)
        indxs = (ntuple(x -> (1,1,x), li1)..., (2,))
        U = splitlegs(U, indxs, inverter)
    end
    li2 = length(indexes[2])
    if !iszero(li2)
        indxs = ((1,), ntuple(x -> (2,2,x), li2)...)
        Vt = splitlegs(Vt, indxs, inverter)
    end
    return (U,S,Vt)
end
