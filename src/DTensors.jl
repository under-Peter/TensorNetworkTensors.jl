"""
    DTensor{T,N} <: AbstractTensor{T,N}

wrapper for generic Array{T,N} to represent dense tensors.
"""
struct DTensor{T,N} <: AbstractTensor{T,N}
    array::Array{T,N}
end

DTensor(T::Type = ComplexF64) = DTensor{T,0}(Array{T,0}(undef))
DTensor{T,N}(ds::NTuple{N,Int}) where {T,N} = DTensor{T,N}(Array{T,N}(undef, ds...))
DTensor{T}(ds::NTuple{N,Int}) where {T,N} = DTensor{T,N}(ds)

function Base.show(io::IO, A::DTensor)
    print(io, typeof(A))
    print(io, A.array)
end


Base.ndims(A::DTensor) = ndims(A.array)
Base.getindex(A::DTensor, args...) = getindex(A.array, args...)
Base.setindex!(A::DTensor, args...) = setindex!(A.array, args...)
Base.size(A::DTensor,i...) = size(A.array,i...)
Base.isapprox(A::DTensor, B::DTensor) = A.array ≈ B.array
Base.:(==)(A::DTensor, B::DTensor) = A.array == B.array
Base.similar(A::DTensor{T,N}, ::Type{S}) where {T,N,S} = DTensor(Base.similar(A.array, S))
Base.similar(A::DTensor{T,N}) where {T,N,S} = DTensor(Base.similar(A.array, T))
Base.copy(A::DTensor) = DTensor(copy(A.array))
Base.copy!(dest::DTensor, source::DTensor) = DTensor(copy!(dest.array, source.array))
Base.copyto!(dest::DTensor, source::DTensor) = Base.copyto!(dest.array, source.array)
Base.eltype(A::DTensor{T,N}) where {T,N} = T
LA.diag(a::DTensor) = diag(a.array,0)
toarray(a::DTensor) = a.array

function initwith!(A::DTensor{T,N}, fun) where {T,N}
    A.array .= fun(T, size(A))
    return A
end

initwithzero!(A::DTensor) = initwith!(A, zeros)
initwithrand!(A::DTensor) = initwith!(A, rand)
