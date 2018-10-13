#= Struct =#
mutable struct U1Tensor{T,N} <: DASTensor{T,N}
    charges::NTuple{N,UnitRange{Int}}
    sizes::NTuple{N,Vector{Int}}
    in_out::NTuple{N,Int}
    tensor::Dict{NTuple{N,Int}, Array{T,N}}
end


#= Print =#
printname(::Type{<:U1Tensor}) = "U1-symmetric Tensor"


#= Constructors =#
U1Tensor{T,N}(charges, dims, in_out) where {T,N} =
    U1Tensor{T,N}(charges, dims, in_out, Dict())

U1Tensor(charges::NTuple{N}, dims, in_out, T = ComplexF64) where N =
    U1Tensor{T,N}(charges, dims, in_out)

U1Tensor(charges, dims, in_out, tensors::Dict{NTuple{N,Int}, Array{T,N}}) where {T,N} =
    U1Tensor{T,N}(charges, dims, in_out, tensors)

U1Tensor(T::Type = ComplexF64) = U1Tensor{T,0}((), (), (), Dict())

function constructnew(::Type{<:U1Tensor}, newfields, newtensor::Dict{NTuple{M,Int},Array{T,M}}) where {M,T}
    return U1Tensor{T,M}(newfields...,newtensor)
end


#= Helper Functions =#
scalar(A::U1Tensor{T,0}) where T = first(first(values(A.tensor)))

filterfun(::Type{<:U1Tensor})  = (x, y) -> iszero(sum(map(*, x, y)))

isinvariant(A::U1Tensor{T,N}) where {T,N} =
    all(iszero ∘ sum, map(*,in_out(A),k) for k in keys(tensor(A)))

charge(A::U1Tensor) = -sum(map(*,in_out(A),first(keys(tensor(A)))))

function fusecharge(::Type{<:U1Tensor}, oldcharges::NTuple{N}, io::NTuple{N}, out) where N
    charges = map((x, y) -> x .* y , io, oldcharges)
    lower, upper = mapreduce(extrema, (x,y) -> x .+ y, charges)
    out == 1 && return lower:upper
    return (-upper):(-lower)
end

function fusecharge(::Type{<:U1Tensor}, oldcharges::UnitRange, io::Int, out)
    lower, upper = extrema(oldcharges)
    out == io && return lower:upper
    return (-upper):(-lower)
end

fusecharges(::Type{<:U1Tensor}, in_out) = x -> sum(x .* in_out)
fusecharges(::Type{<:U1Tensor}, in_out, out) = x -> out * sum(x .* in_out)


#= Copy and Similarity Functions =#
function Base.deepcopy(A::U1Tensor{T,N}) where {T,N}
    U1Tensor{T,N}(charges(A), deepcopy(sizes(A)), in_out(A), deepcopy(tensor(A)))
end

function similar_from_indices(T::Type, index::NTuple{N,Int},
     A::U1Tensor{S}, ::Type{Val{CA}} = Val{:N}) where {N,S,CA}
    return U1Tensor{T,N}(charges(A,index), sizes(A,index), in_out(A,index))
end

function similar_from_indices(T::Type, index::NTuple{N,Int}, A::U1Tensor, B::U1Tensor,
            ::Type{Val{CA}} = Val{:N}, ::Type{Val{CB}} = Val{:N}) where {N,CA,CB}
    chargesC = TT.getindices(TT.vcat(charges(A), charges(B)), index)
    sizesC = TT.getindices(TT.vcat( CA == :N ? sizes(A) : reverse.(sizes(A)),
                                    CB == :N ? sizes(B) : reverse.(sizes(B)),
                                    index))
    in_outC = TT.getindices(TT.vcat( CA == :N ? in_out(A) : -in_out(A),
                                     CB == :N ? in_out(B) : -in_out(B)),
                                     index)
    return U1Tensor{T,N}(chargesC , sizesC, in_outC)
 end

function similar_from_indices(T::Type, poA, poB, p1, p2,
        A::U1Tensor, B::U1Tensor,
        ::Type{Val{CA}} = Val{:N}, ::Type{Val{CB}} = Val{:N}) where {CA,CB}
    p12 = (p1...,p2...)
    chargesC = TT.getindices(TT.vcat(charges(A,poA), charges(B,poB)), p12)
    sizesC = TT.getindices(TT.vcat(
                TT.getindices(CA == :N ? sizes(A) : reverse.(sizes(A)), poA),
                TT.getindices(CB == :N ? sizes(B) : reverse.(sizes(B)), poB)),
                p12)
    in_outsC = TT.getindices(TT.vcat(
                TT.getindices(CA == :N ? in_out(A) : -1 .* in_out(A), poA),
                TT.getindices(CB == :N ? in_out(B) : -1 .* in_out(B), poB)),
                p12)
    return U1Tensor{T, length(p12)}( chargesC, sizesC, in_outsC)
end

Base.similar(A::U1Tensor{T,N}, ::Type{S}) where {T,N,S} =
    U1Tensor{S,N}(charges(A), sizes(A), in_out(A))

Base.similar(A::U1Tensor{T,N}) where {T,N} =
    U1Tensor{T,N}(charges(A), sizes(A), in_out(A))
