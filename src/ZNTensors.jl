#= Struct =#
mutable struct ZNTensor{T,N,M} <: DASTensor{T,N}
    charges::NTuple{N,UnitRange{Int}}
    sizes::NTuple{N,Vector{Int}}
    in_out::NTuple{N,Int}
    tensor::Dict{NTuple{N,Int}, Array{T,N}}
    function ZNTensor{T,N,M}(charges, sizes, in_out, tensor) where {T,N,M}
        all(in([-1,1]), in_out) || throw(ArgumentError("in/outs needs to be ∈ [-1,1]"))
        all(map(==, length.(charges), length.(sizes))) ||
            throw(ArgumentError("charges need to be same length as sizes"))
        all(map(isequal(0:(M-1)),charges)) ||
            throw(ArgumentError("charges need to be 0:$(M-1) for Z$M"))
        new{T,N,M}(charges, sizes, in_out, tensor)
    end
end


#= Print =#
printname(::Type{ZNTensor{T,N,M}}) where {T,N,M} = "Z$M-symmetric Tensor"


#= Constructors =#
ZNTensor{T,N,M}(dims, in_out) where {T,N,M} =
    ZNTensor{T,N,M}(ntuple(x->0:(M-1),N),dims, in_out, Dict())

ZNTensor{M}(dims::NTuple{N}, in_out, T = ComplexF64) where {N,M} =
    ZNTensor{T,N,M}(dims, in_out)

ZNTensor(dims::NTuple{N}, in_out, T = ComplexF64) where N =
    ZNTensor{T,N,length(dims[1])}(dims, in_out)

ZNTensor(dims, in_out, tensors::Dict{NTuple{N,Int}, Array{T,N}}) where {T,N} =
    ZNTensor{T,N,length(dims[1])}(ntuple(x->0:length(dims[1])-1,N),dims, in_out, tensors)

ZNTensor{M}(T::Type = ComplexF64) where M = ZNTensor{T,0,M}((),(),(),Dict())

ZNTensor{T,N,M}(charges, dims, in_out) where {T,N,M} =
    ZNTensor{T,N,M}(charges, dims, in_out, Dict())

function constructnew(::Type{ZNTensor{S,L,M}}, newfields, newtensor::Dict{NTuple{N,Int},Array{T,N}}) where {N,M,L,T,S}
    return ZNTensor{T,N,M}(newfields...,newtensor)
end

#= Helper Functions =#
filterfun(::Type{<:ZNTensor{T,N,M}}) where {T,N,M}  = (x, y) -> iszero(mod(sum(x .* y), M))

isinvariant(A::ZNTensor{T,N,M}) where {T,N,M} =
    all(iszero ∘ (x->mod(x,M)) ∘ sum, in_out(A) .* k for k in keys(tensor(A)))

charge(A::ZNTensor{T,N,M}) where {T,N,M} = -mod(sum(map(*,in_out(A),first(keys(tensor(A))))),M)

fusecharge(::Type{<:ZNTensor{T,N,M}}, oldcharges, in_out, out) where {T,N,M} = 0:(M-1)

fusecharges(::Type{<:ZNTensor{T,N,M}}, in_out, out) where {T,M,N} = x -> mod(out * sum(x .* in_out),M)

Base.rand(::Type{ZNTensor{T,N,M}}, dims, in_out) where {T,N,M} =
    Base.rand(ZNTensor{T,N,M}, ntuple(x -> 0:(M-1),N), dims, in_out)


#= Copy and Similarity Functions =#
function Base.deepcopy(A::ZNTensor{T,N,M}) where {T,N,M}
    ZNTensor{T,N,M}(charges(A), deepcopy(sizes(A)), in_out(A), deepcopy(tensor(A)))
end

function similar_from_indices(T::Type, index::NTuple{N,Int},
     A::ZNTensor{S,NA,M}, ::Type{Val{CA}} = Val{:N}) where {N,S,CA,M,NA}
    return ZNTensor{T,N,M}(charges(A,index), deepcopy(sizes(A,index)), in_out(A,index), Dict())
end

function similar_from_indices(T::Type, index::NTuple{N,Int}, A::ZNTensor{S,NA,M}, B::ZNTensor,
            ::Type{Val{CA}} = Val{:N}, ::Type{Val{CB}} = Val{:N}) where {N,CA,CB,S,NA,M}
    chargesC = TT.getindices(TT.vcat(charges(A), charges(B)), index)
    sizesC = TT.getindices(TT.vcat( CA == :N ? sizes(A) : reverse.(sizes(A)),
                                    CB == :N ? sizes(B) : reverse.(sizes(B)),
                                    index))
    in_outC = TT.getindices(TT.vcat( CA == :N ? in_out(A) : -in_out(A),
                                     CB == :N ? in_out(B) : -in_out(B)),
                                     index)
    return ZNTensor{T,N,M}(chargesC , deepcopy(sizesC), in_outC)
 end

function similar_from_indices(T::Type, poA, poB, p1, p2,
        A::ZNTensor{S,N,M}, B::ZNTensor,
        ::Type{Val{CA}} = Val{:N}, ::Type{Val{CB}} = Val{:N}) where {CA,CB,S,N,M}
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
    return ZNTensor{T,length(p12),M}(chargesC, deepcopy(sizesC), in_outsC,Dict())
end

Base.similar(A::ZNTensor{T,N,M}, ::Type{S}) where {T,N,S,M} =
    ZNTensor{S,N,M}(charges(A), deepcopy(sizes(A)), in_out(A))

Base.similar(A::ZNTensor{T,N,M}) where {T,N,M} =
    ZNTensor{T,N,M}(charges(A), deepcopy(sizes(A)), in_out(A))
