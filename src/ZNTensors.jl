#= Charges, Charge and Sectors =#
struct ZNCharges{N} <: DASCharges end
⊕(::ZNCharges{N}, ::ZNCharges{N}) where N = ZNCharges{N}()
-(::ZNCharges{N}) where N = ZNCharges{N}()
length(::ZNCharges{N}) where N = N

struct ZNCharge{N} <: DASCharge
    ch::Int
    ZNCharge{N}(a) where {N} = new(mod(a,N))
end
⊕(a::ZNCharge{N}, b::ZNCharge{N}) where {N} = ZNCharge{N}(a.ch + b.ch)

chargeindex(ch::ZNCharge{N}, chs::ZNCharges{N}) where N = (ch.ch + 1)

getindex(s::ZNCharges{N}, i) where N = ZNCharge{N}((0:(N-1))[i])

struct ZNSector{L,N} <: DASSector{L}
    chs::NTuple{L, ZNCharge{N}}
    ZNSector{0,N}(::Tuple{}) where {N} = new(())
    ZNSector{L,N}(k::NTuple{L,Int}) where {L,N} =
        new(ntuple(i -> ZNCharge{N}(k[i]),L))
    ZNSector{L,N}(k::NTuple{L,ZNCharge}) where {L,N} =
        new(k)
end

ZNSector{N}(t::NTuple{L,Int}) where {N,L} =
    ZNSector{N,L}(ntuple(i -> ZNCharge{N}(t[i]),L))
getindex(s::ZNSector{L,N}, i::NTuple{M}) where {L,N,M} =
    ZNSector{M,N}(TT.getindices(s.chs,i))
vcat(s1::ZNSector{L1,N}, s2::ZNSector{L2,N}) where {N,L1,L2} =
    ZNSector{L1 + L2,N}(TT.vcat(s1.chs, s2.chs))
vcat(s1::ZNSector) = s1
deleteat(s1::ZNSector{L,N}, i::NTuple{M}) where {L,N,M} =
    ZNSector{L-M,N}(TT.deleteat(s1.chs,i))

allsectors(::NTuple{L,ZNCharges{N}}) where {L,N} =
    (ZNSector{L,N}(s) for s in Iterators.product([0:(N-1) for ch in 1:L]...))

#= Struct =#
mutable struct ZNTensor{T,L,N} <: DASTensor{T,L}
    charges::NTuple{L,ZNCharges{N}}
    sizes::NTuple{L,Vector{Int}}
    in_out::InOut{L}
    tensor::Dict{ZNSector{L,N}, Array{T,L}}
    function ZNTensor{T,L,N}(charges, sizes, in_out::NTuple{L}, tensor) where {T,L,N}
        new{T,L,N}(
            ntuple(i -> ZNCharges{N}(),L),
            sizes,
            InOut(in_out),
            Dict(ZNSector{L,N}(k) => v for (k,v) in tensor))
    end
    function ZNTensor{T,L,N}(charges, sizes, in_out::InOut{L}, tensor) where {T,L,N}
        new{T,L,N}(
            charges,
            sizes,
            in_out,
            tensor)
    end
end

@inline znn(::ZNTensor{T,L,N}) where {T,L,N} = N


#= Print =#
printname(::Type{ZNTensor{T,L,N}}) where {T,L,N} = "Z$N-symmetric Tensor"


#= Constructors =#
ZNTensor{T,L,N}(dims, in_out) where {T,L,N} =
    ZNTensor{T,L,N}(ntuple(x->0:(N-1),L), dims, in_out, Dict())

ZNTensor{N}(dims::NTuple{L}, in_out, T = ComplexF64) where {N,L} =
    ZNTensor{T,L,N}(dims, in_out)

ZNTensor(dims::NTuple{L}, in_out, T = ComplexF64) where L =
    ZNTensor{T,L,length(dims[1])}(dims, in_out)

ZNTensor(dims, in_out, tensors::Dict{NTuple{L,Int}, Array{T,L}}) where {T,L} =
    ZNTensor{T,L,length(dims[1])}(ntuple(x->0:length(dims[1])-1,L),dims, in_out, tensors)

ZNTensor{N}(T::Type = ComplexF64) where N = ZNTensor{T,0,N}((),(),(),Dict())

ZNTensor{T,L,N}(charges, dims, in_out) where {T,L,N} =
    ZNTensor{T,L,N}(charges, dims, in_out, Dict())

function constructnew(::Type{ZNTensor{T1,L1,N1}}, newfields,
            newtensor::Dict{ZNSector{L,N},Array{T,L}}) where {T1,L1,N1,L,N,T}
    return ZNTensor{T,L,N}(newfields...,newtensor)
end

#= Helper Functions =#
Base.rand(::Type{ZNTensor{T,L,N}}, dims, in_out) where {T,L,N} =
    Base.rand(ZNTensor{T,L,N}, ntuple(x -> 0:(N-1),L), dims, in_out)


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
    in_outsC = vcat(
        ifelse(CA == :N, in_out(A), - in_out(A))[poA],
        ifelse(CB == :N, in_out(B), - in_out(B))[poB])[p12]
    return ZNTensor{T,length(p12),M}(chargesC, deepcopy(sizesC), in_outsC,Dict())
end

Base.similar(A::ZNTensor{T,N,M}, ::Type{S}) where {T,N,S,M} =
    ZNTensor{S,N,M}(charges(A), deepcopy(sizes(A)), in_out(A))

Base.similar(A::ZNTensor{T,N,M}) where {T,N,M} =
    ZNTensor{T,N,M}(charges(A), deepcopy(sizes(A)), in_out(A))
