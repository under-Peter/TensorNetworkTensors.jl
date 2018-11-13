#= =#
struct U1Charges <: DASCharges
    v::StepRange{Int,Int}
end

length(a::U1Charges) = length(a.v)

function ⊕(a::U1Charges, b::U1Charges)
    loa, hia = extrema(a.v)
    lob, hib = extrema(b.v)
    step = min(abs(a.v.step), abs(b.v.step))
    return U1Charges((loa + lob):step:(hia + hib))
end
-(a::U1Charges) = U1Charges(-a.v.stop:(a.v.step):-a.v.start)
# -(a::U1Charges) = U1Charges(-a.v.start:(-a.v.step):-a.v.stop)

struct U1Charge <: DASCharge
    ch::Int
end
⊕(a::U1Charge, b::U1Charge) = U1Charge(a.ch + b.ch)

chargeindex(ch::U1Charge, chs::U1Charges) = div(ch.ch - chs.v.start, chs.v.step)+1

getindex(s::U1Charges, i) = U1Charge(s.v[i])
struct U1Sector{L} <: DASSector{L}
    chs::NTuple{L, U1Charge}
    U1Sector(k::NTuple{L,Int}) where L =
        new{L}(ntuple(i -> U1Charge(k[i]), L))
    U1Sector{L}(k::NTuple{L,U1Charge}) where L =
        new{L}(k)
end

getindex(s::U1Sector{N},i::NTuple{M}) where {N,M} = U1Sector{M}(TT.getindices(s.chs,i))
vcat(s1::U1Sector{N1}, s2::U1Sector{N2}) where {N1,N2} =
    U1Sector{N1 + N2}(TT.vcat(s1.chs, s2.chs))
deleteat(s1::U1Sector{M}, i::NTuple{L}) where {L,M} = U1Sector{M-L}(TT.deleteat(s1.chs,i))

allsectors(chs::NTuple{N,U1Charges}) where {N} =
    (U1Sector(s) for s in Iterators.product(ntuple(i -> chs[i].v,N)...))

#= Struct =#
mutable struct U1Tensor{T,L} <: DASTensor{T,L}
    charges::NTuple{L,U1Charges}
    sizes::NTuple{L,Vector{Int}}
    in_out::InOut{L}
    tensor::Dict{U1Sector{L}, Array{T,L}}
    function U1Tensor{T,L}(charges::NTuple{L}, sizes, in_out::NTuple{L}, tensor) where {T,L}
        new{T,L}(
            ntuple(i -> U1Charges(charges[i]),L),
            sizes,
            InOut(in_out),
            Dict(U1Sector{L}(k) => v for (k,v) in tensor))
    end
    function U1Tensor{T,L}(charges::NTuple{L}, sizes, in_out::InOut{L}, tensor) where {T,L}
        new{T,L}( charges, sizes, in_out, tensor)
    end
end


#= Print =#
printname(::Type{<:U1Tensor}) = "U1-symmetric Tensor"


#= Constructors =#
U1Tensor{T,N}(charges, dims, in_out) where {T,N} =
    U1Tensor{T,N}(charges, dims, in_out, Dict())

U1Tensor(charges::NTuple{N}, dims, in_out, T = ComplexF64) where N =
    U1Tensor{T,N}(charges, dims, in_out, Dict())

U1Tensor(charges, dims, in_out, tensors::Dict{NTuple{N,Int}, Array{T,N}}) where {T,N} =
    U1Tensor{T,N}(charges, dims, in_out, tensors)

U1Tensor(T::Type = ComplexF64) = U1Tensor{T,0}((), (), (), Dict())

function constructnew(::Type{<:U1Tensor}, newfields,
        newtensor::Dict{U1Sector{M},Array{T,M}}) where {M,T}
    return U1Tensor{T,M}(newfields...,newtensor)
end


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
    in_outsC = vcat(
        ifelse(CA == :N, in_out(A), - in_out(A))[poA],
        ifelse(CB == :N, in_out(B), - in_out(B))[poB])[p12]
    return U1Tensor{T, length(p12)}( chargesC, sizesC, in_outsC)
end

Base.similar(A::U1Tensor{T,N}, ::Type{S}) where {T,N,S} =
    U1Tensor{S,N}(charges(A), sizes(A), in_out(A))

Base.similar(A::U1Tensor{T,N}) where {T,N} =
    U1Tensor{T,N}(charges(A), sizes(A), in_out(A))
