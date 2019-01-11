"""
    NDAS{N,S} <: DAS
wraps multiple independent symmetries to work together.

# Example
```julia-repl
julia>NDAS(U1(), Z2())
NDAS{2,(U1, Z2)}()
```
"""
struct NDAS{N,S} <: DAS end
NDAS(s...) where S<:Tuple = NDAS{length(s),s}()

"""
    NDASCharges{N,T} <: DASCharges
holds combinations of multiple independent `DASCharges` that are grouped together.
Yields `NDASCharge{N,T}`s upon iteration.

# Example
```julia-repl
julia> a = NDASCharges(U1Charges(-1:1), ZNCharges{3}());
julia> foreach(println, a)
NDASCharge(U1Charge(-1), Z3Charge(0))
NDASCharge(U1Charge(0), Z3Charge(0))
NDASCharge(U1Charge(1), Z3Charge(0))
NDASCharge(U1Charge(-1), Z3Charge(1))
NDASCharge(U1Charge(0), Z3Charge(1))
NDASCharge(U1Charge(1), Z3Charge(1))
NDASCharge(U1Charge(-1), Z3Charge(2))
NDASCharge(U1Charge(0), Z3Charge(2))
NDASCharge(U1Charge(1), Z3Charge(2))
```
"""
struct NDASCharges{N,S} <: DASCharges
    v::S
    NDASCharges(v...) = new{length(v),typeof(v)}(v)
end
Base.show(io::IO, s::NDASCharges) = print(io,"NDASCharges", s.v)

⊕(a::T, b::T) where {T<:NDASCharges{N}} where N = NDASCharges(map(⊕, a.v, b.v)...)
Base.inv(a::T) where {T<:NDASCharges} = NDASCharges(inv.(a.v)...)
Base.length(a::NDASCharges) = prod(length.(a.v))
Base.intersect(a::NDASCharges{N}, b::NDASCharges{N}) where N = NDASCharges(map(intersect, a.v, b.v)...)
function Base.getindex(a::NDASCharges, i::Int)
    i <= length(a) || throw(BoundsError(a,i))
    inds = Tuple(CartesianIndices(map(x -> 1:length(x), a.v))[i])
    NDASCharge(map(getindex, a.v, inds)...)
end
chargestype(::NDAS{N,S}) where {N,S} =
    NDASCharges{N,Tuple{(chargestype(TT) for TT in S)...}}


"""
    NDASCharge{N} <: DASCharge
holds the charge of multiple independent symmetries grouped together.

# Example
```julia-repl
julia> a = NDASCharge(U1Charge(1), Z2Charge(1));
julia> a ⊕ a
NDASCharge(U1Charge(2), Z2Charge(0))
```
"""
struct NDASCharge{N,S} <: DASCharge
    ch::S
    NDASCharge(ch...) = new{length(ch),typeof(ch)}(ch)
    NDASCharge{N,S}(ch...) where {N,S} = new{N,S}(ch)
end
Base.show(io::IO, s::NDASCharge) = print(io,"NDASCharge", s.ch)

Base.eltype(::NDASCharges{N,S}) where {N,S} =
    NDASCharge{N,Tuple{(eltype(T) for T in S.types)...}}
Base.iszero(a::NDASCharge) = all(iszero,a.ch)
⊕(a::T, b::T) where {T<:NDASCharge} = NDASCharge(map(⊕, a.ch, b.ch)...)
Base.:-(a::T) where {T<:NDASCharge} = T(map(x -> -x, a.ch)...)
Base.isless(a::T, b::T) where {T<:NDASCharge{N}} where N = a.ch < b.ch
Base.zero(::Type{NDASCharge{N,T}}) where {N,T} = NDASCharge{N,T}(ntuple(i -> zero(T.types[i]), N)...)
Base.zero(::NDASCharge{N,T}) where {N,T} = NDASCharge{N,T}(ntuple(i -> zero(T.types[i]), N)...)
Base.:+(chs::T, ch::S) where {T<:NDASCharges, S<:NDASCharge} = NDASCharges(map(+, chs.v, ch.ch)...)
function chargeindex(ch::NDASCharge, chs::NDASCharges)
    linds = LinearIndices(map(x -> 1:x, length.(chs.v)))
    inds = map(chargeindex, ch.ch, chs.v)
    linds[inds...]
end
chargetype(::NDAS{N,S}) where {N,S} = NDASCharge{N,Tuple{(chargetype(TT) for TT in S)...}}
