"""
    DAS
Abstract supertype of all Discrete Abelian Symmetries
"""
abstract type DAS end

"""
    DASCharges
Abstract supertype of all collections of Charges of Discrete Abelian Symmetries
"""
abstract type DASCharges end

"""
    DASCharge
Abstract supertype of all Charges of Discrete Abelian Symmetries
"""
abstract type DASCharge end
Base.hash(a::DASCharge, h::UInt) = hash(a.ch, h)

"""
    ⊕(a::T, bs::T...) where {T<:Union{DASCharge, DASCharges}}
returns the result of fusing one or more `DASCharge` or `DASCharges` together.
Fusing `DASCharges` yields a `DASCharges` that holds all elements that result
from fusing the `DASCharges`.
"""
⊕(a::DASCharges) = a
⊕(a::T, b::T, c::T...) where {T<:DASCharges} where N = ⊕((a ⊕ b),c...)
const oplus = ⊕

function Base.iterate(a::DASCharges, i = 1)
    i <= length(a) && return (a[i], i+1)
    return nothing
end
Base.IteratorSize(::DASCharges) = Base.HasLength()
Base.IteratorEltype(::DASCharges) = Base.HasEltype()

Base.iszero(a::DASCharge) = iszero(a.ch)
Base.:-(a::T) where {T<:DASCharge} = InOut(-1) ⊗ a
Base.:(==)(a::DASCharge, b::DASCharge) = a.ch == b.ch
Base.isless(a::DASCharge, b::DASCharge) = isless(a.ch, b.ch)

"""
    chargeindex(ch::DASCharge, chs::DASCharges)
returns the index i of ch in chs

# Example
```julia-repl
julia> ch = U1Charge(0); chs = U1Charges(-1:1);
julia> chargeindex(ch, chs)
2
julia> chs[2] == ch
true
```
"""
function chargeindex end

include("dastensor/U1.jl")
include("dastensor/ZN.jl")
include("dastensor/NDAS.jl");

#Sector
"""
    DASSector{N,T}
DASSectors are a configuration of charges that are allowed under a given
symmetry and index degeneracy spaces in DASTensors{T,N}.

# Example
```julia-repl
julia> DASSector(U1Charge(1), U1Charge(2))
DASSector(U1Charge(1), U1Charge(2))
```
"""
struct DASSector{N,T}
    chs::NTuple{N,T}
    DASSector(chs::T...) where T = new{length(chs),T}(chs)
    DASSector{N,T}(chs::T...) where {N,T} = new{N,T}(chs)
end

Base.show(io::IO, s::DASSector{N,S}) where {N,S}  = print(io, "DASSector", s.chs)

function Base.iterate(a::DASSector{N}, i = 1) where N
    i <= length(a) && return (a[i], i+1)
    return nothing
end
Base.IteratorSize(::DASSector) = Base.HasLength()
Base.IteratorEltype(::DASSector) = Base.HasEltype()

Base.length(s::DASSector{N}) where N = N
Base.eltype(::DASSector{N,T}) where {N,T} = T
Base.hash(a::DASSector, h::UInt) = hash(a.chs, h)
⊕(s1::DASSector{N,T}, s2::DASSector{N,T}) where {N,T} = DASSector(map(⊕, s1.chs, s2.chs))
Base.getindex(s::DASSector,i::Int) = s.chs[i]
permute(s::DASSector{N,T}, perm) where {N,T}  = DASSector{N,T}(TT.permute(s.chs,perm)...)

"""
    charge(a::DASSector)
returns the charge which is calculated as minus the sum of all charges it contains.
"""
charge(s::DASSector) = -reduce(⊕, s.chs)
Base.isless(a::T, b::T) where {T<:DASSector} = a.chs < b.chs
Base.:(==)(a::T, b::T) where {T<:DASSector} = hash(a) == hash(b)
Base.:(==)(a::DASSector, b::DASSector) = false

Base.vcat(s::DASSector) = s
Base.vcat(s1::DASSector, s2::DASSector, s3::DASSector...) =
    vcat(vcat(s1,s2),s3...)
Base.getindex(s::DASSector{N,T}, i::NTuple{M}) where {N,T,M} =
    DASSector{M,T}(TT.getindices(s.chs,i)...)
Base.vcat(s1::DASSector{N1,T}, s2::DASSector{N2,T}) where {T,N1,N2} =
    DASSector{N1+N2,T}(TT.vcat(s1.chs, s2.chs)...)
deleteat(s1::DASSector{N,T}, i::NTuple{M}) where {N,T,M} =
    DASSector{N-M,T}(TT.deleteat(s1.chs,i)...)

#InOut
"""
    InOut{N}
InOut describes whether representations of the DAS act on an index of a
DASTensor{T,N} directly or via their dual.
InOut(1,1,1,-1) can be read as the first three indices corresponding to incoming,
the last as an outgoing index w.r.t the group action.
"""
struct InOut{N}
    v::NTuple{N,Int}
    function InOut(v::Int...)
        all(in((1,-1)), v) || throw(ArgumentError("elements must either be 1 or -1"))
        new{length(v)}(v)
    end
end
Base.show(io::IO, s::InOut) = print(io, "InOut", s.v)

function Base.iterate(a::InOut{N}, i = 1) where N
    i <= N && return (InOut(a.v[i]), i+1)
    return nothing
end
Base.IteratorSize(::InOut) = Base.HasLength()
Base.IteratorEltype(::InOut) = Base.HasEltype()

Base.length(::InOut{L}) where L = L
Base.eltype(::InOut) = InOut{1}
⊗(a::InOut{N}, b::T) where {T<:DASSector,N} = T(map(⊗, ntuple(i -> a[i] ,Val{N}()) , b.chs)...)
⊗(a::InOut{1}, b::T) where {T<:DASCharge} = T(a.v[1] * b.ch)
⊗(a::InOut{1}, b::T) where {T<:NDASCharge} = NDASCharge(map(⊗(a),b.ch)...)
⊗(a::InOut{M}, b::NTuple{M,DASCharges}) where {M} = ntuple(i -> a[i] ⊗ b[i], Val(M))
⊗(a::InOut{1}, b::DASCharges) = a.v == (1,) ? b : inv(b)


⊗(io::InOut) = x -> io ⊗ x
Base.inv(io::InOut) = InOut((-1 .* io.v)...)
Base.getindex(a::InOut, i::Int) = InOut(a.v[i])
Base.getindex(a::InOut, i::Union{NTuple{L},UnitRange}) where L = InOut(TT.getindices(a.v,i)...)
Base.lastindex(::InOut{L}) where L = L
Base.vcat(a::InOut) = a
Base.vcat(a::InOut, b::InOut) = InOut(a.v..., b.v...)
Base.vcat(a::InOut, b::InOut, c::InOut...) = vcat(vcat(a,b),c...)

#sector generators
"""
    allsectors(chs)
returns a generator that generates all possible combinations of charges in
`chs` wrapped in a `DASSector`.

#Example
```julia-repl
julia> allsectors((U1Charges(-1:1), U1Charges(4:5))) |> collect
3×2 Array{DASSector{2,U1Charge},2}:
 DASSector(U1Charge(-1), U1Charge(4))  DASSector(U1Charge(-1), U1Charge(5))
 DASSector(U1Charge(0), U1Charge(4))   DASSector(U1Charge(0), U1Charge(5))
 DASSector(U1Charge(1), U1Charge(4))   DASSector(U1Charge(1), U1Charge(5))
```
"""
allsectors(a::NTuple{N,DASCharges}) where N = (DASSector(s...) for s in Iterators.product(a...))

"""
    covariantsectors(chs, io[, ch = zero(chs)])
returns  all sectors in allsectors(io ⊗ chs) that have total charge ch
"""
covariantsectors(charges, io::InOut, ch = zero(eltype(first(charges)))) =
    Iterators.filter(==(ch) ∘ charge ∘ ⊗(io), allsectors(charges))

"""
    covariantsectors(chs, io)
returns all sectors in  allsectors(io ⊗ chs) that have total charge zero.
"""
invariantsectors(charges, io::InOut) = covariantsectors(charges, io)

# TENSOR
mutable struct DASTensor{T,N,SYM,CHARGES,SIZES,CHARGE} <: AbstractTensor{T,N}
    chs     ::NTuple{N, CHARGES}
    dims    ::NTuple{N, SIZES}
    io      ::InOut{N}
    ch      ::Union{CHARGE, Missing}
    tensor  ::Dict{DASSector{N,CHARGE},Array{T,N}}
end

function Base.show(io::IO, A:: DASTensor{T,N,SYM}) where {T,N,SYM}
    print(io, "DASTensor{",T,",",N,",",SYM,"}\n")
    print(io, "charges: ", charges(A), "\n")
    print(io, "charge: ",  charge(A), "\n")
    print(io, "sizes: ",   sizes(A), "\n")
    print(io, "in/out: ",  in_out(A), "\n")
    print(io, "Tensors ",  eltype(A), "\n")
end

##Functions
function DASTensor{T,N}(sym::ST, charges, dims, io,
    ch = zero(chargetype(sym)), dict = Dict()) where {T,N,ST <: DAS}
    length.(dims) == length.(charges) || throw(ArgumentError("incompatible dims/charges"))
    CHARGES = chargestype(sym)
    CHARGE  = chargetype(sym)
    SIZES   = Vector{Int}
    DASTensor{T,N,sym,CHARGES,SIZES,CHARGE}(charges,dims,io,ch,dict)
end

"""
    charges(A::DASTensor[,i])
returns the charges of `A`.
If `i` is specified as either `Int` or `Tuple`, returns only the charges of the
indices in `i`.
"""
charges(A::DASTensor)        = A.chs
charges(A::DASTensor,i::Int) = A.chs[i]
charges(A::DASTensor,i)      = TT.getindices(A.chs,i)

"""
    sizes(A::DASTensor[,i])
returns the sizes of `A` as a tuple of vectors `v` such that the degeneracy
space associated with a charge `ch` has size `v[chargeindex(ch, chs)]` where
`chs` is the `DASCharges` associated with the specified leg.
If `i` is specified as either `Int` or `Tuple`, returns only the charges of the
indices in `i`.
"""
sizes(A::DASTensor)        = A.dims
sizes(A::DASTensor,i::Int) = A.dims[i]
sizes(A::DASTensor,i)      = TT.getindices(A.dims,i)

"""
    in_out(A::DASTensor[,i])
returns the `InOut` of `A` which specifies the action of the symmetry group
on the corresponding leg.
If `i` is specified as either `Int` or `Tuple`, returns only the charges of the
indices in `i`.
"""
in_out(A::DASTensor)   = A.io
in_out(A::DASTensor,i) = A.io[i]

"""
    tensor(A::DASTensor[,i])
returns a dictionary of `DASSectors` and their associated degeneracy spaces.
"""
tensor(A::DASTensor)    = A.tensor
Base.getindex(A::DASTensor, i) = A.tensor[i]
Base.setindex!(A::DASTensor{T,N}, t, i::DASSector{N}) where {T,N} =
    (A.tensor[i] = t; A)

Base.values(A::DASTensor) = values(A.tensor)
Base.keys(A::DASTensor) = keys(A.tensor)
Base.haskey(A::DASTensor,k) = haskey(A.tensor,k)
"""
    symmetry(A::DASTensor)
returns the symmetry of a tensor `A`
"""
symmetry(A::DASTensor{<:Any,<:Any,SYM}) where {SYM} = SYM
Base.eltype(A::DASTensor{T,N}) where {T,N} = T
chargetype(A::DASTensor{<:Any,<:Any,<:Any,<:Any,<:Any,CHARGE}) where {CHARGE} = CHARGE
chargestype(A::DASTensor{<:Any,<:Any,<:Any,CHARGES}) where {CHARGES} = CHARGES
dimstype(A::DASTensor{<:Any,<:Any,<:Any,<:Any,SIZES}) where {SIZES} = SIZES
Base.ndims(A::DASTensor{T,N}) where {T,N} = N

"""
    setcharges!(A::DASTensor, chs)
set the charges of `A` to be `chs` where the latter is a tuple of `DASCharges`.
"""
setcharges!(A::DASTensor, charges) = (A.chs = charges; A)
"""
    setsizes!(A::DASTensor, s)
set the sizes of `A` to be `s` where the latter is a tuple of `Vector{Int}`.
"""
setsizes!(A::DASTensor, sizes)     = (A.dims   = sizes;   A)
"""
    setin_out!(A::DASTensor, io)
set the `InOut` of `A` to be `io` where the latter is a `InOut`.
"""
setin_out!(A::DASTensor, in_out)   = (A.io  = in_out;  A)
settensor!(A::DASTensor, tensor)   = (A.tensor  = tensor;  A)

"""
    charge(a::DASTensor)
returns the charge of a tensor.
"""
charge(A::DASTensor) = A.ch
setcharge!(A::DASTensor, ch) = (A.ch = ch; A)

"""
    isinvariant(a::DASTensor)
return true if `charge(a)` is `zero`.
"""
isinvariant(a::DASTensor) = iszero(charge(a))

function chargesize(charge::DASCharge, charges, dims)
    charge in charges || throw(
        ArgumentError(string(charge, " is not in ", charges)))
    dims[chargeindex(charge, charges)]
end
chargesize(A::DASTensor, i, charge)          = chargesize(charge, charges(A,i), sizes(A,i))
degeneracysize(sector, charges, dims)        = map(chargesize, sector, charges, dims)

"""
    initwith!(A::DASTensor{T}, fun [,ch])
modifies `A` such that each sector with charge `ch` (default=zero) is (independently)
set to `fun(T, dims...)` where `dims` is the size of the degeneracy space for the sector.
"""
function initwith!(A::DASTensor{T,N}, fun) where {T,N}
    empty!(A.tensor)
    dims, chs = sizes(A), charges(A)
    for k in covariantsectors(chs, in_out(A), charge(A))
        A.tensor[k] = fun(T, degeneracysize(k, chs, dims)...)
    end
    return A
end

function Base.rand(::Type{DASTensor{T,N}}, sym, charges, dims, io,
                    ch = zero(eltype(first(charges)))) where {T,N}
    A = DASTensor{T,N}(sym, charges, dims, io)
    initwithrand!(A, ch)
end

"""
    initwithzero!(A::DASTensor)
construct all valid sectors  in `A` and initialize their
degeneracy spaces with zeros.
"""
initwithzero!(A::DASTensor) = initwith!(A, zeros)

"""
    initwithrand!(A::DASTensor)
construct all valid sectors in `A` and initialize their
degeneracy spaces with `rand`.
"""
initwithrand!(A::DASTensor) = initwith!(A, rand)

function Base.convert(::Type{Array{S}}, A::DASTensor{T,N,<:Any,CHARGES,SIZES,CHARGE}) where
        {S,T,N,CHARGES,SIZES,CHARGE}
    iszero(N) && return convert(Array{S}, first(values(tensor(A))))
    cumdims = ntuple(i -> prepend!(cumsum(sizes(A,i)),0) , N)
    array = zeros(S, last.(cumdims)...)
    for (sector, degeneracy) in tensor(A)
        inds   = map(chargeindex, sector.chs, charges(A))
        ranges = map((i, cdims) -> cdims[i]+1 : cdims[i+1], inds, cumdims)
        array[ranges...] = degeneracy
    end
    array
end

toarray(a::DASTensor{T}) where {T} = convert(Array{T}, a)

Base.convert(::Type{DTensor{S}}, A::DASTensor) where S = DTensor(convert(Array{S}, A))
todense(A::DASTensor{T,N}) where {T,N} = convert(DTensor{T},A)

function LA.diag(A::DASTensor{T,2}) where T # TODO!!
    ks = sort!(collect(keys(A)))
    return reduce(vcat, LA.diag(A[k]) for k in ks)
end

#= Equalities =#
function Base.:(==)(A::TT, B::TT) where {TT <: DASTensor{T,N}} where {T,N}
    issimilar(A, B) && _tensor_equ(tensor(A), tensor(B))
end

Base.:(==)(A::DASTensor, B::DASTensor) = false

function Base.:(≈)(A::DASTensor{T,N}, B::DASTensor{T,N}) where {T,N}
    issimilar(A, B) && _tensor_approx(A, B)
end

Base.:(≈)(A::DASTensor, B::DASTensor) = false

function _tensor_equ(A::DASTensor, B::DASTensor)
    keys(A) == keys(B) || return false
    all(k -> A[k] == B[k], keys(A))
end

function _tensor_approx(A::DASTensor, B::DASTensor)
    keys(A) == keys(B) || return false
    all(k -> A[k] ≈ B[k], keys(A))
end

_tensor_equ(A, B) = false

function issimilar(A::DASTensor, B::DASTensor)
    return  charge(A) == charge(B) &&
            charges(A) == charges(B) &&
            sizes(A) == sizes(B) &&
            in_out(A) == in_out(B)
end

Base.:(≈)(A::DASTensor, B) = false
Base.:(≈)(A, B::DASTensor) = false

#= Copy =#
function Base.copy!(dest::TT, source::TT) where {TT <: DASTensor{T,N}} where {T,N}
    setcharges!(dest, charges(source))
    setsizes!(dest, deepcopy(sizes(source)))
    setin_out!(dest, in_out(source))
    settensor!(dest, deepcopy(tensor(source)))
    return dest
end

Base.copy(A::DASTensor) = Base.deepcopy(A)

function Base.copyto!(dest::TT, source::TT) where {TT <: DASTensor}
    setcharges!(dest, charges(source))
    setsizes!(dest, deepcopy(sizes(source)))
    setin_out!(dest, in_out(source))
    setcharge!(dest, charge(source))
    empty!(dest.tensor)
    for (k,v) in source.tensor
        dest.tensor[k] = copy(v)
    end
    return dest
end

Base.similar(A::DASTensor{T,N,SYM}, ::Type{S}) where {T,N,SYM,S} =
    DASTensor{S,N}(SYM, charges(A), deepcopy(sizes(A)), in_out(A),charge(A))

Base.similar(A::DASTensor{T}) where T = similar(A,T)
