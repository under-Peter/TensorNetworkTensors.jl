###Type Definitions
## Symmetries; DAS =  Discrete Abelian Symmetries
abstract type DAS end
abstract type DASCharges end
abstract type DASCharge end
Base.hash(a::DASCharge, h::UInt) = hash(a.ch, h)

⊕(a::DASCharges) = a
⊕(a::T, b::T, c::T...) where {T<:DASCharges} where N = ⊕((a ⊕ b),c...)

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

include("U1.jl")
include("ZN.jl")
include("NDAS.jl");

#Sector
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
charge(s::DASSector) = reduce(⊕, s.chs)
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
allsectors(a::NTuple{N,DASCharges}) where N = (DASSector(s...) for s in Iterators.product(a...))
covariantsectors(charges, io::InOut, ch = zero(eltype(first(charges)))) =
    Iterators.filter(==(ch) ∘ charge ∘ ⊗(io), allsectors(charges))
invariantsectors(charges, io::InOut) = covariantsectors(charges, io)

# TENSOR
mutable struct DASTensor{T,N,SYM,CHARGES,SIZES,CHARGE} <: AbstractTensor{T,N}
    chs     ::NTuple{N, CHARGES}
    dims    ::NTuple{N, SIZES}
    io      ::InOut{N}
    tensor  ::Dict{DASSector{N,CHARGE},Array{T,N}}
end

function Base.show(io::IO, A:: DASTensor{T,N,SYM}) where {T,N,SYM}
    print(io, "DASTensor{",T,",",N,",",SYM,"}\n")
    print(io, "charges: ", charges(A), "\n")
    print(io, "sizes: ",   sizes(A), "\n")
    print(io, "in/out: ",  in_out(A), "\n")
    print(io, "Tensors ",  eltype(A), "\n")
end

##Functions
DASTensor{T,N}(sym::ST,charges, dims, io, dict=Dict()) where {T,N,ST <: DAS} =
    DASTensor{T,N,sym,chargestype(sym),Vector{Int},chargetype(sym)}(charges,dims,io,dict)

charges(A::DASTensor)        = A.chs
charges(A::DASTensor,i::Int) = A.chs[i]
charges(A::DASTensor,i)      = TT.getindices(A.chs,i)

sizes(A::DASTensor)          = A.dims
sizes(A::DASTensor,i::Int)   = A.dims[i]
sizes(A::DASTensor,i)        = TT.getindices(A.dims,i)

in_out(A::DASTensor)    = A.io
in_out(A::DASTensor,i)  = A.io[i]

tensor(A::DASTensor)    = A.tensor
Base.getindex(A::DASTensor, i) = A.tensor[i]
Base.setindex!(A::DASTensor{T,N}, t, i::DASSector{N}) where {T,N} =
    (A.tensor[i] = t; A)

Base.values(A::DASTensor) = values(A.tensor)
Base.keys(A::DASTensor) = keys(A.tensor)
Base.haskey(A::DASTensor,k) = haskey(A.tensor,k)
Base.eltype(A::DASTensor{T,N}) where {T,N} = T
Base.ndims(A::DASTensor{T,N}) where {T,N} = N

setcharges!(A::DASTensor, charges) = (A.chs = charges; A)
setsizes!(A::DASTensor, sizes)     = (A.dims   = sizes;   A)
setin_out!(A::DASTensor, in_out)   = (A.io  = in_out;  A)
settensor!(A::DASTensor, tensor)   = (A.tensor  = tensor;  A)

function Base.adjoint(A::DASTensor)
    B = apply(A, conj!)
    B.io = inv(B.io)
    return B
end

function charge(a::DASTensor)
    io = in_out(a)
    och = charge(io ⊗ first(keys(a)))
    for k in keys(a)
        och == charge(io ⊗ k) || throw(ArgumentError("Tensor does not have unique charge"))
    end
    return och
end

isinvariant(a::DASTensor) = all(iszero ∘ charge ∘ ⊗(in_out(a)), keys(a))

chargesize(charge::DASCharge, charges, dims) = dims[chargeindex(charge, charges)]
chargesize(A::DASTensor, i, charge)          = chargesize(charge, charges(A,i), sizes(A,i))
degeneracysize(sector, charges, dims)        = map(chargesize, sector, charges, dims)

function initwith!(A::DASTensor{T,N}, fun, ch = zero(eltype(first(charges(A))))) where {T,N}
    empty!(A.tensor)
    dims, chs = sizes(A), charges(A)
    for k in covariantsectors(charges(A), in_out(A), ch)
        A.tensor[k] = fun(T, degeneracysize(k, chs, dims)...)
    end
    return A
end

function Base.rand(::Type{DASTensor{T,N}}, sym, charges, dims, io,
                    ch = zero(eltype(first(charges)))) where {T,N}
    A = DASTensor{T,N}(sym, charges, dims, io)
    initwithrand!(A, ch)
end

initwithzero!(A, ch = zero(eltype(first(charges(A))))) = initwith!(A, zeros, ch)
initwithrand!(A, ch = zero(eltype(first(charges(A))))) = initwith!(A, rand,  ch)

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

function diag(A::DASTensor{T,2}) where T # TODO!!
    ks = sort!(collect(keys(A)))
    return reduce(vcat, diag(A[k]) for k in ks)
end

#= Equalities =#
function Base.:(==)(A::TT, B::TT) where {TT <: DASTensor{T,N}} where {T,N}
    _equality_helper(A, B) && _tensor_equ(tensor(A), tensor(B))
end

Base.:(==)(A::DASTensor, B::DASTensor) = false

function Base.:(≈)(A::DASTensor{T,N}, B::DASTensor{T,N}) where {T,N}
    _equality_helper(A, B) && _tensor_approx(A, B)
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

function _equality_helper(A, B)
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
    for (k,v) in source.tensor
        if haskey(dest.tensor,k)
            copyto!(dest[k], v)
        else
            dest[k] = copy(v)
        end
    end
end

Base.similar(A::DASTensor{T,N,SYM}, ::Type{S}) where {T,N,SYM,S} =
    DASTensor{S,N}(SYM, charges(A), deepcopy(sizes(A)), in_out(A))

Base.similar(A::DASTensor{T}) where T = similar(A,T)

#= Operations on Tensors =#
function apply!(A::DASTensor, fun!)
    foreach(fun!, values(A))
    return A
end
apply(A::DASTensor, fun!) = apply!(deepcopy(A), fun!)

Base.:(*)(α::Number, A::DASTensor) = apply(A, x -> x .= α .* x)
Base.:(*)(A::DASTensor, α::Number) = α   * A
Base.:(/)(A::DASTensor, α::Number) = 1/α * A

Base.conj!(A::DASTensor) = apply!(A, conj!)
Base.conj(A::DASTensor)  = apply(A, conj!)

include("tensoroperations.jl")
include("splitfuse.jl")
