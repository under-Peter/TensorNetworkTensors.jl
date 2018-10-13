using Base.Iterators: product, filter
import TensorOperations: ndims, similar_from_indices, add!, trace!, contract!, scalar, numind

abstract type DASTensor{T,N} <: AbstractTensor{T,N} end

include("ZNTensors.jl")
include("U1Tensors.jl")

#= Print =#
function Base.show(io::IO, A::TT) where {TT<:DASTensor}
    print(io, printname(TT))
    println(io)
    print(io, "charges: ", charges(A))
    println(io)
    print(io, "sizes: ", sizes(A))
    println(io)
    print(io, "in/out: ", in_out(A))
    println(io)
    print(io, "Tensors", eltype(A.tensor))
end

#= Rand =#
_get_sectors(charges, in_out, filterfun) = (sector for sector in product(charges...)
                                                   if filterfun(sector, in_out))

function _get_degeneracy(charges, sector::NTuple{N,Int}, sizes, ::Type{T}) where {N,T}
    dsizes = map((si,se,ch) -> si[findfirst(isequal(se), ch)],sizes,sector,charges)
    return rand(T, dsizes...)
end

function Base.rand(::Type{TT}, charges, dims, in_out) where {TT <: DASTensor{S}} where S
    #singular charges are allowed to have covariant tensors
    nonsing = collect(map(!isequal(1)∘length, charges))
    a = TT(charges[nonsing], dims[nonsing], in_out[nonsing])
    sectors = _get_sectors(charges, in_out, filterfun(TT))
    for sector in sectors
        a.tensor[sector[nonsing]] = _get_degeneracy(charges[nonsing],
                                                    sector[nonsing],
                                                    dims[nonsing], S)
    end
    return a
end

#= Convert =#
function Base.convert(::Type{DTensor{S}}, A::DASTensor{T,N}) where {S,T,N}
    iszero(N) && return DTensor(convert(Array{S}, tensor(A)[()]))

    cumdims = (prepend!(cumsum(d),0) for d in sizes(A))
    degenrange = [map((x, y) -> x+1:y, cd[1:end-1], cd[2:end]) for cd in cumdims]
    rangedict = Dict{NTuple{2,Int},UnitRange}()
    for (i, charges) in enumerate(charges(A)), (j, charge) in enumerate(charges)
        rangedict[(i, charge)] = degenrange[i][j]
    end
    array = zeros(S, map(last, cumdims)...)
    for (sector, degeneracy) in tensor(A)
        indexrange = [rangedict[(i, s)] for (i, s) in enumerate(sector)]
        array[indexrange...] = degeneracy
    end
    DTensor{S,N}(array)
end

todense(A::DASTensor{T,N}) where {T,N} = convert(DTensor{T},A)

#getters
@inline charges(A::DASTensor) = A.charges
@inline sizes(A::DASTensor)   = A.sizes
@inline in_out(A::DASTensor)  = A.in_out
@inline tensor(A::DASTensor)  = A.tensor

@inline charges(A::DASTensor,i::Int) = A.charges[i]
@inline charges(A::DASTensor,i) = TT.getindices(A.charges,i)
@inline sizes(A::DASTensor,i::Int)   = A.sizes[i]
@inline sizes(A::DASTensor,i)   = TT.getindices(A.sizes,i)
@inline chargesize(A::DASTensor, ch, i::Int) = A.sizes[i][findfirst(isequal(ch), A.charges[i])]
@inline in_out(A::DASTensor,i::Int)  = A.in_out[i]
@inline in_out(A::DASTensor,i)  = TT.getindices(A.in_out,i)
@inline tensor(A::DASTensor,i)  = A.tensor[i]
@inline Base.getindex(A::DASTensor, i) = A.tensor[i]
@inline Base.setindex!(A::DASTensor{T,N}, t, i::NTuple{N}) where {T,N} = (A.tensor[i] = t; A)

#setters
@inline setcharges!(A::DASTensor, charges) = (A.charges = charges; A)
@inline setsizes!(A::DASTensor, sizes)     = (A.sizes   = sizes;   A)
@inline setin_out!(A::DASTensor, in_out)   = (A.in_out  = in_out;  A)
@inline settensor!(A::DASTensor, tensor)   = (A.tensor  = tensor;  A)

Base.eltype(A::DASTensor{T,N}) where {T,N} = T
Base.ndims(A::DASTensor{T,N}) where {T,N} = N
TensorOperations.numind(A::DASTensor{T,N}) where {T,N} = N
function Base.adjoint(A::DASTensor)
    B = apply(A, conj!)
    B.in_out = -1 .* B.in_out
    B
end

#= Equalities =#
function Base.:(==)(A::TT, B::TT) where {TT <: DASTensor{T,N}} where {T,N}
    _equality_helper(A, B) &&
    _tensor_equ(tensor(A), tensor(B))
end

Base.:(==)(A::DASTensor, B::DASTensor) = false

function Base.:(≈)(A::DASTensor{T,N}, B::DASTensor{T,N}) where {T,N}
    _equality_helper(A, B) &&
    _tensor_approx(tensor(A), tensor(B))
end

Base.:(≈)(A::DASTensor, B::DASTensor) = false

function _tensor_equ(A::Dict{V,K}, B::Dict{V,K}) where {V,K}
    kA = keys(A)
    kB = keys(B)
    kAB = intersect(kA, kB)
    kAonly = setdiff(kA, kB)
    kBonly = setdiff(kB, kA)
    return  all([A[k] == B[k] for k in kAB]) &&
            all([iszero(A[k]) for k in kAonly]) &&
            all([iszero(B[k]) for k in kBonly])
end

function _tensor_approx(A::Dict{V,K}, B::Dict{V,K}) where {V,K}
    kA = keys(A)
    kB = keys(B)
    kAB = intersect(kA, kB)
    kAonly = setdiff(kA, kB)
    kBonly = setdiff(kB, kA)
    return all([A[k] ≈ B[k] for k in kAB]) &&
        all([isapprox(A[k], zero(A[k]), atol = 10^-14.) for k in kAonly]) &&
        all([isapprox(B[k], zero(B[k]), atol = 10^-14.) for k in kBonly])
end

_tensor_equ(A, B) = false

function _equality_helper(A, B)
    return  charges(A) == charges(B) &&
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
    for (k,v) in source.tensor
        if haskey(dest.tensor,k)
            copyto!(dest[k], v)
        else
            dest[k] = copy(v)
        end
    end
end

function apply!(A::DASTensor, op)
    for degeneracy in values(tensor(A))
        degeneracy .= op(degeneracy)
    end
    return A
end

similar_from_indices(T::Type, p1::Tuple, p2::Tuple, A::DASTensor, CA::Type{<:Val}) =
    similar_from_indices(T, (p1...,p2...), A, CA)

#= Operations on Tensors =#
apply(A::DASTensor, op) = apply!(deepcopy(A), op)

Base.:(*)(α::Number, A::DASTensor) = apply(A, x -> α .* x)
Base.conj!(A::DASTensor) = apply!(A, conj!)
Base.conj(A::DASTensor) = apply(A, conj!)

function add!(α::Number, A::DASTensor{T,N}, conjA::Type{Val{CA}},
     β::Number, C::DASTensor{S,N}, indCinA) where {T,S,N,CA}
    #conditions
    perm = TT.sortperm(indCinA)
    in_out(A) == TT.getindices(in_out(C), perm) ||
        throw( ArgumentError("Leg directions don't agree"))
    charges(A) == TT.getindices(charges(C), perm) ||
        throw( ArgumentError("charges don't agree"))
    sizes(A) == TT.getindices(sizes(C), perm) ||
        throw( ArgumentError("sizes don't agree"))

    for (sector, degeneracy) in tensor(A)
        permsector = TT.permute(sector, perm)
        if haskey(tensor(C), permsector)
            add!( α, degeneracy, conjA, β, C[permsector], indCinA)
        else
            C[permsector] = similar_from_indices(T, indCinA, degeneracy)
            add!( α, degeneracy, conjA, 0, C[permsector], indCinA)
        end
    end
    return C
end

add!(α, A::DASTensor, CA, β, C::DASTensor, p1, p2) = add!(α, A, CA, β, C, (p1...,p2...))

function trace!(α, A::DASTensor{T,N}, ::Type{Val{CA}}, β, C::DASTensor{S,M},
                indCinA, cindA1, cindA2) where {T,N,S,M,CA}
    #conditions
    length(cindA1) == length(cindA2) ||
        throw(ArgumentError("indices to contract don't pair up") )
    all(in_out(A,i) == -in_out(A,j) for (i, j) in zip(cindA1, cindA2)) ||
        throw(ArgumentError("leg directions don't agree"))
    all(charges(A,i) == charges(A,j) for (i, j) in zip(cindA1, cindA2)) ||
        throw(ArgumentError("charges don't agree"))

    sectors = filter(x -> isequal(TT.getindices(x, cindA1), TT.getindices(x, cindA2)),
                        collect(keys(tensor(A))))
    newsectors = map(x -> TT.deleteat(x, TT.vcat(cindA1, cindA2)), sectors)
    passedset = Set{}()
    sizehint!(tensor(C), length(tensor(C)) + length(newsectors))
    for (sector, newsector) in zip(sectors, newsectors)
        array = A[sector]
        if haskey(tensor(C), newsector)
            if !in(newsector, passedset)
                trace!(α, array, Val{CA}, β, C[newsector], indCinA, cindA1, cindA2)
                push!(passedset, newsector)
            else
                trace!(α, array, Val{CA}, 1, C[newsector], indCinA, cindA1, cindA2)
            end
        else
            C[newsector] = similar_from_indices(T, indCinA, array)
            trace!(α, array, Val{CA}, 0, C[newsector], indCinA, cindA1, cindA2)
            push!(passedset, newsector)
        end
    end
    return C
end
trace!(α, A::DASTensor, CA, β, C::DASTensor, p1, p2, cindA1, cindA2) =
    trace!(α, A, CA, β, C, (p1..., p2...), cindA1, cindA2)

function _getnewsector(sectorA, sectorB, oindA, oindB, indCinoAB)
    TT.permute(TT.vcat(
        TT.getindices(sectorA, oindA),
        TT.getindices(sectorB, oindB)),
        indCinoAB)
end

function contract!(α, A::DASTensor{TA,NA}, ::Type{Val{CA}},
                      B::DASTensor{TB,NB}, ::Type{Val{CB}}, β,
                      C::DASTensor{TC,NC}, oindA, cindA, oindB, cindB,
                      indCinoAB, ::Type{Val{M}}=Val{:native}) where
                      {TA,NA,TB,NB,TC,NC,CA,CB,M}
    #conditions
    length(cindA) == length(cindB) ||
        throw(ArgumentError("indices to contract don't tpair up"))
    all(in_out(A,i) == -in_out(B,j) for (i, j) in zip(cindA, cindB)) ||
        throw(ArgumentError("leg directions don't agree"))
    all(charges(A,i) == charges(B,j) for (i, j) in zip(cindA, cindB)) ||
        throw( ArgumentError("charges don't agree"))

    oinAB = TT.vcat(oindA, .+(oindB, NA))
    indCinAB = map(x -> oinAB[x], indCinoAB)
    sectorscA = groupby(x -> TT.getindices(x, cindA), keys(tensor(A)))
    sectorscB = groupby(x -> TT.getindices(x, cindB), keys(tensor(B)))
    # collect sectors that contract with each other
    sectorscAB = intersect(keys(sectorscA), keys(sectorscB))
    passedset = Set()
    for sec in sectorscAB
        for sectorA in sectorscA[sec], sectorB in sectorscB[sec]
            newsector = _getnewsector(sectorA, sectorB, oindA, oindB, indCinoAB)
            if haskey(tensor(C), newsector)
                if !in(newsector, passedset) #firstpass
                    push!(passedset, newsector)
                    contract!(α, A[sectorA], Val{CA},
                                 B[sectorB], Val{CB},
                              β, C[newsector],
                              oindA, cindA, oindB, cindB, indCinoAB, Val{M})
                else
                    contract!(α, A[sectorA], Val{CA},
                                 B[sectorB], Val{CB},
                              1, C[newsector],
                              oindA, cindA, oindB, cindB, indCinoAB, Val{M})
                 end
             else
                C[newsector] = similar_from_indices(TC, oindA, oindB, indCinoAB, (),
                                    A[sectorA], B[sectorB], Val{:N}, Val{:N})

                contract!(α, A[sectorA], Val{CA},
                             B[sectorB], Val{CB},
                          0, C[newsector],
                          oindA, cindA, oindB, cindB, indCinoAB, Val{M})
                push!(passedset, newsector)
             end
         end
     end
     return C
 end

contract!(α, A::DASTensor, CA, B::DASTensor, CB, β, C::DASTensor,
    oindA, cindA, oindB, cindB, p1, p2, method::Type{<:Val} = Val{:BLAS}) =
    contract!(α, A, CA, B, CB, β, C, oindA, cindA, oindB, cindB, (p1..., p2...), method)

#= Reshaping =#
