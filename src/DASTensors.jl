using Base.Iterators: product, filter
using TupleTools
const TT = TupleTools
import TensorOperations: ndims, similar_from_indices, add!, trace!, contract!, scalar, numind
import Base: iszero, -, ==, isless, getindex, vcat, length

abstract type DASTensor{T,N} <: AbstractTensor{T,N} end

abstract type DASCharges end
⊕(a::DASCharges) = a
⊕(a::T, b::T, c::T...) where {T<:DASCharges} = ⊕((a ⊕ b),c...)

function Base.iterate(a::DASCharges,i = 1)
    if i <= length(a)
        return (a[i], i+1)
    else
        return nothing
    end
end

abstract type DASCharge end
iszero(a::DASCharge) = iszero(a.ch)
⊕(a::DASCharge) = a
⊕(a::DASCharge, b::DASCharge, c::DASCharge...) = ⊕(a ⊕ b, c...)
-(a::DASCharge) = -1 ⊗ a
isless(a::DASCharge, b::DASCharge) = isless(a.ch, b.ch)
==(a::DASCharge, b::DASCharge) = a.ch == b.ch

abstract type DASSector{N} end
getindex(s::DASSector,i::Int) = s.chs[i]
permute(s::T,perm) where {T <: DASSector} = T(TT.permute(s.chs,perm))
charge(s::DASSector) = ⊕(s.chs...)
⊕(a::T, b::T) where {T <: DASSector} = T(a.chs .⊕ b.chs)
function isless(a::DASSector{N}, b::DASSector{N}) where N
    @inbounds for i in 1:N
        a.chs[i] < b.chs[i] && return true
        b.chs[i] < a.chs[i] && return false
    end
    return false
end
==(a::DASSector{N}, b::DASSector{N}) where N = a.chs == b.chs
==(a::DASSector, b::DASSector) = false

vcat(s::DASSector) = s
vcat(s1::DASSector, s2::DASSector, s3::DASSector...) = vcat(vcat(s1,s2),s3...)

struct InOut{N}
    v::NTuple{N,Int}
    InOut{N}(i::Int) where N = new{1}((i,))
    InOut(i::Int) = new{1}((i,))
    function InOut{N}(v::NTuple{N,Int}) where N
        all(in((1,-1)), v) || throw(ArgumentError("in-out = $v !∈ (1,-1)"))
        new{N}(v)
    end
    InOut(v::NTuple{N}) where N = InOut{N}(v)
end


⊗(a::InOut{M}, b::T) where {M, T <: DASSector} = T(a.v .⊗ b.chs)
⊗(a::Int, b::T) where {T<:DASCharge} = T(a * b.ch)
⊗(io::InOut{N}, ss::NTuple{L,T}) where {N,L,T<:DASSector} = ntuple(i -> io ⊗ ss[i], L)
⊗(io::InOut) = x -> io ⊗ x
-(io::InOut) = InOut(-1 .* io.v)
getindex(a::InOut, i::Int) = InOut{1}(a.v[i])
getindex(a::InOut, i::NTuple{L}) where L = InOut{L}(TT.getindices(a.v,i))
length(::InOut{L}) where L = L
vcat(a::InOut) = a
vcat(a::InOut{L1}, b::InOut{L2}) where {L1,L2} = InOut{L1+L2}(TT.vcat(a.v,b.v))
vcat(a::InOut, b::InOut, c::InOut...) = vcat(vcat(a,b),c...)

filterfun = iszero ∘ charge
invariantsectors(charges, io::InOut) = Iterators.filter(filterfun ∘ ⊗(io), allsectors(charges))
charge(a::DASTensor) = charge( in_out(a) ⊗ first(keys(a)))
isinvariant(a::DASTensor) = all(iszero ∘ charge, keys(a))
scalar(a::DASTensor) = scalar(first(values(a)))

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
    print(io, "Tensors ", eltype(A))
end

#= Rand =#
function _get_degeneracy(charges, sector, sizes, ::Type{T}) where {T}
    dsizes = map((si,se,ch) -> si[chargeindex(se,ch)], sizes, sector.chs, charges)
    return rand(T, dsizes...)
end

function Base.rand(::Type{T}, chs, dims, io) where {T <: DASTensor{S}} where S
    #singular charges are allowed to have covariant tensors
    sing = map(isequal(1)∘length, chs)
    inds = tuple(findall(.!sing)...)
    a = T(TT.getindices(chs,inds),
            TT.getindices(dims,inds),
            TT.getindices(io,inds))
    for sector in invariantsectors(charges(a), in_out(a))
        a[sector[inds]] =
            _get_degeneracy(
                charges(a,inds),
                sector[inds],
                TT.getindices(dims,inds), S)
    end
    return a
end

#= Convert =#
function Base.convert(::Type{DTensor{S}}, A::DASTensor{T,N}) where {S,T,N}
    iszero(N) && return DTensor(convert(Array{S}, first(values(tensor(A)))))

    cumdims = (prepend!(cumsum(d),0) for d in sizes(A))
    degenrange = [map((x, y) -> x+1:y, cd[1:end-1], cd[2:end]) for cd in cumdims]
    rangedict = Dict{Tuple{Int,Union{U1Charge,ZNCharge}},UnitRange}()
    for (i, chs) in enumerate(charges(A)), j in 1:length(chs)
        rangedict[(i, chs[j])] = degenrange[i][j]
    end
    array = zeros(S, map(last, cumdims)...)
    for (sector, degeneracy) in tensor(A)
        indexrange = [rangedict[(i, s)] for (i, s) in enumerate(sector.chs)]
        array[indexrange...] = degeneracy
    end
    DTensor{S,N}(array)
end

todense(A::DASTensor{T,N}) where {T,N} = convert(DTensor{T},A)

function diag(A::DASTensor{T,2}) where T
    ks = sort(collect(keys(A)))
    return reduce(vcat, diag(A[k]) for k in ks)
end

#getters
@inline charges(A::DASTensor) = A.charges
@inline sizes(A::DASTensor)   = A.sizes
@inline in_out(A::DASTensor)  = A.in_out
@inline tensor(A::DASTensor)  = A.tensor

@inline charges(A::DASTensor,i::Int) = A.charges[i]
@inline charges(A::DASTensor,i) = TT.getindices(A.charges,i)
@inline sizes(A::DASTensor,i::Int)   = A.sizes[i]
@inline sizes(A::DASTensor,i)   = TT.getindices(A.sizes,i)
@inline in_out(A::DASTensor,i)  = A.in_out[i]
@inline tensor(A::DASTensor,i)  = A.tensor[i]

@inline Base.getindex(A::DASTensor, i) = A.tensor[i]
@inline Base.setindex!(A::DASTensor{T,N}, t, i::DASSector{N}) where {T,N} = (A.tensor[i] = t; A)
@inline Base.values(A::DASTensor) = values(A.tensor)
@inline Base.keys(A::DASTensor) = keys(A.tensor)

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
    B.in_out = -B.in_out
    return B
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
    for degeneracy in values(A)
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

function _errorsadd(A, C, perm::NTuple{N}) where N
    mask = in_out(A).v .== in_out(C, perm).v
    for (m, iA, iC) in zip(mask, 1:N, perm)
        charges(A, iA) == ifelse(m, charges(C, iC), -charges(C, iC)) ||
            throw(ArgumentError("charges don't agree"))
        sizes(A, iA) == ifelse(m,sizes(C, iC), reverse(sizes(C,iC))) ||
            throw(DimensionMismatch())
    end
    modio = InOut(ntuple(i -> ifelse(mask[i],1,-1),N))
    return ⊗(modio)
end

function add!(α::Number, A::DASTensor{T,N}, conjA::Type{Val{CA}},
     β::Number, C::DASTensor{S,N}, indCinA) where {T,S,N,CA}
    perm = TT.sortperm(indCinA)
    maskfun = _errorsadd(A, C, perm)

    for (sector, degeneracy) in tensor(A)
        permsector = permute(maskfun(sector), perm)
        if haskey(tensor(C), permsector)
            add!(α, degeneracy, conjA, β, C[permsector], indCinA)
        else
            C[permsector] = similar_from_indices(T, indCinA, degeneracy)
            add!(α, degeneracy, conjA, 0, C[permsector], indCinA)
        end
    end
    return C
end

add!(α, A::DASTensor, CA, β, C::DASTensor, p1, p2) = add!(α, A, CA, β, C, (p1...,p2...))

function _errorstrace(A::DASTensor{T,N}, cindA1::NTuple{M,Int},
                                         cindA2::NTuple{M,Int},
                                         C::DASTensor{TC,NC},
                                         indCinA::NTuple{NC}) where {M,T,N,TC,NC}
    maskA = in_out(A, cindA1).v .== (-in_out(A, cindA2)).v
    for (m, iA1, iA2) in zip(maskA, cindA1, cindA2) #trace in A
        charges(A, iA1) == ifelse(m, charges(A, iA2), -charges(A,iA2)) ||
            throw(ArgumentError("charges don't agree"))
        sizes(A, iA1) == ifelse(m, sizes(A, iA2), reverse(sizes(A,iA2))) ||
            throw(DimensionMismatch())
    end
    indsA = TT.sort(indCinA)
    perm = TT.sortperm(indCinA)

    maskC = in_out(A, indsA).v .== in_out(C, perm).v
    for (m, iA, iC) in zip(maskC, indsA, perm) #trace in A
        charges(A, iA) == ifelse(m, charges(C, iC), -charges(C,iC)) ||
            throw(ArgumentError("charges don't agree"))
        sizes(A, iA) == ifelse(m, sizes(C, iC), reverse(sizes(C,iC))) ||
            throw(DimensionMismatch())
    end
    maskAio = InOut{M}(ntuple(i -> ifelse(maskA[i],1,-1), M))
    maskCio = InOut{NC}(ntuple(i -> ifelse(maskC[i],1,-1), NC))
    return (⊗(maskAio),
            ⊗(maskCio))
end

function trace!(α, A::DASTensor{T,N}, ::Type{Val{CA}}, β, C::DASTensor{S,M},
                indCinA, cindA1, cindA2) where {T,N,S,M,CA}
    #conditions
    maskAfun, maskCfun = _errorstrace(A, cindA1, cindA2, C, indCinA)

    perm = TT.sortperm(indCinA)
    sectors = filter(x -> isequal(maskAfun(x[cindA1]), x[cindA2]), keys(A))
    cinds = TT.vcat(cindA1, cindA2)
    t = typeof(maskCfun(permute(deleteat(first(sectors), cinds),perm)))
    passedset = Vector{t}() #might be slower for more elements
    for sector in sectors
        newsector = maskCfun(permute(deleteat(sector, cinds),perm))
        if haskey(tensor(C), newsector)
            if !in(newsector, passedset)
                trace!(α, A[sector], Val{CA}, β, C[newsector], indCinA, cindA1, cindA2)
                push!(passedset, newsector)
            else
                trace!(α, A[sector], Val{CA}, 1, C[newsector], indCinA, cindA1, cindA2)
            end
        else
            C[newsector] = similar_from_indices(T, indCinA, A[sector])
            trace!(α, A[sector], Val{CA}, 0, C[newsector], indCinA, cindA1, cindA2)
            push!(passedset, newsector)
        end
    end
    return C
end

trace!(α, A::DASTensor, CA, β, C::DASTensor, p1, p2, cindA1, cindA2) =
    trace!(α, A, CA, β, C, (p1..., p2...), cindA1, cindA2)


function _errorscontract(A::DASTensor{TA,NA}, (oindA, cindA)::Tuple{NTuple{NoA,Int}, NTuple{CoA,Int}},
                        B::DASTensor{TB,NB}, (oindB, cindB)::Tuple{NTuple{NoB,Int}, NTuple{CoB,Int}},
                        C::DASTensor{TC,NC}, indCinoAB::NTuple{NoC,Int}) where {NA, NB, NC, TA, TB, TC, NoA, CoA, NoB, CoB, NoC}
    CoA == CoB || throw(ArgumentError("indices to contract don't pair up"))
    NoA + NoB == NoC || throw(ArgumentError("indices to contract don't pair up"))

    maskB = in_out(A, cindA).v .== (-in_out(B, cindB)).v
    for i in 1:CoA
        m = maskB[i]
        iA = cindA[i]
        iB = cindB[i]
        charges(A,iA) == ifelse(m, charges(B,iB), -charges(B,iB)) ||
            throw(ArgumentError("charges don't agree"))
        sizes(A,iA) == ifelse(m, sizes(B,iB), reverse(sizes(B,iB))) ||
            throw(DimensionMismatch())
    end
    ioAB  = vcat(in_out(A, oindA), in_out(B, oindB))
    chsAB = TT.vcat(charges(A, oindA), charges(B, oindB))
    dsAB  = TT.vcat(sizes(A, oindA), sizes(B, oindB))

    maskAB = ioAB[indCinoAB].v .== in_out(C).v
    for i in 1:NoC
        m = maskAB[i]
        iAB = indCinoAB[i]
        iC = i
        chsAB[iAB] == ifelse(m, charges(C, iC), -charges(C,iC)) ||
            throw(ArgumentError("charges don't agree"))
        dsAB[iAB] == ifelse(m, sizes(C,iC), reverse(sizes(C,iC))) ||
         throw(DimensionMismatch())
    end

    maskBio = InOut{CoA}(tuple(ifelse.(maskB,1,-1)...))
    maskABio = InOut{NoC}(tuple(ifelse.(maskAB,1,-1)...))
    return (⊗(maskBio), ⊗(maskABio))
end

function contract!(α, A::DASTensor{TA,NA}, ::Type{Val{CA}},
                      B::DASTensor{TB,NB}, ::Type{Val{CB}}, β,
                      C::DASTensor{TC,NC}, oindA, cindA, oindB, cindB,
                      indCinoAB, ::Type{Val{M}}=Val{:native}) where
                      {TA,NA,TB,NB,TC,NC,CA,CB,M}
    #conditions
    maskBfun, maskABfun = _errorscontract(A, (oindA, cindA), B, (oindB, cindB), C, indCinoAB)

    oinAB = TT.vcat(oindA, oindB .+ NA)
    indCinAB = TT.getindices(oinAB, indCinoAB)
    secsA = groupby(x -> x[cindA], keys(A))
    secsB = groupby(x -> maskBfun(x[cindB]), keys(B))

    # collect sectors that contract with each other
    secsAB = intersect(keys(secsA), keys(secsB))
    passedset = Set()
    for sector in secsAB
        for secA in secsA[sector], secB in secsB[sector]
            newsector = maskABfun(permute(vcat(secA[oindA], secB[oindB]), indCinoAB))
            if haskey(tensor(C), newsector)
                if !in(newsector, passedset) #firstpass
                    push!(passedset, newsector)
                    contract!(α, A[secA], Val{CA}, B[secB], Val{CB},
                              β, C[newsector],
                              oindA, cindA, oindB, cindB, indCinoAB, Val{M})
                else
                    contract!(α, A[secA], Val{CA}, B[secB], Val{CB},
                              1, C[newsector],
                              oindA, cindA, oindB, cindB, indCinoAB, Val{M})
                 end
             else
                C[newsector] = similar_from_indices(TC, oindA, oindB, indCinoAB, (),
                                    A[secA], B[secB], Val{:N}, Val{:N})

                contract!(α, A[secA], Val{CA}, B[secB], Val{CB},
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
#== Fusion ==#
_ktype(::ZNTensor{T,L,N}, M) where {T,L,N} = ZNSector{M,N}
_ktype(::U1Tensor, M) = U1Sector{M}
_chtype(::ZNTensor{T,L,N}) where {T,L,N} = ZNCharge{N}
_chtype(::U1Tensor) where {T,L,N} = U1Charge
_chstype(::ZNTensor{T,L,N}) where {T,L,N} = ZNCharges{N}
_chstype(::U1Tensor) where {T,L,N} = U1Charges

function fusiondict(A::T, indexes::NTuple{N,Int}, ld::Int) where {T<:DASTensor,N}
    #=
    fdict: oldcharges → newcharge & indices
    ddict: newcharges → dimension
    sdict: newcharges → [oldcharges & indices]
    =#
    sdict = Dict{_chtype(A),Vector{Tuple{_ktype(A,N),UnitRange}}}()
    fdict = Dict{_ktype(A,N),Tuple{_chtype(A),UnitRange}}()
    ddict = Dict{_chtype(A),Int}()

    ochs = charges(A, indexes)
    oios = in_out(A, indexes)
    ods  = sizes(A, indexes)

    for chs in allsectors(ochs)
        nch = ld ⊗ charge(oios ⊗ chs)
        if !haskey(sdict, nch)
            sdict[nch] = []
            ddict[nch] = 0
        end
        d = 1
        for i in 1:N
            d *= ods[i][chargeindex(chs[i],ochs[i])]
        end
        push!(sdict[nch], (chs, (1:d) .+ ddict[nch]))
        fdict[chs] = (nch, sdict[nch][end][2])
        ddict[nch] += d
    end
    return (fdict, ddict, sdict)
end

function fusiondict(A::T, i::Int, ld::Int) where {T<:DASTensor}
    sdict = Dict{_chtype(A),Vector{Tuple{_ktype(A,1),UnitRange}}}()
    fdict = Dict{_chtype(A),Tuple{_chtype(A),UnitRange}}()
    ddict = Dict{_chtype(A),Int}()

    for j in 1:length(charges(A,i))
        ch = charges(A,i)[j]
        nch = ld ⊗ (in_out(A,i).v[1] ⊗ ch)
        d = sizes(A,i)[j]
        sdict[nch] = [(_ktype(A,1)((ch,)),1:d)]
        fdict[ch] = (nch, 1:d)
        ddict[nch] = d
    end
    return (fdict, ddict, sdict)
end

function fusiondicts(A, indexes, lds::NTuple{N,Int}) where N
    fds = Dict[]
    sds = Dict[]
    dds = Dict{_chtype(A),Int}[]
    for i in 1:N
        fd, dd, sd = fusiondict(A, indexes[i], lds[i])
        push!(fds, fd)
        push!(dds, dd)
        push!(sds, sd)
    end
    return (fds, dds, sds)
end

function fusefields(A::T, indexes, lds::NTuple{N}, ddicts) where {T<:DASTensor,N}
    ochs = [charges(A,inds) for inds in indexes]
    oios = [in_out(A,inds) for inds in indexes]
    newchs = _chstype(A)[]
    for i in 1:length(indexes)
        if indexes[i] isa Int
            newch = ifelse(oios[i].v[1] == 1, ochs[i], -ochs[i])
            newch = ifelse(lds[i] == 1, newch, -newch)
            push!(newchs, newch)
        else
            newch = ifelse(oios[i][1].v[1] == 1, ochs[i][1], -ochs[i][1])
            for j in 2:length(indexes[i])
                newch = newch ⊕ ifelse(oios[i][j].v[1] == 1, ochs[i][j], -ochs[i][j])
            end
            push!(newchs,ifelse(lds[i]==1, newch, -newch))
        end
    end
    newds  = tuple(map((dict,chs) -> [dict[ch] for ch in chs], ddicts, newchs)...)
    return (tuple(newchs...)::NTuple{N}, newds::NTuple{N,Vector{Int}}, InOut(lds))
end


fuselegs(A::T, indexes::NTuple{M}) where {M,T<:DASTensor} = fuselegs(A, indexes, ntuple(i -> 1,M))
function fuselegs(A::T, indexes, lds::NTuple{M,Int}) where {T<:DASTensor{S,N}, M} where {S,N}
    fdicts, ddicts, sdicts = fusiondicts(A, indexes, lds)
    newfields = fusefields(A, indexes, lds, ddicts)
    perm = TT.vcat(indexes...)
    isperm(perm) || throw(ArgumentError("not valid specification of indexes"))

    ntensor = Dict{_ktype(A,M),Array{S,M}}()
    for (sector, degeneracy) in tensor(A)
        tuples = map((i, fdict) -> fdict[sector[i]], indexes, fdicts)
        nsec, nranges = collect(zip(tuples...))
        nsector = _ktype(A,M)(nsec)
        if !haskey(ntensor, nsector)
            ntensor[nsector] = zeros(S, map(getindex, ddicts, nsector.chs)...)
        end
        s = size(degeneracy)
        dims = map(i -> prod(s[vec(collect(i))]), indexes)
        ntensor[nsector][nranges...] = reshape(permutedims(degeneracy, perm), dims...)
    end
    _totuple(x) = ifelse(typeof(x) <: Tuple, x, (x,))
    ochs = [_totuple(charges(A, i)) for i in indexes]
    oios = [in_out(A, i)  for i in indexes]
    ods  = [_totuple(sizes(A, i))   for i in indexes]
    inverter = (ochs, oios, ods, sdicts)
    return (constructnew(T, newfields, ntensor), inverter)
end


#== Splitting ==#
function splitlegs(A::T, inds::NTuple{M,Union{Int,NTuple{3,Int}}}, inverter) where {M,T<:DASTensor{S}} where S
    nchs, nios, ndims, sdicts = inverter
    cchs, cdims = charges(A), sizes(A)
    ntensor = Dict{_ktype(A,M),Array{S,M}}()
    rinds, iperm, perm = _reduceinds(inds)
    for (sector, degen) in tensor(A)
        nsecs, nranges = _splitsec(sector,rinds,sdicts)
        for i in 1:length(nranges)
            nsec = nsecs[i]
            nrange = nranges[i]
            dims = _splitsdims((sector, cchs, cdims),
                               (nsec, nchs,ndims),
                               inds, perm)
            nsec = nsec[iperm]
            ntensor[nsec] = copy(permutedims(reshape(view(degen, nrange...),
                                        dims...), iperm))
        end
    end
    _pick(i::Int, current, old)  = current[i]
    _pick((l,m,n), current, old) = old[m][n]
    newchs = tuple([_pick(i, charges(A), nchs) for i in inds]...)
    newios = vcat([_pick(i, in_out(A), nios)  for i in inds]...)
    newds  = tuple([_pick(i, sizes(A), ndims)    for i in inds]...)
    return constructnew(T, (newchs,newds,newios), ntensor)
end

function _splitsec(sector, inds, sdicts)
    _pick(i::Int) = ((sector[(i,)], :),)
    _pick((l,m,n)) = sdicts[m][sector[l]]
    nseccharges = [_pick(i) for i in inds]
    nsecs = []
    nranges = []
    for ns in Iterators.product(nseccharges...)
        nsec = vcat([s[1] for s in ns]...)
        push!(nsecs, nsec)
        push!(nranges, [s[2] for s in ns])
    end
    return (nsecs, nranges)
end

function _splitsdims((sector, cchs, cdims), (nsec, nchs, ndims), inds, perm)
    _pick(i::Int, k)  = cdims[i][chargeindex(sector[i], cchs[i])]
    _pick((l,m,n), k) = ndims[m][n][chargeindex(nsec[k], nchs[m][n])]
    [_pick(i,k) for (k,i) in enumerate(TT.getindices(inds,perm))]
end

function _reduceinds(inds)
    rinds = tuple(sort(unique(first, inds), by=first)...)
    tinds = map(i -> ifelse(i isa Int, (i,), i), inds)
    perm  = TT.sortperm(tinds)
    iperm = TT.invperm(perm)
    return (rinds, iperm, perm)
end

#= Functions =#
function tensorsvd(A::T; svdcutfunction = svdcutfun_default) where {T <: DASTensor{Q,N}} where {Q,N}
    N == 2 || throw(ArgumentError("SVD only works on rank 2 tensors"))
    tU = Dict{_ktype(A,N),Array{Q,N}}()
    tS = Dict{_ktype(A,N),Array{Q,N}}()
    tV = Dict{_ktype(A,N),Array{Q,N}}()

    lch = connectingcharge(T, charges(A), in_out(A), charge(A))
    ld =  sizes(A,2)[[chargeindex(c,charges(A,2)) for c in lch]]

    chU = (charges(A,1), lch)
    dU  = deepcopy.((sizes(A,1), ld))
    ioU = in_out(A)

    chS = (lch, lch)
    dS  =  deepcopy.((ld,ld))
    ioS = vcat(-in_out(A,2), in_out(A,2))

    chV = (lch, charges(A,2))
    dV  = deepcopy((ld,sizes(A,2)))
    ioV = vcat(-in_out(A,2), in_out(A,2))

    for (k, degen) in tensor(A)
        in, out = k[1], k[2]
        tU[_ktype(A,2)((in, out))], tS[_ktype(A,2)((out, out))],
            tV[_ktype(A,2)((out, out))], cutoff =
                _tensorsvd(degen, svdcutfunction = svdcutfunction, helper = true)
        dU[2][chargeindex(out,chU[2])] = cutoff
        dS[1][chargeindex(out,chS[1])] = cutoff
        dS[2][chargeindex(out,chS[2])] = cutoff
        dV[1][chargeindex(out,chV[1])] = cutoff
    end
    U = constructnew(T, (chU, dU, ioU), tU)
    S = constructnew(T, (chS, dS, ioS), tS)
    V = constructnew(T, (chV, dV, ioV), tV)
    return (U, S, V)
end

function tensorsvd(A::DASTensor, indexes; svdcutfunction = svdcutfun_default)
    fA, inverter = fuselegs(A, indexes)
    U, S, Vt = tensorsvd(fA, svdcutfunction=svdcutfunction)
    li1 = length(indexes[1])
    if !iszero(li1)
        indxs = (ntuple(x -> (1,(1,x)), li1)..., (2,))
        U = splitlegs(U, indxs, inverter)
    end
    li2 = length(indexes[2])
    if !iszero(li2)
        indxs = ((1,), ntuple(x -> (2,(2,x)), li2)...)
        Vt = splitlegs(Vt, indxs, inverter)
    end
    return (U,S,Vt)
end

function _tensorsvd(A::AbstractArray; svdcutfunction = svdcutfun_default,
        helper::Bool = false)
    F = svd(A)
    svals = F.S
    cutoff = svdcutfunction(svals)
    U = F.U[:, 1:cutoff]
    S = diagm(0=>svals[1:cutoff])
    Vt = F.Vt[1:cutoff, :]
    helper && return (U, S, Vt, cutoff)
    return (U, S, Vt)
end


function connectingcharge(::Type{<:U1Tensor}, (ch1,ch2), io, charge)
    # ch1→[A]→ch2
    ch1r = ifelse(io[1].v[1] ==  1, -ch1, ch1).v .- charge.ch
    ch2r = ifelse(io[2].v[1] == -1, -ch2, ch2).v
    ch3r = intersect(ch1r, ch2r)
    io[2].v[1] == -1 && return U1Charges(-ch3r)
    return U1Charges(ch3r)
end

function connectingcharge(::Type{<:ZNTensor}, (ch1,ch2), ios, charge)
    ch1 == ch2 || throw(ArgumentError("illegal connection: $ch1↔$ch2"))
    return ch1
end

svdcutfun_default = length
svdcutfun_discardzero = x -> length(filter(!iszero, x))
svdcutfun_maxχ(χ) = x -> min(length(x), χ)
svdcutfun_maxcumerror(ϵ; χ::Int = Inf64) = x -> _maxcumerror(x, ϵ, χ)
svdcutfun_maxerror(ϵ; χ::Int = Inf64) = x -> _maxerror(x, ϵ, χ)

function _maxcumerror(xs, ϵ, χ)
    cs = reverse(cumsum(reverse(xs)))
    index = findfirst(x -> x < ϵ, cs)
    index == nothing && return min(length(xs), χ)
    return min(index-1, χ)
end

function _maxerror(xs, ϵ, χ)
    index = findfirst(x -> x < ϵ, xs)
    index == nothing && return min(length(xs), χ)
    return min(index-1, χ)
end
