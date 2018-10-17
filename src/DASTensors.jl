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
    print(io, "Tensors ", eltype(A.tensor))
end

#= Rand =#
_get_sectors(charges, in_out, filterfun) = (sector for sector in product(charges...)
                                                   if filterfun(sector, in_out))

function _get_degeneracy(charges, sector::NTuple{N,Int}, sizes, ::Type{T}) where {N,T}
    dsizes = map((si,se,ch) -> si[findfirst(==(se), ch)],sizes,sector,charges)
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
diag(A::DASTensor{T,2}) where T = reduce(vcat,(diag(degen) for degen in values(tensor(a))))

#getters
@inline charges(A::DASTensor) = A.charges
@inline sizes(A::DASTensor)   = A.sizes
@inline in_out(A::DASTensor)  = A.in_out
@inline tensor(A::DASTensor)  = A.tensor

@inline charges(A::DASTensor,i::Int) = A.charges[i]
@inline charges(A::DASTensor,i) = TT.getindices(A.charges,i)
@inline sizes(A::DASTensor,i::Int)   = A.sizes[i]
@inline sizes(A::DASTensor,i)   = TT.getindices(A.sizes,i)
@inline chargesize(A::DASTensor, ch, i::Int) = A.sizes[i][findfirst(==(ch), A.charges[i])]
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

function _errorsadd(A, C, perm::NTuple{N}) where N
    _invert(a::UnitRange) = (-a.stop):(-a.start)
    mask = in_out(A) .== in_out(C, perm)
    for (m, iA, iC) in zip(mask, 1:N, perm)
        if m
            charges(A, iA) == charges(C, iC) || throw(ArgumentError("charges don't agree"))
            sizes(A, iA) == sizes(C, iC) || throw(ArgumentError("sizes don't agree"))
        else
            charges(A, iA) == _invert(charges(C, iC)) || throw(ArgumentError("charges don't agree"))
            sizes(A, iA) == reverse(sizes(C, iC)) || throw(ArgumentError("sizes don't agree"))
        end
    end
    return tuple([ifelse(b,1,-1) for b in mask]...)
end

function add!(α::Number, A::DASTensor{T,N}, conjA::Type{Val{CA}},
     β::Number, C::DASTensor{S,N}, indCinA) where {T,S,N,CA}
    perm = TT.sortperm(indCinA)
    mask = _errorsadd(A, C, perm)

    for (sector, degeneracy) in tensor(A)
        permsector = TT.permute(mask .* sector, perm)
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

function _errorstrace(A::DASTensor{T,N}, cindA1::NTuple{M1,Int}, cindA2::NTuple{M2,Int}, C, indCinA) where {M1,M2,T,N}
    _invert(a::UnitRange) = (-a.stop):(-a.start)
    M1 == M2 || throw(ArgumentError("indices to contract don't pair up") )
    maskA = in_out(A, cindA1) .== -1 .* in_out(A, cindA2)
    for (m, iA1, iA2) in zip(maskA, cindA1, cindA2) #trace in A
        if m
            charges(A, iA1) == charges(A, iA2) || throw(ArgumentError("charges don't agree"))
            sizes(A, iA1) == sizes(A, iA2) || throw(ArgumentError("sizes don't agree"))
        else
            charges(A, iA1) == _invert(charges(A, iA2)) || throw(ArgumentError("charges don't agree"))
            sizes(A, iA1) == reverse(sizes(A, iA2)) || throw(ArgumentError("sizes don't agree"))
        end
    end
    indsA = TT.sort(indCinA)
    perm = TT.sortperm(indCinA)
    maskC = in_out(A, indsA) .== in_out(C, perm)
    for (m, iA, iC) in zip(maskC, indsA, perm) #trace in A
        if m
            charges(A, iA) == charges(C, iC) || throw(ArgumentError("charges don't agree"))
            sizes(A, iA) == sizes(C, iC) || throw(ArgumentError("sizes don't agree"))
        else
            charges(A, iA) == _invert(charges(C, iC)) || throw(ArgumentError("charges don't agree"))
            sizes(A, iA) == reverse(sizes(C, iC)) || throw(ArgumentError("sizes don't agree"))
        end
    end
    return (tuple([ifelse(b,1,-1) for b in maskA]...), tuple([ifelse(b,1,-1) for b in maskC]...))
end

function trace!(α, A::DASTensor{T,N}, ::Type{Val{CA}}, β, C::DASTensor{S,M},
                indCinA, cindA1, cindA2) where {T,N,S,M,CA}
    #conditions
    maskA, maskC = _errorstrace(A, cindA1, cindA2, C, indCinA)

    perm = TT.sortperm(indCinA)
    sectors = filter(x -> isequal(maskA .* TT.getindices(x, cindA1), TT.getindices(x, cindA2)),
                        collect(keys(tensor(A))))
    newsectors = map(x -> maskC .* TT.permute(TT.deleteat(x, TT.vcat(cindA1, cindA2)),perm), sectors)
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

function _errorscontract(A, (oindA, cindA), B, (oindB, cindB), C, indCinoAB)
    length(cindA) == length(cindB) || throw(ArgumentError("indices to contract don't pair up"))
    length(oindA) + length(oindB) == length(indCinoAB) || throw(ArgumentError("indices to contract don't pair up"))
    _invert(a::UnitRange) = (-a.stop):(-a.start)
    maskB = in_out(A,cindA) .== -1 .* in_out(B,cindB)
    for (m, iA, iB) in zip(maskB, cindA, cindB)
        if m
            charges(A,iA) == charges(B,iB) || throw(ArgumentError("charges don't agree"))
            sizes(A,iA) == sizes(B,iB) || throw(ArgumentError("sizes don't agree"))
        else
            charges(A,iA) == _invert(charges(B,iB)) || throw(ArgumentError("charges don't agree"))
            sizes(A,iA) == reverse(sizes(B,iB)) || throw(ArgumentError("sizes don't agree"))
        end
    end
    ioAB  = TT.vcat(in_out(A), in_out(B))
    chsAB = TT.vcat(charges(A), charges(B))
    dsAB  = TT.vcat(sizes(A), sizes(B))
    maskAB = TT.getindices(ioAB,indCinoAB) .== in_out(C)
    for (m, iAB, iC) in zip(maskB, indCinoAB, 1:length(indCinoAB))
        if m
            chsAB[iAB] == charges(C, iC)|| throw(ArgumentError("charges don't agree"))
            dsAB[iAB] == sizes(C,iC) || throw(ArgumentError("sizes don't agree"))
        else
            chsAB[iAB] == _invert(charges(C, iC)) || throw(ArgumentError("charges don't agree"))
            dsAB[iAB] == reverse(sizes(C,iC)) || throw(ArgumentError("sizes don't agree"))
        end
    end
    return (tuple([ifelse(b,1,-1) for b in maskB]...), tuple([ifelse(b,1,-1) for b in maskAB]...))
end

function contract!(α, A::DASTensor{TA,NA}, ::Type{Val{CA}},
                      B::DASTensor{TB,NB}, ::Type{Val{CB}}, β,
                      C::DASTensor{TC,NC}, oindA, cindA, oindB, cindB,
                      indCinoAB, ::Type{Val{M}}=Val{:native}) where
                      {TA,NA,TB,NB,TC,NC,CA,CB,M}
    #conditions
    maskB, maskAB = _errorscontract(A, (oindA, cindA), B, (oindB, cindB), C, indCinoAB)

    oinAB = TT.vcat(oindA, .+(oindB, NA))
    indCinAB = TT.getindices(oinAB, indCinoAB)
    secsA = groupby(x -> TT.getindices(x, cindA), keys(tensor(A)))
    secsB = groupby(x -> maskB .* TT.getindices(x, cindB), keys(tensor(B)))
    # collect sectors that contract with each other
    secsAB = intersect(keys(secsA), keys(secsB))
    passedset = Set()
    for sector in secsAB
        for secA in secsA[sector], secB in secsB[sector]
            newsector = maskAB .* _getnewsector(secA, secB, oindA, oindB, indCinoAB)
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
function fusiondict(A::T, indexes::NTuple{N,Int}, ld::Int) where {T<:DASTensor,N}
    #=
    fdict: oldcharges → newcharge & indices
    ddict: newcharges → dimension
    sdict: newcharges → [oldcharges & indices]
    =#
    sdict = Dict{Int,Vector{Tuple{NTuple{N,Int},UnitRange}}}()
    fdict = Dict{NTuple{N,Int},Tuple{Int,UnitRange}}()
    ddict = Dict{Int,Int}()
    ochs = charges(A, indexes)
    oios = in_out(A, indexes)
    ods  = sizes(A, indexes)

    for chs in Iterators.product(ochs...)
        nch = fusecharges(T, oios, ld)(chs)
        if !haskey(sdict, nch)
            sdict[nch] = []
            ddict[nch] = 0
        end
        d = prod((ods[i][findfirst(==(ch), ochs[i])] for (i,ch) in enumerate(chs)))
        push!(sdict[nch], (chs, (1:d) .+ ddict[nch]))
        fdict[chs] = (nch, sdict[nch][end][2])
        ddict[nch] += d
    end
    return (fdict, ddict, sdict)
end

function fusiondict(A::T, i::Int, ld::Int) where {T<:DASTensor}
    sdict = Dict{Int,Vector{Tuple{Int,UnitRange}}}()
    fdict = Dict{Int,Tuple{Int,UnitRange}}()
    ddict = Dict{Int,Int}()
    for ch in charges(A,i)
        nch = fusecharges(T, (in_out(A,i),), ld)((ch,))
        d = chargesize(A,ch,i)
        sdict[nch] = [(ch,1:d)]
        fdict[ch] = (nch, 1:d)
        ddict[nch] = d
    end
    return (fdict, ddict, sdict)
end

function fusiondicts(A, indexes, lds)
    dicts = [fusiondict(A,i,ld) for (i,ld) in zip(indexes, lds)]
    return collect(zip(dicts...))
end

function fusefields(A::T, indexes, lds, ddicts) where {T<:DASTensor}
    ochs = [charges(A,i) for i in indexes]
    oios = [in_out(A,i) for i in indexes]
    newchs = tuple(map((ch,io,ld) -> fusecharge(T,ch,io,ld), ochs, oios, lds)...)
    newds  = tuple(map((dict,chs) -> [dict[ch] for ch in chs], ddicts, newchs)...)
    return (newchs, newds, lds)
end


function fuselegs(A::T, indexes, lds::NTuple{M,Int}) where {T<:DASTensor{S,N}, M} where {S,N}
    _pick(sector, i::Int) = sector[i]
    _pick(sector, i::NTuple) = TT.getindices(sector, i)

    fdicts, ddicts, sdicts = fusiondicts(A, indexes, lds)
    newfields = fusefields(A, indexes, lds, ddicts)
    perm = TT.vcat(indexes...)
    isperm(perm) || throw(ArgumentError("not valid specification of indexes"))

    ntensor = Dict{NTuple{M,Int},Array{S,M}}()
    for (sector, degeneracy) in tensor(A)
        tuples = map((i, fdict) -> fdict[_pick(sector, i)], indexes, fdicts)
        nsector, nranges = collect(zip(tuples...))
        if !haskey(ntensor, nsector)
            ntensor[nsector] = zeros(S, map(getindex, ddicts, nsector)...)
        end
        s = size(degeneracy)
        dims = map(i -> prod(s[vec(collect(i))]), indexes)
        ntensor[nsector][nranges...] = reshape(permutedims(degeneracy, perm), dims...)
    end
    _totuple(x) = ifelse(x isa Tuple, x, (x,))
    ochs = [_totuple(charges(A, i)) for i in indexes]
    oios = [_totuple(in_out(A, i))  for i in indexes]
    ods  = [_totuple(sizes(A, i))   for i in indexes]
    inverter = (ochs, oios, ods, sdicts)
    return (constructnew(T, newfields, ntensor), inverter)
end


#== Splitting ==#
function splitlegs(A::T, inds::NTuple{M,Union{Int,NTuple{3,Int}}}, inverter) where {M,T<:DASTensor{S}} where S
    nchs, nios, ndims, sdicts = inverter
    cchs, cdims = charges(A), sizes(A)
    ntensor = Dict{NTuple{M,Int},Array{S,M}}()
    rinds, iperm, perm = _reduceinds(inds)
    for (sector, degen) in tensor(A)
        nsecs, nranges = _splitsec(sector,rinds,sdicts)
        for (nsec, nrange) in zip(nsecs, nranges)
            dims = _splitsdims((sector, cchs, cdims), (nsec, nchs,ndims), inds, perm)
            nsec = TT.getindices(nsec, iperm)
            ntensor[nsec] = copy(permutedims(reshape(view(degen, nrange...),
                                        dims...), iperm) )
        end
    end
    _pick(i::Int, current, old)  = current[i]
    _pick((l,m,n), current, old) = old[m][n]
    newchs = tuple([_pick(i, charges(A), nchs) for i in inds]...)
    newios = tuple([_pick(i, in_out(A), nios)  for i in inds]...)
    newds  = tuple([_pick(i, sizes(A), ndims)    for i in inds]...)
    return constructnew(T, (newchs,newds,newios), ntensor)
end

function _splitsec(sector, inds, sdicts)
    _pick(i::Int) = ((sector[i], :),)
    _pick((l,m,n)) = sdicts[m][sector[l]]
    nseccharges = [_pick(i) for i in inds]
    nsecs = []
    nranges = []
    for ns in Iterators.product(nseccharges...)
        push!(nsecs, TT.vcat([s[1] for s in ns]...))
        push!(nranges, [s[2] for s in ns])
    end
    return (nsecs, nranges)
end

function _splitsdims((sector, cchs, cdims), (nsec, nchs, ndims), inds, perm)
    _pick(i::Int, k)  = cdims[i][findfirst(==(sector[i]), cchs[i])]
    _pick((l,m,n), k) = ndims[m][n][findfirst(==(nsec[k]), nchs[m][n])]
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
    tU = Dict{NTuple{N,Int},Array{Q,N}}()
    tS = Dict{NTuple{N,Int},Array{Q,N}}()
    tV = Dict{NTuple{N,Int},Array{Q,N}}()

    lch = connectingcharge(T, charges(A), in_out(A), charge(A))
    ld =  sizes(A,2)[[c in lch for c in charges(A,2)]]

    chU = (charges(A,1), lch)
    dU  = deepcopy.((sizes(A,1), ld))
    ioU = in_out(A)

    chS = (lch, lch)
    dS  =  deepcopy.((ld,ld))
    ioS = (-in_out(A,2), in_out(A,2))

    chV = (lch, charges(A,2))
    dV  =  deepcopy.((ld,sizes(A,2)))
    ioV = (-in_out(A,2), in_out(A,2))

    for ((in, out), degen) in tensor(A)
        tU[(in, out)], tS[(out, out)], tV[(out, out)], cutoff =
            _tensorsvd(degen, svdcutfunction = svdcutfunction, helper = true)
        dU[2][findfirst(==(out), chU[2])] = cutoff
        dS[1][findfirst(==(out), chS[1])] = cutoff
        dS[2][findfirst(==(out), chS[2])] = cutoff
        dV[1][findfirst(==(out), chV[1])] = cutoff
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


function connectingcharge(::Type{<:U1Tensor}, (ch1,ch2), (io1,io2), charge)
    _invertcharge(a::UnitRange) = (-a.stop):(-a.start)
    # ch1→[A]→ch2
    ch1 = ifelse(io1 ==  1, _invertcharge(ch1), ch1) .- charge
    ch2 = ifelse(io2 == -1, _invertcharge(ch2), ch2)
    ch3 = intersect(ch1, ch2)
    io2 == -1 && return _invertcharge(ch3)
    return ch3
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
