#AbstractTensor
to3tuple(x) = ntuple(i -> x[i] isa Int ? (x[i],0,0) : x[i], length(x))
"""
	fuselegs(A, indexes)

Fuse the indices in `A` as given by `indexes` where `indexes` is a tuple
containing indices either alone or grouped in tuples - the latter will be fused.
Returns a tuple of a tensor and the object necessary to undo the fusion.

# Examples
```julia-repl
julia> a = DTensor(collect(reshape(1:8,2,2,2))
DTensor{Int64,3}[1 3; 2 4]
[5 7; 6 8]
julia> fuselegs(a, ((1,2),3))
(DTensor{Int64,2}[1 5; 2 6; 3 7; 4 8], ((2, 2),))
```
"""
function fuselegs end

"""
	fuselegs!(AF,A, indexes)

In-place version of `fuselegs` where `AF` is a tensor of the correct type
and size to hold the fused version of `A`.
See [`fuselegs`](@ref)
"""
function fuselegs! end

"""
	splitlegs(A, indexes, rs...)

Split the indices in `A` as given by `indexes` and `rs`.
`indexes` is a tuple of integers and 3-tuple where each 3-tuple (i,j,k)
specifies that index `i` in `A` is split according to `rs[j]`, index `k` therein.
Returns tensor with fused legs.

# Examples
```julia-repl
julia> a = DTensor(collect(reshape(1:8,2,2,2))
DTensor{Int64,3}[1 3; 2 4]
[5 7; 6 8]
julia> af, rs = fuselegs(a, ((1,2),3))
(DTensor{Int64,2}[1 5; 2 6; 3 7; 4 8], ((2, 2),))
julia> splitlegs(af, ((1,1,1),(1,1,2),2), rs...)
DTensor{Int64,3}[1 3; 2 4]
[5 7; 6 8]
```
"""
function splitlegs end

"""
	splitlegs!(AS,A, indexes, rs...)

In-place version of `splitlegs` where `AS` is a tensor of the correct type
and size to hold the split version of `A`.
See [`splitlegs`](@ref)
"""
function splitlegs! end


#DTensor
function fuselegs(A::DTensor{T}, indexes) where T
    inds = map(x -> x isa Int ? (x,) : x, indexes)
    s = size(A)
    dims = map(i -> prod(TT.getindices(s,i)), inds)
	AF = DTensor(Array{T}(undef,dims...))
	fuselegs!(AF, A, indexes)
end

function fuselegs!(AF::DTensor{T}, A::DTensor{T}, indexes) where T
    perm = TT.vcat(indexes...)
	dims = size(AF)
	s = size(A)
	tmp = reshape(AF.array, size(A))
	permutedims!(tmp, A.array, perm)
	copyto!(AF.array, reshape(tmp, dims))
    rs = tuple((TT.getindices(s,i) for i in indexes if i isa Tuple)...)
    return  (AF, rs)
end

function splitlegs(A::DTensor{T}, indexes::NTuple{N,Union{NTuple{3,Int},Int}}, rs...) where {T,N}
    s = size(A)
    inds = to3tuple(indexes)
	perm = TT.sortperm(inds)
	inds = TT.getindices(inds, perm)
    dims = ntuple(N) do i
        inds[i][2] == 0 ? s[inds[i][1]] : rs[inds[i][2]][inds[i][3]]
    end
	AS = DTensor(Array{T}(undef, dims...))
	splitlegs!(AS, A, indexes, rs...)
end

function splitlegs!(AS::DTensor{T}, A::DTensor{T}, indexes::NTuple{N,Union{NTuple{3,Int},Int}}, rs...) where {T,N}
    inds = to3tuple(indexes)
    dims  = size(AS)
	iperm  = TT.invperm(TT.sortperm(inds))
	tmp = reshape(A.array, dims...)
	permutedims!(AS.array, tmp, iperm)
	return AS
end

#DASTensor
struct Reshaper{M,CHARGE,CHARGES}
    sectors ::Vector{DASSector{M,CHARGE}}
    chs     ::Vector{CHARGE}
    ochs    ::NTuple{M,CHARGES}
    oios    ::InOut{M}
    ods     ::NTuple{M,Vector{Int}}
    nchs    ::CHARGES
    nios    ::InOut{1}
    ranges  ::Vector{UnitRange{Int}}
    dims    ::Vector{Int}
end

function fusesector(r::Reshaper, s::DASSector)
    @unpack sectors, chs, ranges, dims, nchs, ochs = r
    i = findfirst(==(s), sectors)
    i == nothing && error()
    ch = chs[i]
    d  = dims[chargeindex(ch,nchs)]
    return (ch, ranges[i], d)
end

struct SplitChargeIt{M,CHARGE,CHARGES}
    r::Reshaper{M,CHARGE,CHARGES}
    ch::CHARGE
end

function splitchargeit(r::Reshaper{M,CHARGE,CHARGES}, ch::CHARGE) where
        {M,CHARGES,CHARGE}
    return SplitChargeIt{M,CHARGE,CHARGES}(r,ch)
end

function Base.iterate(spchit::SplitChargeIt, i = 1)
    @unpack r, ch = spchit
    @unpack chs, sectors, ranges = r
    i = findnext(==(ch), chs, i)
    i == nothing && return nothing
    return ((sectors[i], ranges[i]), i+1)
end

Base.eltype(::SplitChargeIt{M,CHARGE,CHARGES}) where {M,CHARGE,CHARGES} =
    Tuple{DASSector{M,CHARGE}, UnitRange{Int}}
Base.length(spit::SplitChargeIt) = count(==(spit.ch),spit.r.chs)


function chargedim(M, ods, sec, ochs)
    d = 1
    @inbounds for j in 1:M
        d *= ods[j][chargeindex(sec[j], ochs[j])]
    end
    return d
end

function fusiondict(A::DASTensor, indexes::NTuple, ld::InOut{1})
    ochs, oios, ods = charges(A, indexes), in_out(A,indexes), sizes(A,indexes)
    return fusiondict(ochs, oios, ods, indexes, ld)
end

function fusiondict(ochs::NTuple{M, CHARGES},
                    oios::InOut{M},
                    ods::NTuple{M},
                    indexes::NTuple{M,Int},
                    ld::InOut{1}) where {M,CHARGES}
    nchs = ⊕(ochs...)
    l, ln = prod(length.(ochs)), length(nchs)
    CHARGE = eltype(ochs[1])
    dims    = zeros(Int, ln)
    sectors = Vector{DASSector{M,CHARGE}}(undef, l)
    chs     = Vector{CHARGE}(undef, l)
    ranges  = Vector{UnitRange{Int}}(undef, l)
    for (i, sec) in enumerate(allsectors(ochs))
        nch = ld ⊗ charge(oios ⊗ sec)
        d   = chargedim(M, ods, sec, ochs)
        nd = dims[chargeindex(nch, nchs)]
        ranges[i]  = (1:d) .+ nd
        sectors[i], chs[i] = sec, nch
        dims[chargeindex(nch, nchs)] += d
    end
    return Reshaper{M,CHARGE,CHARGES}(sectors,
        chs,
        ochs, oios, ods,
        nchs, ld,
        ranges, dims)
end

function fusiondicts(A::DASTensor{T,N,SYM,CHARGES,SIZES,CHARGE},
                    indexes, lds::InOut{M}) where
                        {T,N,SYM,CHARGES,SIZES,CHARGE,M}
    inds = map(x -> x isa Int ? (x,) : x, indexes)
    rs   = map((i, io) -> fusiondict(A, i, io), inds, ntuple(i -> lds[i], Val(M)))
end

fuselegs(A::DASTensor, indexes::Tuple)  =
    fuselegs(A, indexes, InOut(ntuple(i -> 1,length(indexes))...))

function fuselegs(A::DASTensor{T,N,SYM}, indexes, lds::InOut{M}) where {M,T,N,SYM}
    rs = fusiondicts(A, indexes, lds)
    _totuple(x) = x isa Tuple ?  x : (x,)
    inds = map(_totuple, indexes)
    perm = TT.vcat(indexes...)
    isperm(perm) || throw(ArgumentError("not valid specification of indexes"))
    newchs = ntuple(i -> rs[i].nchs, M)
    newds  = ntuple(i -> copy(rs[i].dims), M)
    newios = reduce(vcat, ntuple(i -> rs[i].nios, M))

    AF = DASTensor{T,M}(SYM, newchs, newds, newios)
    initwithzero!(AF, charge(A))
    fuselegs!(AF, A, indexes, lds, rs)
end

function fuselegs!(AF, A, indexes, lds, rs)
    _totuple(x) = x isa Tuple ?  x : (x,)
    inds = map(_totuple, indexes)
    perm = TT.vcat(indexes...)
    for (sector, degeneracy) in tensor(A)
        tuples = map((i, r) -> fusesector(r, sector[i]), inds, rs)
        nsector = DASSector(getindex.(tuples,1)...)
        nranges = getindex.(tuples, 2)
        s    = size(degeneracy)
        dims = map(i -> prod(s[vec(collect(i))]), inds)
        AF[nsector][nranges...] .= reshape(permutedims(degeneracy, perm), dims...)
    end
    return (AF, rs)
end

### Splitting
function findith(pred, list, i)
    j = 0
    for d in 1:i
        j = findnext(pred, list, j+1)::Int
    end
    return j
end

selectfrombool(a,b, M = count(b)) = ntuple(i -> a[findith(identity, b, i)], M)
uniqueind(b::NTuple{M}) where M = ntuple(i -> findfirst(x -> x[1] == b[i][1], b) == i, Val(M))
_issplitcount(x, M) = ntuple(i -> count(y -> first(y) == i, x), M)
_issplitconv(x::NTuple{M}) where M = ntuple(i -> x[i][2] == 0 ? 0 : x[i][1] , M)
issplit(x,M) = map(x -> !iszero(x), _issplitcount(_issplitconv(x), M))
reduceinds(inds) = selectfrombool(inds, uniqueind(inds))

function splitpick(tinds, old, new)
    map(tinds) do i
        iszero(i[2]) ? old[i[1]] : new[i[2]][i[3]]
    end
end

unpacksectorrange(x) = vcat(getindex.(x,1)...), getindex.(x,2)

function getsectorrange(sector, rinds, issplits, rs)
    its = map(rinds, issplits) do i, s
        s ? splitchargeit(rs[i[2]], sector[i[1]]) : Ref((DASSector(sector[i[1]]), :))
    end
    return (unpacksectorrange(x) for x in Iterators.product(its...))
end

function splitlegs(A::DASTensor{T,N,SYM},
                    inds::NTuple{M,Union{Int,NTuple{3,Int}}},
                    rs::Reshaper...) where {M,T,N,SYM}
    tinds    = to3tuple(inds)
    ncharges = splitpick(tinds, charges(A), getproperty.(rs,:ochs))
    ndims    = deepcopy(splitpick(tinds, sizes(A),   getproperty.(rs,:ods )))
    nios     = reduce(vcat, splitpick(tinds, in_out(A),  getproperty.(rs,:oios)))

    AS = DASTensor{T,M}(SYM, ncharges, ndims, nios)
    initwithzero!(AS, charge(A))
    return splitlegs!(AS, A, inds, rs...)
end

function splitlegs!(AS::DASTensor{T,M},
                    A::DASTensor{T,N},
                    inds::NTuple{M,Union{Int,NTuple{3,Int}}},
                    rs::Reshaper...) where {M,T,N}
    tinds    = to3tuple(inds)
    issplits = issplit(tinds, Val(N))
    srinds   = TT.sort(reduceinds(tinds))
    perm     = TT.sortperm(tinds)
    iperm    = TT.invperm(perm)
    cchs, cdims = charges(A), sizes(A)
    for (sector, degen) in tensor(A)
        for (nsec, nrange) in getsectorrange(sector, srinds, issplits, rs)
            nsec = permute(nsec,iperm)
            dims = TT.permute(size(AS[nsec]), perm)
            reshaped = reshape(view(degen,nrange...), dims...)
            permdimed = TO.tensorcopy(collect(reshaped), ntuple(identity,M), iperm)
            AS[nsec] = permdimed
        end
    end
    return AS
end
