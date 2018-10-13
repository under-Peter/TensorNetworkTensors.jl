#= RESHAPING W/O CONTRACTION =#
#fdict: oldcharges → newcharge & indices
#ddict: newcharges → dimension
#sdict: newcharges → [oldcharges & indices]

function fusiondict(A::T, indexes::NTuple{N,Int}, ld::Int) where {T<:DASTensor,N}
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
        d = prod((ods[i][findfirst(isequal(ch), ochs[i])] for (i,ch) in enumerate(chs)))
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

function constructnew(::Type{U1Tensor{T,N}}, newfields, newtensor::Dict{NTuple{M,Int},Array{T,M}}) where {N,M,T}
    return U1Tensor{T,M}(newfields...,newtensor)
end

function constructnew(::Type{ZNTensor{T,L,M}}, newfields, newtensor::Dict{NTuple{N,Int},Array{T,N}}) where {N,M,L,T}
    return ZNTensor{T,N,M}(newfields...,newtensor)
end


#indexes has elements (i_a, i_s, j) where i_a refers to the index in A that is to be split
#according to i_s in the inverter and taken index j of said splitting
function splitlegs(A::T, inds::NTuple{M,Union{Int,NTuple{3,Int}}}, inverter) where {M,T<:DASTensor{S}} where S
    nchs, nios, ndims, sdicts = inverter
    cchs, cdims = charges(A), sizes(A)
    ntensor = Dict{NTuple{M,Int},Array{S,M}}()
    rinds, iperm, perm = _reduceinds(inds)
    for (sec, degen) in tensor(A)
        #get new resultings secs and their ranges
        nsecs, nranges = _splitsec(sec,rinds,sdicts)
        for (nsec, nrange) in zip(nsecs, nranges)
            #inds should be wrong
            dims = _splitsdims((sec, cchs, cdims), (nsec, nchs,ndims), inds, perm)
            nsec = TT.getindices(nsec, iperm)
            ntensor[nsec] = copy(   permutedims(
                                        reshape(
                                            view(degen, nrange...),
                                        dims...),
                                    iperm)
                                )
        end
    end
    _pick(i::Int, current, old)  = current[i]
    _pick((l,m,n), current, old) = old[m][n]
    newchs = tuple([_pick(i, charges(A), nchs) for i in inds]...)
    newios = tuple([_pick(i, in_out(A), nios)  for i in inds]...)
    newds  = tuple([_pick(i, sizes(A), ndims)    for i in inds]...)
    return constructnew(T, (newchs,newds,newios), ntensor)
end

# sec = (0,1)
# inds = (1,(2,1,1))
function _splitsec(sec, inds, sdicts)
    _pick(i::Int) = ((sec[i], :),)
    _pick((l,m,n)) = sdicts[m][sec[l]]
    nseccharges = [_pick(i) for i in inds]
    nsecs = []
    nranges = []
    for ns in Iterators.product(nseccharges...)
        push!(nsecs, TT.vcat([s[1] for s in ns]...))
        push!(nranges, [s[2] for s in ns])
    end
    return (nsecs, nranges)
end

function _splitsdims((sec, cchs, cdims), (nsec, nchs, ndims), inds, perm)
    _pick(i::Int, k)  = cdims[i][findfirst(isequal(sec[i]), cchs[i])]
    _pick((l,m,n), k) = ndims[m][n][findfirst(isequal(nsec[k]), nchs[m][n])]
    [_pick(i,k) for (k,i) in enumerate(TT.getindices(inds,perm))]
end

function _reduceinds(inds)
    rinds = tuple(sort(unique(first, inds), by=first)...)
    tinds = map(i -> ifelse(i isa Int, (i,), i), inds)
    perm  = TT.sortperm(tinds)
    iperm = TT.invperm(perm)
    return (rinds, iperm, perm)
end
