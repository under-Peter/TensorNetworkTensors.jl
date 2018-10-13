#= Struct =#
mutable struct ZNTensor{T,N,M} <: DASTensor{T,N}
    charges::NTuple{N,UnitRange{Int}}
    sizes::NTuple{N,Vector{Int}}
    in_out::NTuple{N,Int}
    tensor::Dict{NTuple{N,Int}, Array{T,N}}
    function ZNTensor{T,N,M}(charges, sizes, in_out, tensor) where {T,N,M}
        all(in([-1,1]), in_out) || throw(ArgumentError("in/outs needs to be ∈ [-1,1]"))
        all(map(==, length.(charges), length.(sizes))) ||
            throw(ArgumentError("charges need to be same length as sizes"))
        all(map(isequal(0:(M-1)),charges)) ||
            throw(ArgumentError("charges need to be 0:$(M-1) for Z$M"))
        new{T,N,M}(charges, sizes, in_out, tensor)
    end
end


#= Print =#
printname(::Type{ZNTensor{T,N,M}}) where {T,N,M} = "Z$M-symmetric Tensor"


#= Constructors =#
ZNTensor{T,N,M}(dims, in_out) where {T,N,M} =
    ZNTensor{T,N,M}(ntuple(x->0:(M-1),N),dims, in_out, Dict())

ZNTensor{M}(dims::NTuple{N}, in_out, T = ComplexF64) where {N,M} =
    ZNTensor{T,N,M}(dims, in_out)

ZNTensor(dims::NTuple{N}, in_out, T = ComplexF64) where N =
    ZNTensor{T,N,length(dims[1])}(dims, in_out)

ZNTensor(dims, in_out, tensors::Dict{NTuple{N,Int}, Array{T,N}}) where {T,N} =
    ZNTensor{T,N,length(dims[1])}(ntuple(x->0:length(dims[1])-1,N),dims, in_out, tensors)

ZNTensor{M}(T::Type = ComplexF64) where M = ZNTensor{T,0,M}((),(),(),Dict())

ZNTensor{T,N,M}(charges, dims, in_out) where {T,N,M} =
    ZNTensor{T,N,M}(charges, dims, in_out, Dict())

#= Helper Functions =#
scalar(A::ZNTensor{T,0,M}) where {T,M} = first(first(values(A.tensor)))

filterfun(::Type{<:ZNTensor{T,N,M}}) where {T,N,M}  = (x, y) -> iszero(mod(sum(map(*,x,y)),M))

isinvariant(A::ZNTensor{T,N,M}) where {T,N,M} =
    all(iszero ∘ (x->mod(x,M)) ∘ sum, map(*,in_out(A),k) for k in keys(tensor(A)))

charge(A::ZNTensor{T,N,M}) where {T,N,M} = -mod(sum(map(*,in_out(A),first(keys(tensor(A))))),M)

fusecharge(::Type{<:ZNTensor{T,N,M}}, oldcharges, in_out, out) where {T,N,M} = 0:(M-1)

fusecharges(::Type{<:ZNTensor{T,N,M}}, in_out, out) where {T,M,N} = x -> mod(out * sum(map(*,x,in_out)),M)

Base.rand(::Type{ZNTensor{T,N,M}}, dims, in_out) where {T,N,M} =
    Base.rand(ZNTensor{T,N,M},ntuple(x -> 0:(M-1),N), dims, in_out)


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
    in_outsC = TT.getindices(TT.vcat(
                TT.getindices(CA == :N ? in_out(A) : -1 .* in_out(A), poA),
                TT.getindices(CB == :N ? in_out(B) : -1 .* in_out(B), poB)),
                p12)
    return ZNTensor{T,length(p12),M}(chargesC, deepcopy(sizesC), in_outsC,Dict())
end

Base.similar(A::ZNTensor{T,N,M}, ::Type{S}) where {T,N,S,M} =
    ZNTensor{S,N,M}(charges(A), deepcopy(sizes(A)), in_out(A))

Base.similar(A::ZNTensor{T,N,M}) where {T,N,M} =
    ZNTensor{T,N,M}(charges(A), deepcopy(sizes(A)), in_out(A))

# #= RESHAPING W/O CONTRACTION =#
# function fusiondict(A::U1Tensor, indexes::Tuple, direction)
#     oldcharges = TT.getindices(charges(A), indexes)
#     oldios     = TT.getindices(in_out(A),  indexes)
#     olddims    = TT.getindices(sizes(A),   indexes)
#     return fusiondict(oldcharges, oldios, olddims, direction)
# end
#
# function fusiondict(As::NTuple{N,U1Tensor}, indexes::NTuple{N,Int}, direction) where N
#     oldcharges = tuple([charges(A,i) for (A,i) in zip(As,indexes)]...)
#     oldios     = tuple([in_out(A,i)  for (A,i) in zip(As,indexes)]...)
#     olddims    = tuple([sizes(A,i)   for (A,i) in zip(As,indexes)]...)
#     return fusiondict(oldcharges, oldios, olddims, direction)
# end
#
# function fusiondict(oldcharges::NTuple{N,T}, oldios::NTuple{N,Int},
#      olddims::NTuple{N,Vector{Int}}, dir::Int) where {N,T}
#
#     sdict = Dict{Int,Vector{Tuple{NTuple{N,Int},T}}}()
#     fdict = Dict{NTuple{N,Int},Tuple{Int,T}}()
#     ddict = Dict{Int,Int}()
#     for charges::NTuple{N,Int} in IterTools.product(oldcharges...)
#         d = prod([olddims[i][findfirst(x -> x == ch, oldcharges[i])]
#             for (i, ch) in zip(1:N, charges)])
#         ncharge = dir * fusecharges(U1Tensor, oldios)(charges)
#         if haskey(sdict, ncharge)
#             push!(sdict[ncharge], (charges, (1:d) + ddict[ncharge]))
#             ddict[ncharge] += d
#         else
#             sdict[ncharge] = [(charges,1:d)]
#             ddict[ncharge] = d
#         end
#         fdict[charges] = (ncharge, sdict[ncharge][end][2])
#     end
#     return (sdict, fdict, ddict)
# end
#
#
# fuselegs(A::U1Tensor, indexes, legdirs) = fuselegs(A, map(totuple, indexes), legdirs)
# splitlegs(A::U1Tensor, indexes, inverter)    = splitlegs(A, map(totuple, indexes), inverter)
#
# fusiondicts(A::U1Tensor, indexes, legdirs) =
#     fusiondicts(tupleapply((i,A) -> A, indexes, A), indexes, legdirs)
#
# function fusiondicts(As::NTuple{N,Any}, indexes::NTuple{N,Any}, legdirs) where N
#     indexes = map(totuple,indexes)
#     dicts = map((A, i, dir) -> Tensors.fusiondict(A, i, dir),
#                 As, indexes, legdirs)
#     sdicts, fdicts, ddicts = collect(zip(dicts...))
#     oldcharges  = tupleapply((i,A) -> charges(A,i),indexes, As)
#     oldios      = tupleapply((i,A) -> in_out(A,i),indexes,  As)
#     olddims     = tupleapply((i,A) -> sizes(A,i),indexes,   As)
#     inverter = (oldcharges, oldios, olddims,
#                 sdicts)
#     return (fdicts, ddicts, inverter)
# end
#
# function fusefields(As::NTuple, indexes, legdirs, ddicts)
#     oldchs = tupleapply((i,A) -> charges(A,i), indexes, As)
#     oldios = tupleapply((i,A) -> in_out(A,i), indexes, As)
#     fusechargeparams = map((a, b, c) -> totuple.((a, b, c)), oldchs, oldios, legdirs)
#     newcharges = (map(x -> Tensors.fusecharge(U1Tensor{1}, x...), fusechargeparams)...)
#     newsizes   = tuple(map((dict, chs) -> [dict[ch] for ch in chs],
#                             ddicts, newcharges)...)
#     newin_outs = legdirs
#     return (newcharges, newsizes, newin_outs)
# end
#
#
# fuselegs(A::U1Tensor{T,N}, indexes::NTuple{M,Tuple}, legdirs) where
#     {T,N,M} = fuselegs(A,indexes,legdirs,
#                     fusiondicts(A, indexes, legdirs)...)
#
# fuselegs(A::U1Tensor{T,N}, indexes::NTuple{M,Tuple}, legdirs,
#     fdicts, ddicts, inverter) where {T,N,M} =
#         fuselegs(A, indexes, legdirs, fdicts, ddicts, inverter,
#             fusefields(A, indexes, legdirs, ddicts),)
#
# function fuselegs(A::U1Tensor{T,N}, indexes::NTuple{M,Tuple}, legdirs,
#     fdicts, ddicts, inverter, fusefields) where {T,N,M}
#     newtensor = Dict()
#     for (sector, degeneracy) in tensor(A)
#         tuples = map((i, dict) -> dict[TT.getindices(sector, i)],
#                         indexes, fdicts)
#         newsector, newranges = collect(zip(tuples...))
#         if !haskey(newtensor, newsector)
#             newtensor[newsector] = zeros(T, map(getindex, ddicts, newsector)...)
#         end
#         newtensor[newsector][newranges...] = _fuselegs(degeneracy, indexes)
#     end
#     return (U1Tensor{T,M}( fusefields..., newtensor),
#             inverter)
# end
#
# function _fuselegs(A::Array{T,N}, indexes) where {T,N}
#     s = size(A)
#     perm = TT.vcat(indexes...)
#     dims = map(i -> prod(s[vec(collect(i))]), indexes)
#     return  reshape(permutedims(A, perm), dims...)
# end
#
# function fusefields(A, indexes, legdirs, ddicts)
#     oldchs = tupleapply((i,A) -> charges(A,i), indexes, A)
#     oldios = tupleapply((i,A) -> in_out(A,i),  indexes, A)
#     fusechargeparams = map((a, b, c) -> totuple.((a, b, c)),
#                             oldchs, oldios, legdirs)
#     newcharges = tuple(map(x -> Tensors.fusecharge(U1Tensor{1},x...),
#                             fusechargeparams)...)
#     newsizes   = tuple(map((dict, chs) -> [dict[ch] for ch in chs],
#                             ddicts, newcharges)...)
#     newin_outs = legdirs
#     return (newcharges, newsizes, newin_outs)
# end
#
#
# #indexes has elements (i_a, (i_s, j)) where i_a refers to the index in A that is to be split
# #according to i_s in the inverter and taken index j of said splitting
# function splitlegs(A::U1Tensor, indexes::NTuple{M,Tuple}, inverter) where {M}
# 	sortedindexes = TT.sort(indexes)
# 	iperm = TT.invperm(TT.sortperm(indexes))
# 	B = sortedsplitlegs(A, sortedindexes, inverter)
# 	C = TensorOperations.similar_from_indices(eltype(B), iperm, B)
# 	return TensorOperations.add!(1, B, Val{:N}, 0, C, iperm)
# end
#
# function sortedsplitlegs(A::U1Tensor{T,N}, sortedindexes::NTuple{M,Tuple}, inverter) where {T,N,M}
#     oldcharges, oldios, olddims, sdicts = inverter
#     currentcharges = charges(A)
#     currentdims    = sizes(A)
#     reducedindexes = _getreducedindexes(sortedindexes)
#     newtensor = Dict()
#
#     for (sector, degeneracy) in tensor(A)
#         nsranges = _newsectorsranges(sector, reducedindexes, sdicts)
#         for (newsector, range) in nsranges
#             dims = _getsplitdims(sector, newsector, oldcharges,
#                                 currentcharges, olddims, currentdims,
#                                 sortedindexes)
#             newtensor[newsector] = copy(reshape(view(degeneracy, range...), dims...))
#         end
#     end
#     newcharges = _splitpick(sortedindexes, oldcharges, charges(A))
#     newin_outs = _splitpick(sortedindexes, oldios,     in_out(A))
#     newdims    = _splitpick(sortedindexes, olddims,    sizes(A))
#     return U1Tensor{T,M}(newcharges, newdims, newin_outs, newtensor)
# end
#
# function _newsectorsranges(sector, reducedindexes::NTuple{N,Any}, sdicts) where {N}
#     newsectors = []
#     for i in reducedindexes
#         if length(i) == 1
#             push!(newsectors, ((TT.getindices(sector, i),:),))
#         else
#             (l, (m, n)) = i
#             push!(newsectors, sdicts[m][sector[l]])
#         end
#     end
#     tmp = IterTools.product(newsectors...)
#     newsectorranges = []
#     for ns in tmp
#         tmps = TT.vcat([s[1] for s in ns]...)
#         tmpr = tuple([s[2] for s in ns]...)
#         push!(newsectorranges, (tmps, tmpr))
#     end
#     return newsectorranges
# end
#
# function _getreducedindexes(indexes)
#     reducedindexes = []
#     useddicts = []
#     for i in indexes
#         if length(i) == 1
#             push!(reducedindexes, i)
#         else
#             (l, (m, n)) = i
#             if m in useddicts
#                 continue
#             else
#                 push!(useddicts, m)
#                 push!(reducedindexes, i)
#             end
#         end
#     end
#     return tuple(reducedindexes...)
# end
#
# function _getsplitdims(sector, newsector, oldcharges, currentcharges, olddims,
# 	 currentdims, indexes)
#     dims = zeros(Int, length(indexes))
#     for (k, i) in enumerate(indexes)
#         if length(i) == 1
#             i = i[1]
#             dims[k] = currentdims[i][findfirst(x -> x == sector[i], currentcharges[i])]
#         else
#             (l, (m, n)) = i
#             dims[k] = olddims[m][n][findfirst(x -> x == newsector[k], oldcharges[m][n])]
#         end
#     end
#     return dims
# end
#
# function _splitpick(indexes, oldthings, currenthings)
#     newthings = []
#     for i in indexes
#         if length(i) == 1
#             push!(newthings, currenthings[i[1]])
#         else
#             (l, (m, n)) = i
#             push!(newthings, oldthings[m][n])
#         end
#     end
#     return (newthings...)
# end
#
# invertcharge(a::UnitRange) = (-a.stop):(-a.start)
#
# function connectingcharge(chs, ios, charge)
#     ch1 = ios[1] ==  1 ? invertcharge(chs[1]) : chs[1]
#     ch1 -= charge
#     ch2 = ios[2] == -1 ? invertcharge(chs[2]) : chs[2]
#     # ch1→[A]→ch2
#     ch3 = intersect(ch1, ch2)
#     ios[2] == -1 && return invertcharge(ch3)
#     return ch3
# end
#
#
# function tensorsvd(A::U1Tensor{T,N}; svdcutfunction = svdcutfun_default) where {T,N}
#     N == 2 || throw(ArgumentError("SVD only works on rank 2 tensors"))
#     tU, tS, tV = (Dict(), Dict(), Dict())
#
#     lch = connectingcharge(charges(A), in_out(A), charge(A))
#     ld =  sizes(A,2)[[any(c .== lch) for c in charges(A,2)]]
#
#     chU = (charges(A,1), lch)
#     dU  = deepcopy.((sizes(A,1), ld))
#     ioU = in_out(A)
#
#     chS = (lch, lch)
#     dS  =  deepcopy.((ld,ld))
#     ioS = (-in_out(A,2), in_out(A,2))
#
#     chV = (lch, charges(A,2))
#     dV  =  deepcopy.((ld,sizes(A,2)))
#     ioV = (-in_out(A,2), in_out(A,2))
#
#     for ((in, out), degen) in tensor(A)
#         tU[(in, out)], tS[(out, out)], tV[(out, out)], cutoff =
#             _tensorsvd(degen, svdcutfunction = svdcutfunction, helper = true)
#         dU[2][findfirst(x -> x == out,  chU[2])] = cutoff
#         dS[1][findfirst(x -> x == out,  chS[1])] = cutoff
#         dS[2][findfirst(x -> x == out, chS[2])] = cutoff
#         dV[1][findfirst(x -> x == out, chV[1])] = cutoff
#     end
#     U, S, V = ( U1Tensor{T,2}(chU, dU, ioU, tU),
#                 U1Tensor{T,2}(chS, dS, ioS, tS),
#                 U1Tensor{T,2}(chV, dV, ioV, tV))
#     return (U, S, V)
# end
#
# function _tensorsvd(A::AbstractArray; svdcutfunction = svdcutfun_default, helper = false)
#     fact   = svdfact(A)
#     cutoff = svdcutfunction(fact[:S])
#     U =       fact[:U][:, 1:cutoff]
#     S = diagm(fact[:S][1:cutoff])
#     V =       fact[:Vt][1:cutoff, :]
#     helper && return (U, S, V, cutoff)
#     return (U, S, V)
# end
#
#
#
#
# svdcutfun_default = x -> length(x)
# svdcutfun_discardzero = x -> length(filter(!iszero, x))
# svdcutfun_maxχ(χ) = x -> min(length(x), χ)
# svdcutfun_maxcumerror(ϵ; χ = 100000) = x -> _maxcumerror(x, ϵ, χ)
# svdcutfun_maxerror(ϵ; χ = 1000000) = x -> _maxerror(x, ϵ, χ)
#
# function _maxcumerror(xs, ϵ, χ)
#     cs = cumsum(xs[end:-1:1])[end:-1:1]
#     index = findfirst(x -> x < ϵ, cs)
#     iszero(index) && return min(length(xs), χ)
#     return min(index-1, χ)
# end
#
# function _maxerror(xs, ϵ, χ)
#     index = findfirst(x -> x < ϵ, xs)
#     iszero(index) && return min(length(xs), χ)
#     return min(index-1, χ)
# end
