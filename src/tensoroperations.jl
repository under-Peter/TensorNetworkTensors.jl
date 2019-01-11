#DTensor
TO.scalar(A::DTensor)  = TO.scalar(A.array)
TO.numind(A::DTensor{T,N}) where {T,N} = N
TO.similar_from_indices(T::Type, indices, A::DTensor, ::Type{<:Val}=Val{:N}) =
    DTensor(TO.similar_from_indices(T, indices, A.array, Val{:N}))

TO.similar_from_indices(T::Type, poA, poB, p1, p2, A::DTensor, B::DTensor, CA, CB) =
    DTensor(TO.similar_from_indices(T, poA, poB, p1, p2, A.array, B.array, CA, CB))

TO.similar_from_indices(T::Type, index, A::DTensor, B::DTensor,
    ::Type{Val{CA}} = Val{:N}, ::Type{Val{CB}} = Val{:N}) where
    {CA,CB} = DTensor(TO.similar_from_indices(T, index, A.array, B.array, Val{CA}, Val{CB}))

TO.similar_from_indices(T::Type, p1::Tuple, p2, A::DTensor, CA::Type{<:Val}) =
 DTensor(TO.similar_from_indices(T, (p1...,p2...), A.array, CA))

TO.add!(α, A::DTensor{T,N}, CA::Type{<:Val}, β, C::DTensor{S,M},
    p1, p2) where {T,N,S,M} = TO.add!(α, A, CA, β, C, (p1..., p2...))

TO.add!(α, A::DTensor{T,N}, ::Type{Val{CA}}, β, C::DTensor{S,M}, indCinA) where
    {CA,T,N,S,M} = DTensor(TO.add!(α, A.array, Val{CA}, β, C.array, indCinA))

TO.add!(α, A::DTensor, CA::Type{<:Val}, β, C::DTensor, p1::Tuple, p2::Tuple) =
    TO.add!(α, A, CA, β, C, (p1...,p2...))

TO.trace!(α, A::DTensor, CA::Type{<:Val}, β, C::DTensor, p1, p2, cindA1, cindA2) =
    TO.trace!(α, A, CA, β, C, (p1..., p2...), cindA1, cindA2)
TO.trace!(α, A::DTensor, ::Type{Val{CA}}, β, C::DTensor, indCinA, cindA1, cindA2) where
    {CA} = DTensor(TO.trace!(α, A.array, Val{CA}, β, C.array, indCinA, cindA1, cindA2))

TO.contract!(α, A::DTensor, ::Type{Val{CA}}, B::DTensor, ::Type{Val{CB}}, β,
    C::DTensor, oindA, cindA, oindB, cindB, indCinoAB,
     ::Type{Val{M}} = Val{:BLAS}) where {CA,CB,M} =
    DTensor(TO.contract!(α, A.array, Val{CA}, B.array, Val{CB},
                         β, C.array, oindA, cindA, oindB, cindB,
                         indCinoAB, Val{M}))

TO.contract!(α, A::DTensor, CA::Type{<:Val}, B::DTensor, CB::Type{<:Val}, β,
                C::DTensor, oindA, cindA, oindB, cindB, p1, p2,
                method::Type{<:Val} = Val{:BLAS}) =
    TO.contract!(α, A, CA, B, CB, β, C, oindA, cindA, oindB, cindB, (p1..., p2...), method)

#DASTensor
TO.scalar(a::DASTensor) = TO.scalar(first(values(a)))
TO.numind(::DASTensor{<:Any,N}) where N = N

TO.similar_from_indices(T::Type, p1::Tuple, p2::Tuple, A::DASTensor, CA::Type{<:Val}) =
    TO.similar_from_indices(T, (p1...,p2...), A, CA)

function TO.similar_from_indices(S::Type, index::NTuple{M,Int},
        A::DASTensor{T,N,SYM}, ::Type{Val{CA}} = Val{:N}) where {M,T,N,SYM, CA}
        dims = CA == :N ? sizes(A,index)  : map(reverse,sizes(A,index))
        ios  = CA == :N ? in_out(A,index) : inv(in_out(A,index))
    return DASTensor{S,M}(SYM, charges(A,index), deepcopy(dims), ios)
end

function TO.similar_from_indices(T::Type, index::NTuple{N,Int},
            A::DASTensor{TA,NA,SYM}, B::DASTensor{TB,NB,SYM},
            ::Type{Val{CA}} = Val{:N}, ::Type{Val{CB}} = Val{:N}) where
                {N,TA,NA,TB,NB,SYM,CA,CB}
    chs = TT.getindices(TT.vcat(charges(A), charges(B)), index)
    dimsA =  CA == :N ? sizes(A) : map(reverse,sizes(A))
    dimsB =  CB == :N ? sizes(B) : map(reverse,sizes(B))
    dims  =  TT.getindices(TT.vcat(dimsA,dimsB), index)
    io  =  vcat(ifelse(CA == :N, in_out(A), inv(in_out(A))),
                ifelse(CB == :N, in_out(B), inv(in_out(B))))[index]
    return DASTensor{T,N}(SYM, chs, deepcopy(dims), io)
 end

function TO.similar_from_indices(T::Type, poA, poB, p1, p2,
        A::DASTensor{TA,NA,SYM}, B::DASTensor{TB,NB,SYM},
        ::Type{Val{CA}} = Val{:N},
        ::Type{Val{CB}} = Val{:N}) where {TA,NA,TB,NB,SYM,CA,CB}
    p12 = (p1...,p2...)
    chs = TT.getindices(TT.vcat(charges(A,poA), charges(B,poB)), p12)
    dims = TT.getindices(TT.vcat(
                TT.getindices(CA == :N ? sizes(A) : reverse.(sizes(A)), poA),
                TT.getindices(CB == :N ? sizes(B) : reverse.(sizes(B)), poB)),
                p12)
    dimsA =  CA == :N ? sizes(A,poA) : map(reverse, sizes(A,poA))
    dimsB =  CB == :N ? sizes(B,poB) : map(reverse, sizes(B,poB))
    dims = TT.getindices(TT.vcat(dimsA,dimsB), p12)
    ios = vcat(ifelse(CA == :N, in_out(A,poA), inv(in_out(A,poA))),
                ifelse(CB == :N, in_out(B,poB), inv(in_out(B,poB))))[p12]
    return DASTensor{T,length(p12)}(SYM,chs, deepcopy(dims), ios)
end

function _errorsadd(A::DASTensor{T,N,SYM}, C, perm::NTuple{M}) where {T,N,SYM,M}
    mask = in_out(A).v .== in_out(C, perm).v
    for (m, iA, iC) in zip(mask, 1:M, perm)
        sizes(A, iA) == ifelse(m,sizes(C, iC), reverse(sizes(C,iC))) ||
            throw(DimensionMismatch())
        charges(A, iA) == ifelse(m, charges(C, iC), inv(charges(C, iC))) ||
            throw(ArgumentError("charges don't agree"))
    end
    modio = InOut(ntuple(i -> ifelse(mask[i],1,-1),Val(N))...)
    return ⊗(modio)
end

function TO.add!(α::Number, A::DASTensor{T,N}, ::Type{Val{CA}},
     β::Number, C::DASTensor{S,N}, indCinA) where {T,S,N,CA}
    perm = TT.sortperm(indCinA)
    maskfun = _errorsadd(A, C, perm)
    for (sector, degeneracy) in tensor(A)
        permsector = permute(maskfun(sector), perm)
        if haskey(tensor(C), permsector)
            TO.add!(α, degeneracy, Val{CA}, β, C[permsector], indCinA)
        else
            C[permsector] = TO.similar_from_indices(T, indCinA, degeneracy)
            TO.add!(α, degeneracy, Val{CA}, 0, C[permsector], indCinA)
        end
    end
    return C
end

TO.add!(α, A::DASTensor, CA, β, C::DASTensor, p1, p2) =
    TO.add!(α, A, CA, β, C, (p1...,p2...))


function _errorstrace(A::DASTensor{T,N,<:Any},
                         cindA1::NTuple{M,Int},
                         cindA2::NTuple{M,Int},
                         C::DASTensor{TC,NC},
                         indCinA::NTuple{NC}) where {M,T,N,TC,NC}
    maskA = in_out(A, cindA1).v .== (inv(in_out(A, cindA2))).v
    for (m, iA1, iA2) in zip(maskA, cindA1, cindA2) #trace in A
        sizes(A, iA1) == ifelse(m, sizes(A, iA2), reverse(sizes(A,iA2))) ||
            throw(DimensionMismatch())
        charges(A, iA1) == ifelse(m, charges(A, iA2), inv(charges(A,iA2))) ||
            throw(ArgumentError("charges don't agree"))
    end
    indsA = TT.sort(indCinA)
    perm = TT.sortperm(indCinA)

    maskC = in_out(A, indsA).v .== in_out(C, perm).v
    for (m, iA, iC) in zip(maskC, indsA, perm) #trace in A
        sizes(A, iA) == ifelse(m, sizes(C, iC), reverse(sizes(C,iC))) ||
            throw(DimensionMismatch())
        charges(A, iA) == ifelse(m, charges(C, iC), inv(charges(C,iC))) ||
            throw(ArgumentError("charges don't agree"))
    end
    maskAio = InOut(ntuple(i -> ifelse(maskA[i],1,-1), Val(M))...)
    maskCio = InOut(ntuple(i -> ifelse(maskC[i],1,-1), Val(NC))...)
    return (⊗(maskAio), ⊗(maskCio))
end

function TO.trace!(α, A::DASTensor{T,N}, ::Type{Val{CA}},
                                β, C::DASTensor{S,M},
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
        if haskey(C, newsector)
            if !in(newsector, passedset)
                TO.trace!(α, A[sector], Val{CA}, β, C[newsector], indCinA, cindA1, cindA2)
                push!(passedset, newsector)
            else
                TO.trace!(α, A[sector], Val{CA}, 1, C[newsector], indCinA, cindA1, cindA2)
            end
        else
            C[newsector] = TO.similar_from_indices(T, indCinA, A[sector])
            TO.trace!(α, A[sector], Val{CA}, 0, C[newsector], indCinA, cindA1, cindA2)
            push!(passedset, newsector)
        end
    end
    return C
end

TO.trace!(α, A::DASTensor, CA, β, C::DASTensor, p1, p2, cindA1, cindA2) =
    TO.trace!(α, A, CA, β, C, (p1..., p2...), cindA1, cindA2)

function _errorscontract(A::DASTensor{TA,NA,<:Any},
                        (oindA, cindA)::Tuple{NTuple{NoA,Int}, NTuple{CoA,Int}},
                        B::DASTensor{TB,NB,<:Any},
                        (oindB, cindB)::Tuple{NTuple{NoB,Int}, NTuple{CoB,Int}},
                        C::DASTensor{TC,NC,<:Any},
                        indCinoAB::NTuple{NoC,Int}) where
                            {NA, NB, NC, TA, TB, TC, NoA, CoA, NoB, CoB, NoC}
    CoA == CoB || throw(ArgumentError("indices to contract don't pair up"))
    NoA + NoB == NoC || throw(ArgumentError("indices to contract don't pair up"))

    maskB = in_out(A, cindA).v .== inv(in_out(B, cindB)).v
    for (iA, iB, m)  in zip(cindA, cindB, maskB)
        charges(A,iA) == ifelse(m, charges(B,iB), inv(charges(B,iB)))||
            throw(ArgumentError("charges don't agree"))
        sizes(A,iA) == ifelse(m, sizes(B,iB), reverse(sizes(B,iB))) ||
            throw(DimensionMismatch())
    end
    ioAB  = vcat(in_out(A, oindA), in_out(B, oindB))
    chsAB = TT.vcat(charges(A, oindA), charges(B, oindB))
    dsAB  = TT.vcat(sizes(A, oindA), sizes(B, oindB))

    maskAB = ioAB[indCinoAB].v .== in_out(C).v
    for (iC, iAB, m) in zip(1:NoC, indCinoAB, maskAB)
        chsAB[iAB] == ifelse(m, charges(C, iC), inv(charges(C,iC)))||
            throw(ArgumentError("charges don't agree"))
        dsAB[iAB] == ifelse(m, sizes(C,iC), reverse(sizes(C,iC))) ||
             throw(DimensionMismatch())
    end

    maskBio = InOut(tuple(ifelse.(maskB,1,-1)...)...)
    maskABio = InOut(tuple(ifelse.(maskAB,1,-1)...)...)
    return (⊗(maskBio), ⊗(maskABio))
end

function TO.contract!(α, A::DASTensor{TA,NA,SYM}, ::Type{Val{CA}},
                      B::DASTensor{TB,NB,SYM}, ::Type{Val{CB}}, β,
                      C::DASTensor{TC,NC,SYM}, oindA, cindA, oindB, cindB,
                      indCinoAB, ::Type{Val{M}}=Val{:native}) where
                      {TA,NA,TB,NB,TC,NC,CA,CB,M,SYM}
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
            if haskey(C, newsector)
                if !in(newsector, passedset) #firstpass
                    push!(passedset, newsector)
                    TO.contract!(α, A[secA], Val{CA}, B[secB], Val{CB},
                              β, C[newsector],
                              oindA, cindA, oindB, cindB, indCinoAB, Val{M})
                else
                    TO.contract!(α, A[secA], Val{CA}, B[secB], Val{CB},
                              1, C[newsector],
                              oindA, cindA, oindB, cindB, indCinoAB, Val{M})
                 end
             else
                C[newsector] = TO.similar_from_indices(TC, oindA, oindB, indCinoAB, (),
                                    A[secA], B[secB], Val{:N}, Val{:N})

                TO.contract!(α, A[secA], Val{CA}, B[secB], Val{CB},
                          0, C[newsector],
                          oindA, cindA, oindB, cindB, indCinoAB, Val{M})
                push!(passedset, newsector)
             end
         end
     end
     return C
 end

TO.contract!(α, A::DASTensor, CA, B::DASTensor, CB, β, C::DASTensor,
    oindA, cindA, oindB, cindB, p1, p2, method::Type{<:Val} = Val{:BLAS}) =
    TO.contract!(α, A, CA, B, CB, β, C, oindA, cindA, oindB, cindB, (p1..., p2...), method)
