#AbstractTensor
TO.checked_similar_from_indices(C, T::Type, p1, p2, A::AbstractTensor, CA::Symbol = :N) =
    TO.checked_similar_from_indices(C, T, (p1..., p2...), A, CA)

TO.checked_similar_from_indices(C, T::Type, poA, poB, p1, p2, A::AbstractTensor,
        B::AbstractTensor, CA::Symbol = :N, CB::Symbol = :N) =
    TO.checked_similar_from_indices(C, T, poA, poB, (p1..., p2...), A, B, CA, CB)

TO.add!(α, A::AbstractTensor, conjA, β, C::AbstractTensor, p1, p2)=
    TO.add!(α, A, conjA, β, C, (p1..., p2...))

TO.trace!(α, A::AbstractTensor, CA, β, C::AbstractTensor, indleft, indright, cind1, cind2) =
    TO.trace!(α, A, CA, β, C, (indleft..., indright...), cind1, cind2)

TO.contract!(α, A::AbstractTensor, CA, B::AbstractTensor, CB, β, C::AbstractTensor, oindA, cindA, oindB,
        cindB, indleft, indright, syms = nothing) =
    TO.contract!(α, A, CA, B, CB, β, C, oindA, cindA, oindB, cindB, (indleft..., indright...), syms)

#DTensor
TO.scalar(A::DTensor)  = TO.scalar(A.array)

function TO.checked_similar_from_indices(C, ::Type{T}, ind, A::DTensor, CA::Symbol) where T
    sz = map(n->size(A, n), ind)
    if C !== nothing && C isa DTensor && sz == size(C) && T == eltype(C)
        CT = DTensor{T,length(sz)}
        return C::CT
    else
        return similar(A, T, sz)
    end
end

function TO.checked_similar_from_indices(C, ::Type{T}, poA, poB, ind, A::DTensor,
        B::DTensor, CA::Symbol, CB::Symbol) where {T,N}
    oszA = map(n->size(A,n), poA)
    oszB = map(n->size(B,n), poB)
    sz = let osz = (oszA..., oszB...)
        map(n->osz[n], ind)
    end
    if C !== nothing && C isa DTensor && sz == size(C) && T == eltype(C)
        CT = DTensor{T,length(sz)}
        return C::CT
    else
        return similar(A, T, sz)
    end
end


TO.add!(α, A::DTensor, conjA, β, C::DTensor, indCinA) =
    DTensor(TO.add!(α, A.array, conjA, β, C.array, indCinA))

TO.trace!(α, A::DTensor, CA, β, C::DTensor, indCinA, cindA1, cindA2) =
    DTensor(TO.trace!(α, A.array, CA, β, C.array, indCinA, cindA1, cindA2))


TO.contract!(α, A::DTensor, CA::Symbol, B::DTensor, CB::Symbol, β, C::DTensor,
        oindA, cindA, oindB, cindB, indCinoAB, syms::Union{Nothing, NTuple{3,Symbol}} = nothing) =
        DTensor(TO.contract!(α, A.array, CA, B.array, CB, β, C.array,
        oindA, cindA, oindB, cindB, indCinoAB, syms))


#DASTensor
TO.scalar(a::DASTensor) = TO.scalar(first(values(a)))

function TO.checked_similar_from_indices(C, ::Type{T}, ind,
        A::DASTensor{TA,NA,SYM,CHS,SS,CH}, CA::Symbol) where {T,TA,NA,SYM,CHS,SS,CH}
    if CA == :N
        dims = sizes(A,ind)
        ios  = in_out(A,ind)
    else
        dims = reverse.(sizes(A,ind))
        ios  = inv(in_out(A,ind))
    end
    chs = charges(A,ind)

    if C !== nothing && C isa DASTensor && dims == sizes(C) && ios == in_out(C) &&
            chs == charges(C) && T == eltype(C) && charge(C) == charge(A)
        CT = DASTensor{T,length(ind),CHS,SS,CH}
        return C::CT
    else
        return DASTensor{T,length(ind)}(SYM, chs, deepcopy(dims), ios, charge(A))
    end
end

function TO.checked_similar_from_indices(C, ::Type{T}, poA, poB, ind,
        A::DASTensor{TA,NA,SYM,CHS,SS,CH},
        B::DASTensor{TB,NB,SYM,CHS,SS,CH},
        CA::Symbol, CB::Symbol) where {T,TA,NA,TB,NB,SYM,CHS,SS,CH}
    chs = TT.getindices(TT.vcat(charges(A,poA), charges(B,poB)), ind)
    dims = TT.getindices(TT.vcat(
                TT.getindices(CA == :N ? sizes(A) : reverse.(sizes(A)), poA),
                TT.getindices(CB == :N ? sizes(B) : reverse.(sizes(B)), poB)),
                ind)
    dimsA =  CA == :N ? sizes(A,poA) : map(reverse, sizes(A,poA))
    dimsB =  CB == :N ? sizes(B,poB) : map(reverse, sizes(B,poB))
    dims = TT.getindices(TT.vcat(dimsA,dimsB), ind)
    ios = vcat(ifelse(CA == :N, in_out(A,poA), inv(in_out(A,poA))),
                ifelse(CB == :N, in_out(B,poB), inv(in_out(B,poB))))[ind]
    if C !== nothing && C isa DASTensor && dims == sizes(C) && ios == in_out(C) &&
            chs == charges(C) && T == eltype(C) && charge(C) == charge(A) ⊕ charge(B)
        CT = DASTensor{T,length(ind),SYM,CHS,SS,CH}
        return C::CT
    else
        return DASTensor{T,length(ind)}(SYM,chs, deepcopy(dims), ios, charge(A) ⊕ charge(B))
    end
end

function _errorsadd(A::DASTensor{T,N,SYM}, C, indCinA::NTuple{M}) where {T,N,SYM,M}
    mask = in_out(A,indCinA).v .== in_out(C).v
    for (m, iA, iC) in zip(mask, indCinA, 1:M)
        sizes(A, iA) == ifelse(m,sizes(C, iC), reverse(sizes(C,iC))) ||
            throw(DimensionMismatch())
        charges(A, iA) == ifelse(m, charges(C, iC), inv(charges(C, iC))) ||
            throw(ArgumentError("charges don't agree"))
    end
    modio = InOut(ntuple(i -> ifelse(mask[i],1,-1),Val(N))...)
    return ⊗(modio)
end

function TO.add!(α::Number, A::DASTensor{T,N}, CA, β::Number, C::DASTensor{S,N},
            indCinA) where {T,S,N}
    maskfun = _errorsadd(A, C, indCinA)
    for (sector, degeneracy) in tensor(A)
        permsector = maskfun(sector)[indCinA]
        if haskey(tensor(C), permsector)
            TO.add!(α, degeneracy, CA, β, C[permsector], indCinA)
        else
            C[permsector] = TO.checked_similar_from_indices(nothing, T, indCinA, degeneracy, CA)
            TO.add!(α, degeneracy, CA, 0, C[permsector], indCinA)
        end
    end
    return C
end

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

    maskC = in_out(A, indCinA).v .== in_out(C).v
    for (m, iA, iC) in zip(maskC, indCinA, 1:NC) #trace in A
        sizes(A, iA) == ifelse(m, sizes(C, iC), reverse(sizes(C,iC))) ||
            throw(DimensionMismatch())
        charges(A, iA) == ifelse(m, charges(C, iC), inv(charges(C,iC))) ||
            throw(ArgumentError("charges don't agree"))
    end
    maskAio = InOut(ntuple(i -> ifelse(maskA[i],1,-1), Val(M))...)
    maskCio = InOut(ntuple(i -> ifelse(maskC[i],1,-1), Val(NC))...)
    return (⊗(maskAio), ⊗(maskCio))
end

function TO.trace!(α, A::DASTensor{T,N}, CA, β, C::DASTensor{S,M},
                    indCinA, cindA1, cindA2) where {T,N,S,M}
    #conditions
    maskAfun, maskCfun = _errorstrace(A, cindA1, cindA2, C, indCinA)

    sectors = filter(x -> isequal(maskAfun(x[cindA1]), x[cindA2]), keys(A))
    cinds = TT.vcat(cindA1, cindA2)

    passedset = Set{DASSector{M,chargetype(symmetry(A))}}()
    for sector in sectors
        secC = maskCfun(sector[indCinA])
        if haskey(C, secC)
            if !in(secC, passedset)
                TO.trace!(α, A[sector], CA, β, C[secC], indCinA, cindA1, cindA2)
                push!(passedset, secC)
            else
                TO.trace!(α, A[sector], CA, 1, C[secC], indCinA, cindA1, cindA2)
            end
        else
            C[secC] = TO.checked_similar_from_indices(nothing, T, indCinA, A[sector], CA)
            TO.trace!(α, A[sector], CA, 0, C[secC], indCinA, cindA1, cindA2)
            push!(passedset, secC)
        end
    end
    return C
end

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

function TO.contract!(α, A::DASTensor{TA,NA,SYM}, CA,
                      B::DASTensor{TB,NB,SYM}, CB, β,
                      C::DASTensor{TC,NC,SYM}, oindA, cindA, oindB, cindB,
                      indCinoAB, syms=nothing) where
                      {TA,NA,TB,NB,TC,NC,M,SYM}
    #conditions
    maskBfun, maskABfun = _errorscontract(A, (oindA, cindA), B, (oindB, cindB), C, indCinoAB)

    oinAB = TT.vcat(oindA, oindB .+ NA)
    indCinAB = TT.getindices(oinAB, indCinoAB)
    secsA = groupby(x -> x[cindA], keys(A))
    secsB = groupby(x -> maskBfun(x[cindB]), keys(B))
    # collect sectors that contract with each other
    secsAB = intersect(keys(secsA), keys(secsB))
    passedset = Set{DASSector{NC,chargetype(SYM)}}()
    for sector in secsAB
        for secA in secsA[sector], secB in secsB[sector]
            secC = maskABfun(permute(vcat(secA[oindA], secB[oindB]), indCinoAB))
            if haskey(C, secC)
                if in(secC, passedset) #firstpass
                    TO.contract!(α, A[secA], CA, B[secB], CB, 1, C[secC],
                              oindA, cindA, oindB, cindB, indCinoAB,syms)
                else
                    push!(passedset, secC)
                    TO.contract!(α, A[secA], CA, B[secB], CB, β, C[secC],
                              oindA, cindA, oindB, cindB, indCinoAB,syms)
                 end
             else
                C[secC] = TO.checked_similar_from_indices(nothing, TC,
                                    oindA, oindB, indCinoAB, (),
                                    A[secA], B[secB], :N, :N)
                TO.contract!(α, A[secA], CA, B[secB], CB, 0, C[secC],
                          oindA, cindA, oindB, cindB, indCinoAB,syms)
                push!(passedset, secC)
             end
         end
     end
     return C
 end
