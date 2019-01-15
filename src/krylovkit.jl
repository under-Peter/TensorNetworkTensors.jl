#AbstractTensor
function LA.norm(v::AbstractTensor)
    n = sqrt(LA.dot(v,v))
    isreal(n) || error("norm should be real")
    return real(n)
end

function LA.rmul!(v::AbstractTensor,a)
    apply!(v, x -> x .= a .* x)
end


function LA.axpby!(a,v::AbstractTensor{<:Any,N},b,w::AbstractTensor{<:Any,N}) where {N}
    TO.tensoradd!(a,v,1:N,b,w,1:N)
end

function LA.axpy!(a,v::AbstractTensor{<:Any,N},w::AbstractTensor{<:Any,N}) where {N}
    LA.axpby!(a,v,1,w)
end

#DTensor
Base.fill!(a::DTensor, w) = (fill!(a.array, w); a)
function LA.mul!(w::DTensor,v::DTensor,a)
    DTensor(LA.mul!(w.array, v.array, a))
end

function LA.dot(v::T,w::T) where {T<:DTensor{S,N}} where {S,N}
    LA.dot(v.array,w.array)
end

#DASTensor
function Base.fill!(a::DASTensor, w)
    initwith!(a, (T,d...) -> fill(w,d...))
    a
end

function LA.mul!(w::DASTensor,v::DASTensor,a)
    kw, kv = keys(w), keys(v)
    for k in setdiff(kw,kv)
        delete!(kw.tensor, k)
    end
    for k in setdiff(kv,kw)
        w[k] = a .* v[k]
    end
    for k in intersect(kw,kv)
        copyto!(w[k], a .* v[k])
    end
    return w
end

function LA.dot(v::T,w::T) where {T<:DASTensor{S,N}} where {S,N}
    ks = intersect(keys(v),keys(w))
    result = zero(S)
    for k in ks
        result += LA.dot(v[k],w[k])
    end
    return result
end
