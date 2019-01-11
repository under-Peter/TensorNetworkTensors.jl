#AbstractTensor
function LinearAlgebra.norm(v::AbstractTensor)
    n = sqrt(dot(v,v))
    isreal(n) || error("norm should be real")
    return real(n)
end

function LinearAlgebra.rmul!(v::AbstractTensor,a)
    apply!(v, x -> x .= a .* x)
end

function LinearAlgebra.axpy!(a,v::AbstractTensor{<:Any,N},w::AbstractTensor{<:Any,N}) where {N}
    axpby!(a,v,1,w)
end

function LinearAlgebra.axpby!(a,v::AbstractTensor{<:Any,N},b,w::AbstractTensor{<:Any,N}) where {N}
    TO.tensoradd!(a,v,1:N,b,w,1:N)
end

#DTensor
Base.fill!(a::DTensor, w) = (fill!(a.array, w); a)
function LinearAlgebra.mul!(w::DTensor,v::DTensor,a)
    DTensor(mul!(w.array, v.array, a))
end

function LinearAlgebra.dot(v::T,w::T) where {T<:DTensor{S,N}} where {S,N}
    dot(v.array,w.array)
end

#DASTensor
function Base.fill!(a::DASTensor, w)
    initwith!(a, (T,d...) -> fill(w,d...))
    a
end

function LinearAlgebra.mul!(w::DASTensor,v::DASTensor,a)
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

function LinearAlgebra.dot(v::T,w::T) where {T<:DASTensor{S,N}} where {S,N}
    ks = intersect(keys(v),keys(w))
    result = zero(S)
    for k in ks
        result += dot(v[k],w[k])
    end
    return result
end
