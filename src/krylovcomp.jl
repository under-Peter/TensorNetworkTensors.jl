using LinearAlgebra

Base.fill!(a::AbstractTensor, w) = apply!(a, x -> x .* 0 .+ w )

function LinearAlgebra.mul!(w::DASTensor,v::DASTensor,a)
    kw, kv = keys(w), keys(v)
    for k in setdiff(kw,kv)
        delete!(kw.tensor, k)
    end
    for k in setdiff(kv,kw)
        w[k] = a .* v[d]
    end
    for k in intersect(kw,kv)
        copyto!(w[k], a .* d)
    end
    return w
end

function LinearAlgebra.mul!(w::DTensor,v::DTensor,a)
    DTensor(mul!(w.array, v.array, a))
end

function LinearAlgebra.rmul!(v::AbstractTensor,a)
    apply!(v, x -> a .* x)
end

function LinearAlgebra.axpy!(a,v::AbstractTensor{<:Any,N},w::AbstractTensor{<:Any,N}) where {N}
    axpby!(a,v,1,w)
end

function LinearAlgebra.axpby!(a,v::AbstractTensor{<:Any,N},b,w::AbstractTensor{<:Any,N}) where {N}
    tensoradd!(a,v,1:N,b,w,1:N)
end

function LinearAlgebra.dot(v::T,w::T) where {T<:AbstractTensor{S,N}} where {S,N}
    scalar(tensorcontract(v,1:N,w',1:N))
end

function LinearAlgebra.norm(v::AbstractTensor)
    sqrt(dot(v,v))
end

function Base.:*(a::AbstractTensor, b::Number)
    apply!(a, x -> b * x)
end
