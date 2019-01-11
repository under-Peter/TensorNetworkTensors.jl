#AbstractTensor
Base.:(*)(α::Number, A::AbstractTensor) = apply(A, x -> @. x *= α)
Base.:(*)(A::AbstractTensor, α::Number) = α   * A
Base.:(/)(A::AbstractTensor, α::Number) = 1/α * A
Base.conj!(A::AbstractTensor) = apply!(A, conj!)
Base.conj(A::AbstractTensor)  = apply(A, conj!)

#DTensor
apply!(A::DTensor, fun!) = (fun!(A.array); return A)
apply(A::DTensor, fun!) = apply!(deepcopy(A), fun!)
Base.adjoint(a::DTensor) = conj(a)

#= Operations on Tensors =#
function apply!(A::DASTensor, fun!)
    foreach(fun!, values(A))
    return A
end
apply(A::DASTensor, fun!) = apply!(deepcopy(A), fun!)
function Base.adjoint(A::DASTensor)
    B = apply(A, conj!)
    B.io = inv(B.io)
    return B
end
