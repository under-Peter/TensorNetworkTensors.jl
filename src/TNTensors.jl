module TNTensors
import TensorOperations
using TupleTools
using LinearAlgebra: svd, diagm
import LinearAlgebra: diag
using TensorOperations
using Parameters

const TT = TupleTools
include("auxiliaryfunctions.jl")

export AbstractTensor
#type-tree
abstract type AbstractTensor{T,N} end

export DTensor
export fuselegs, splitlegs, tensorsvd

include("DTensors.jl")

export charges, sizes, in_out, tensor, chargesize
export diag, apply!
export setcharges!, setsizes!, setin_out!, settensor!
export ZNTensor, U1Tensor, DASTensor, todense
export svdcutfun_default, svdcutfun_discardzero , svdcutfun_maxχ,
    svdcutfun_maxcumerror, svdcutfun_maxerror
include("DASTensors.jl")

toarray(a::DTensor) = a.array
toarray(a::DASTensor) = todense(a).array
export toarray

include("krylovkit.jl")

end # module
