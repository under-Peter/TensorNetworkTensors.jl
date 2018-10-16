module TNTensors
import TensorOperations
using TupleTools
using LinearAlgebra: svd, diagm
import LinearAlgebra: diag

const TT = TupleTools
include("auxiliaryfunctions.jl")

export AbstractTensor
#type-tree
abstract type AbstractTensor{T,N} end

export DTensor
export fuselegs, splitlegs, tensorsvd

include("DTensors.jl")

export charges, sizes, in_out, tensor, chargesize
export setcharges!, setsizes!, setin_out!, settensor!
export ZNTensor, U1Tensor, todense
include("DASTensors.jl")

end # module
