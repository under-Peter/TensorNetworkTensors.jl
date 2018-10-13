module TNTensors
import TensorOperations
using TupleTools
using LinearAlgebra: svd

const TT = TupleTools
include("auxiliaryfunctions.jl")

export AbstractTensor
#type-tree
abstract type AbstractTensor{T,N} end

export DTensor
export fuselegs, splitlegs, tensorsvd

include("DTensors.jl")

export ZNTensor, U1Tensor, todense
include("DASTensors.jl")

include("splitfuse.jl")

end # module
