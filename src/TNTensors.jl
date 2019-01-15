module TNTensors
import TensorOperations, LinearAlgebra
using TupleTools
using Parameters

const TT = TupleTools
const TO = TensorOperations
const LA = LinearAlgebra

export gatherby, groupby
include("auxiliaryfunctions.jl")

export AbstractTensor
"""
    AbstractTensor{T,N}
Abstract supertype for all tensornetwork-tensors.
"""
abstract type AbstractTensor{T,N} end

export DTensor
include("DTensors.jl")

export DAS, DASCharges, DASCharge, NDAS, NDASCharges, NDASCharge,
        U1, U1Charges, U1Charge, ZN, ZNCharges, ZNCharge
export Z2, Z2Charges, Z2Charge
export chargeindex, ⊗, ⊕, DASSector, InOut, charge, deleteat, permute
export allsectors, covariantsectors, invariantsectors
export DASTensor
export charges, sizes, in_out, tensor, symmetry, setcharges!, setsizes!, setin_out!,
    settensor!, isinvariant, initwithrand!, initwithzero!, todense
export chargestype, chargetype, dimstype, symmetry

include("DASTensors.jl")

#Multiplication and Division with Numbers
export apply, apply!
include("tensorops.jl")

#Overload functions from TensorOperations
include("tensoroperations.jl")

#Reshaping: Splitting and fusing legs of tensors
export  fuselegs, fuselegs!,
        splitlegs, splitlegs!
include("splitfuse.jl")

#Factorizations of Tensors
export tensorsvd, tensorsvd!
export  svdtrunc_default,
        svdtrunc_discardzero,
        svdtrunc_maxχ,
        svdtrunc_maxcumerror,
        svdtrunc_maxerror
include("tensorfactorization.jl")

#Overload the functions required for use with KrylovKit
include("krylovkit.jl")
end # module
