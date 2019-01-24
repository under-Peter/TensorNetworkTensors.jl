using Test
using TensorNetworkTensors
using TensorOperations: @tensor, tensorcopy
using LinearAlgebra: diag
@testset "TensorNetworkTensors" begin
    include("tensors.jl")
end
