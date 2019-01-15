using Test
using TNTensors
using TensorOperations: @tensor
using LinearAlgebra: diag
@testset "TNTensors" begin
    include("tensors.jl")
end
