using Documenter, TensorNetworkTensors

makedocs(;
    modules=[TensorNetworkTensors],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/under-Peter/TensorNetworkTensors.jl/blob/{commit}{path}#L{line}",
    sitename="TensorNetworkTensors.jl",
    authors="Andreas Peter",
)

deploydocs(;
    repo="github.com/under-Peter/TensorNetworkTensors.jl",
)
