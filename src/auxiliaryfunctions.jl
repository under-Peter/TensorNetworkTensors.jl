function groupby(f::Function, xs)
    # maintain order in xs
    y = f(first(xs))
    T = eltype(xs)
    dict = Dict{typeof(y), Vector{T}}()
    for x in xs
        y = f(x)
        haskey(dict, y) || (dict[y] = T[])
        push!(dict[y], x)
    end
    return dict
end

gatherby(f::Function, xs) = collect(values(groupby(f,xs)))
