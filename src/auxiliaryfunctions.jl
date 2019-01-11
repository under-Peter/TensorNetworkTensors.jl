"""
    groupby(f, xs)
group values `x ∈ xs` by the result of applying `f`.
Returns a `Dict` with keys `yi` and values `[xi1, xi2,...]` such that `f(xij) = yi`.

# Example
```julia-repl
julia> groupby(isodd,1:10)
Dict{Bool,Array{Int64,1}} with 2 entries:
  false => [2, 4, 6, 8, 10]
  true  => [3, 5, 7, 9]
```
"""
function groupby(f::Function, xs)
    # maintain order in xs
    sxs = Iterators.Stateful(xs)
    T = eltype(sxs)
    x = first(sxs)
    y = f(x)
    dict = Dict{typeof(y), Vector{T}}()
    dict[y] = T[x]
    for x in sxs
        y = f(x)
        haskey(dict, y) || (dict[y] = T[])
        push!(dict[y], x)
    end
    return dict
end


"""
    gatherby(f, xs)
Like `groupby` but only returns values, i.e. elements of `xs` in `Vector`s such that
`f` applied to an element of a group is the same as for any other element of that
`Vector`.

# Example
```julia-repl
julia> gatherby(isodd, 1:10)
2-element Array{Array{Int64,1},1}:
 [2, 4, 6, 8, 10]
 [3, 5, 7, 9]
```
"""
gatherby(f::Function, xs) = collect(values(groupby(f,xs)))
