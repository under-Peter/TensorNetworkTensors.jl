function groupby(f::Function, xs)
    # maintain order in xs
    dict = Dict()
    for x in xs
        y = f(x)
        if haskey(dict,y)
            push!(dict[y],x)
        else
            dict[y] = [x]
        end
    end
    return dict
end

function gatherby(f::Function, xs)
    return collect(values(groupby(f,xs)))
end
