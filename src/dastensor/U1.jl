"""
    U1 <: DAS
U1 symmetry singleton-type
"""
struct U1 <: DAS end
Base.show(io::IO, ::U1) = print(io,"U1")
"""
    U1Charges <: DASCharges
type for collection of `U1Charge` that holds `StepRange{Int,Int}` which represents the valid
values for `U1Charge`.

# Example
```julia-repl
julia>a = U1Charges(-1:1);
julia>foreach(println, a)
U1Charge(-1)
U1Charge(0)
U1Charge(1)
```
"""
struct U1Charges <: DASCharges
    v::StepRange{Int,Int}
end
Base.show(io::IO, s::U1Charges) = print(io,"U1Charges(",s.v,")")

function ⊕(a::U1Charges, b::U1Charges)
    loa, hia = extrema(a.v)
    lob, hib = extrema(b.v)
    step = min(abs(a.v.step), abs(b.v.step))
    return U1Charges((loa + lob):step:(hia + hib))
end
Base.inv(a::U1Charges) = U1Charges(-a.v.stop:(a.v.step):-a.v.start)
Base.length(a::U1Charges) = length(a.v)
Base.getindex(s::U1Charges, i::Int) = U1Charge(s.v[i])
chargestype(::U1) = U1Charges
Base.eltype(::Union{U1Charges, Type{U1Charges}}) = U1Charge
Base.intersect(a::U1Charges, b::U1Charges) = U1Charges(intersect(a.v,b.v))


"""
    U1Charge <: DASCharge
holds the charge of a U1 symmetry as an integer.

# Example
```julia-repl
julia>a = U1Charge(1);
julia>a ⊕ a
U1Charge(2)
```
"""
struct U1Charge <: DASCharge
    ch::Int
end
Base.show(io::IO, s::U1Charge) = print(io,"U1Charge(",s.ch,")")

⊕(a::U1Charge, b::U1Charge) = U1Charge(a.ch + b.ch)
function chargeindex(ch::U1Charge, chs::U1Charges)
    ch in chs || throw(ArgumentError(string(ch, " not in ", chs)))
    div(ch.ch - chs.v.start, chs.v.step)+1
end
chargetype(::U1) = U1Charge
Base.zero(::Type{U1Charge}) = U1Charge(zero(Int))
Base.:+(chs::U1Charges, ch::U1Charge) = U1Charges(chs.v .+ ch.ch)
