struct U1 <: DAS end
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


struct U1Charge <: DASCharge
    ch::Int
end
Base.show(io::IO, s::U1Charge) = print(io,"U1Charge(",s.ch,")")

⊕(a::U1Charge, b::U1Charge) = U1Charge(a.ch + b.ch)
chargeindex(ch::U1Charge, chs::U1Charges) = div(ch.ch - chs.v.start, chs.v.step)+1
chargetype(::U1) = U1Charge
Base.zero(::Type{U1Charge}) = U1Charge(zero(Int))
Base.:+(chs::U1Charges, ch::U1Charge) = U1Charges(chs.v .+ ch.ch)
