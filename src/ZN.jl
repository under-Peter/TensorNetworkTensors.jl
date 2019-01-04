struct ZN{N} <: DAS end
struct ZNCharges{N} <: DASCharges end
Base.show(io::IO, s::ZNCharges{N}) where N = print(io,"Z",N,"Charges(",0:(N-1),")")

⊕(::ZNCharges{N}, ::ZNCharges{N}) where N = ZNCharges{N}()
Base.inv(::ZNCharges{N}) where N = ZNCharges{N}()
Base.length(::ZNCharges{N}) where N = N
Base.getindex(s::ZNCharges{N}, i::Int) where N = ZNCharge{N}(i-1)
chargestype(::ZN{N}) where N = ZNCharges{N}
Base.eltype(::Union{ZNCharges{N},Type{ZNCharges{N}}}) where N = ZNCharge{N}
Base.intersect(::ZNCharges{N}, ::ZNCharges{N}) where N = ZNCharges{N}()
Base.intersect(::ZNCharges, ::ZNCharges) = ArgumentError("cannot intersect different Z charges")


struct ZNCharge{N} <: DASCharge
    ch::Int
    ZNCharge{N}(a) where {N} = new(mod(a,N))
end
Base.show(io::IO, s::ZNCharge{N}) where N = print(io,"Z",N,"Charge(",s.ch,")")

⊕(a::ZNCharge{N}, b::ZNCharge{N}) where N = ZNCharge{N}(a.ch + b.ch)
chargeindex(ch::ZNCharge{N}, chs::ZNCharges{N}) where N = (ch.ch + 1)
chargetype(::ZN{N}) where N = ZNCharge{N}
Base.zero(::Type{ZNCharge{N}}) where N = ZNCharge{N}(zero(Int))
Base.:+(chs::ZNCharges{N}, ch::ZNCharge{N}) where N = ZNCharges{N}()


const Z2 = ZN{2}
const Z2Charges = ZNCharges{2}
const Z2Charge = ZNCharge{2}
