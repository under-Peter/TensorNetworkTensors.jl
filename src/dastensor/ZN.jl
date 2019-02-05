"""
    ZN{N} <: DAS
ZN symmetry singleton type where N is an Int.
"""
struct ZN{N} <: DAS end
Base.show(io::IO, ::ZN{N}) where N = print(io,"Z",N)

"""
    ZNCharges{N} <: DASCharges
singleton-type for collections of `ZNCharge{N}` objects. Only the type parameter
is provided - `ZNCharges{N}` doesn't have a field.
Upon iteration yields `ZNCharge{N}(i)` for `i = 0,...,N-1`

# Example
```julia-repl
julia>a = ZNCharges{2}()
julia>foreach(println, a)
ZNCharge{2}(0)
ZNCharge{2}(1)
```
"""
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


"""
    ZNCharge{N} <: DASCharge
holds the charge of a ZN symmetry as an integer.
The integer is taken `mod N` s.t. the charge is always between `0` and `N-1`

# Example
```julia-repl
julia>a = ZNCharge{2}(1);
julia>a ⊕ a
ZNCharge{2}(0)

julia>a = ZNCharge{2}(3) == ZNCharge{2}(1)
true
```
"""
struct ZNCharge{N} <: DASCharge
    ch::Int
    ZNCharge{N}(a) where {N} = new(mod(a,N))
end
Base.show(io::IO, s::ZNCharge{N}) where N = print(io,"Z",N,"Charge(",s.ch,")")

⊕(a::ZNCharge{N}, b::ZNCharge{N}) where N = ZNCharge{N}(a.ch + b.ch)
function chargeindex(ch::ZNCharge{N}, chs::ZNCharges{N}) where N
    ch in chs || throw(ArgumentError(string(ch, " not in ", chs)))
    (ch.ch + 1)
end
chargetype(::ZN{N}) where N = ZNCharge{N}
Base.zero(::Type{ZNCharge{N}}) where N = ZNCharge{N}(zero(Int))
Base.:+(chs::ZNCharges{N}, ch::ZNCharge{N}) where N = ZNCharges{N}()


const Z2 = ZN{2}
const Z2Charges = ZNCharges{2}
const Z2Charge = ZNCharge{2}
