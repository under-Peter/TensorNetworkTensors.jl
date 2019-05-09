# TensorNetworkTensors.jl
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://under-Peter.github.io/TensorNetworkTensors.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://under-Peter.github.io/TensorNetworkTensors.jl/dev)
[![Build Status](https://travis-ci.com/under-Peter/TensorNetworkTensors.jl.svg?branch=master)](https://travis-ci.com/under-Peter/TensorNetworkTensors.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/under-Peter/TensorNetworkTensors.jl?svg=true)](https://ci.appveyor.com/project/under-Peter/TensorNetworkTensors-jl)
[![Codecov](https://codecov.io/gh/under-Peter/TensorNetworkTensors.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/under-Peter/TensorNetworkTensors.jl)
[![Coveralls](https://coveralls.io/repos/github/under-Peter/TensorNetworkTensors.jl/badge.svg?branch=master)](https://coveralls.io/github/under-Peter/TensorNetworkTensors.jl?branch=master)

Tensors for Tensor Network Methods.
`TensorNetworkTensors.jl` overloads methods from `TensorOperations` and `KrylovKit` to work with Tensors that might be symmetric under some abelina symmetry.

## Installation
Enter `]` in `julia1.x` to enter pkg-mode.
If not installed, install `TensorOperations` first with
```julia
(v1.0) pkg> add TensorOperations
```
then you can go on to install `TensorNetworkTensors.jl` with
```julia
(v1.0) pkg> add https://github.com/under-Peter/TensorNetworkTensors.jl.git
```
and you're set up!

## Usage
### Tensors
You can define tensors without symmetry using the `DTensor` type as
```julia
julia> a = DTensor{Complex{Float64}}((2,2));
```
which will return a `2x2`-tensor with elements of type `Complex{Float64}`.
`DTensor` is just a thin wrapper around `Array`.

A tensor with a symmetry can be defined by specifying the:
1. symmetry
2. possible charge
3. sizes of the degeneracy spaces of those charge
4. action of the group acts on an index

The currently implemented _Discrete Abelian Symmetries_ are `U1` and `ZN` symmetries.
To define a `U1` symmetric rank-3 tensor which, in the particle conservation picture,
has two incoming, one outgoing leg and supports carrying between 3 and -3 particles
per leg where each charge has a degeneracy-size of 2, we write:
```julia
julia> sym = U1()
julia> chs = (U1Charges(-3:3), U1Charges(-3:3), U1Charges(-3:3))
julia> dims = ([2 for ch in U1Charges(-3:3)],
                [2 for ch in U1Charges(-3:3)],
                [2 for ch in U1Charges(-3:3)])
julia> io = InOut(1,1,-1)
julia> a = DASTensor{Float64,3}(sym, chs, dims, io)
```
where `a` is a _Discrete Abelian Symmetric_ rank-3 tensor with degeneracy tensors
of type `Float64`. It's first two indices can be read as _incoming_ while the last is
outgoing.
All indices support charges in `U1Charges(-3:3)` which can be looked at using
```julia
julia> foreach(println, U1Charges(-3:3))
U1Charge(-3)
U1Charge(-2)
U1Charge(-1)
U1Charge(0)
U1Charge(1)
U1Charge(2)
U1Charge(3)
```

The same for a Z2-symmetric tensor looks like
```julia
julia> sym = ZN{2}()
julia> chs = (ZNCharges{2}(), ZNCharges{2}(), ZNCharges{2}())
julia> dims = ([2 for ch in ZNCharges{2}()],
               [2 for ch in ZNCharges{2}()],
               [2 for ch in ZNCharges{2}()])
julia> io = InOut(1,1,-1)
julia> a = DASTensor{Float64,3}(sym, chs, dims, io)
```

Above we just defined tensors but they are either filled with garbage (`DTensor`)
or empty (`DASTensor`).
To initialize a tensor, use `initwithzero!` or `initwithrand!`, e.g.

```julia
julia> a = DTensor{Complex{Float64}}((2,2))
DTensor{Complex{Float64},2}Complex{Float64}[6.93789e-310+6.93789e-310im 6.93789e-310+6.93786e-310im; 6.93789e-310+6.93786e-310im 6.93786e-310+6.93789e-310im]
julia> initwithzero!(a)
DTensor{Complex{Float64},2}Complex{Float64}[0.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im]
```

For tensors with symmetry - `DASTensor` - the same works but the underlying structure is different:
```julia
julia> sym = ZN{2}()
julia> chs = (ZNCharges{2}(), ZNCharges{2}(), ZNCharges{2}())
julia> dims = ([2 for ch in ZNCharges{2}()],
               [2 for ch in ZNCharges{2}()],
               [2 for ch in ZNCharges{2}()])
julia> io = InOut(1,1,-1)
julia> a = DASTensor{Float64,3}(sym, chs, dims, io);
julia> initwithzero!(a)
julia> tensor(a)
```
To look into a `DASTensor`, you can use `tensor` to see the underlying dictionary
which maps `DASSectors`, discrete abelian symmetry sectors, to degeneracy tensors.
```julia
Dict{DASSector{3,ZNCharge{2}},Array{Float64,3}} with 4 entries:
  DASSector(Z2Charge(0), Z2Charge(0), Z2Charge(0)) => [0.0 0.0; 0.0 0.0]…
  DASSector(Z2Charge(0), Z2Charge(1), Z2Charge(1)) => [0.0 0.0; 0.0 0.0]…
  DASSector(Z2Charge(1), Z2Charge(0), Z2Charge(1)) => [0.0 0.0; 0.0 0.0]…
  DASSector(Z2Charge(1), Z2Charge(1), Z2Charge(0)) => [0.0 0.0; 0.0 0.0]…
```

You can also directly access a degeneracy tensor like this:
```julia
julia> a[DASSector(Z2Charge(0), Z2Charge(0), Z2Charge(0))]
2×2×2 Array{Float64,3}:
[:, :, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2] =
 0.0  0.0
 0.0  0.0
```

Specific functions for `DASTensors` include:
- `charges`, `setcharges!`
- `sizes`, `setsizes!`
- `in_out`, `setin_out!`
- `tensor`, `settensor!`
- `isinvariant`
- `charge`

to learn more, use `?` as in e.g.
```julia
julia>?charge
search: charge charges chargedim chargetype chargesize chargestype chargeindex ZNCharge Z2Charge U1Charge ZNCharges Z2Charges U1Charges DASCharge setcharges! DASCharges NDASCharge NDASCharges splitchargeit SplitChargeIt connectingcharge

  charge(a::DASSector)

  returns the charge which is calculated as the the sum of all charges it contains.

  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

  charge(a::DASTensor)

  returns the charge which is calculated as the charge of its non-zero sectors which needs to be unique.
```

### Reshaping

Reshaping can be done by either _fusing_ or _splitting_ legs.
Since splitting with symmetries is highly nontrivial,
it might only be done _after_ a fusion which provides the necessary information
of how to recombine indices and charges.

To fuse indices, we use the function `fuselegs`:
```julia
help?>fuselegs
  fuselegs(A, indexes)

  Fuse the indices in A as given by indexes where indexes is a tuple containing indices either alone or grouped in tuples - the latter will be fused. Returns a tuple of a tensor and the object necessary to undo the fusion.

  Examples
  ≡≡≡≡≡≡≡≡≡≡

  julia> a = DTensor(collect(reshape(1:8,2,2,2))
  DTensor{Int64,3}[1 3; 2 4]
  [5 7; 6 8]
  julia> fuselegs(a, ((1,2),3))
  (DTensor{Int64,2}[1 5; 2 6; 3 7; 4 8], ((2, 2),))

    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

  fuselegs(A, indexes[, io])

  For DASTensors, the directions of the resulting legs might be specified. If io is ommitted, the default is InOut(1,1,1...).
```
Fusion might also include permutation.
Note that `fuselegs` returns two objects:

1. A new tensor that corresponds to the input with the fusion applie
2. An object to _undo_ that fusion.

The latter is achieved using `splitlegs`:
```julia
help?> splitlegs
search: splitlegs splitlegs!

  splitlegs(A, indexes, rs...)

  Split the indices in A as given by indexes and rs. indexes is a tuple of integers and 3-tuple where each 3-tuple (i,j,k) specifies that index i in A is split according to rs[j], index k therein. Returns tensor with fused legs.

  Examples
  ≡≡≡≡≡≡≡≡≡≡

  julia> a = DTensor(collect(reshape(1:8,2,2,2))
  DTensor{Int64,3}[1 3; 2 4]
  [5 7; 6 8]
  julia> af, rs = fuselegs(a, ((1,2),3))
  (DTensor{Int64,2}[1 5; 2 6; 3 7; 4 8], ((2, 2),))
  julia> splitlegs(af, ((1,1,1),(1,1,2),2), rs...)
  DTensor{Int64,3}[1 3; 2 4]
  [5 7; 6 8]
```

The same is true for the case of `DASTensor`s, although `fuselegs` returns a more complicated object
of the type `Reshaper`.
If there is already a tensor of the correct shape and charges available,
both `fuselegs` and `splitlegs` can be used as their `in-place` version
`fuselegs!` and `splitlegs!` (see `help`).

Note that fusion is specified with _permutation_ of the indices of a tensor
and indices grouped in a tuple are fused,
e.g. `((1,2),3)` means the first two indices of a rank-3 tensor are grouped
whereas `(1,(4,3),2)` means that the third and fourth index of a rank-4
tensor are grouped and switched with the second index.

For splitting, indices are a list of integers and 3-tuples `(i,j,k)`
where the latter specifies that index `i` in given tensor is split
according to index `j` in the reshapeing-information and part `k` of that
split is to be at the position of that tuple.

Let's look at an example:
```julia
julia> sym = U1()
julia> chs = (U1Charges(-3:3), U1Charges(-3:3), U1Charges(-3:3))
julia> dims = ([2 for ch in U1Charges(-3:3)],
                [2 for ch in U1Charges(-3:3)],
                [2 for ch in U1Charges(-3:3)])
julia> io = InOut(1,1,1)
julia> a = DASTensor{Float64,3}(sym, chs, dims, io)
julia> initwithrand!(a)
```
First we fuse indices 1 and 3 into `(3,1)` putting them in the first place and
specifying the directions of the legs as both outgoing:
```julia
julia> af, rs = fuselegs(a, ((3,1),2),InOut(-1,-1))
```

To undo that fusion, we need to split the first index of `af` according to the
first reshaper in `rs` (there's a reshaper for each index in the fused tensor in order).
A valid splitting would be:
```julia
julia> splitlegs(af,((1,1,1),(1,1,2),2),rs...)
```
but that would correspond to a permutation `(3,1,2)` of `a`.
Additionally, the unfused index was also changed by switching its `InOut`!
We thus want to specify both the correct permutation and that index 2 needs
to be changed accordingly, arriving at
```julia
julia> splitlegs(af, ((1,1,2),(2,2,1),(1,1,1)), rs...) ≈ a
true
```

## Factorizations

So far the factorizations available are `tensorsvd` which returns the `SVD`
of a tensor, see

```julia
help?> tensorsvd
search: tensorsvd tensorsvd! _tensorsvd TensorOperations

  tensorsvd(A::AbstractTensor, indexes; svdtrunc)

  works like tensorsvd except that A can have arbitrary rank.
  indexes specifies which indices the fuse for A to be a rank-2 tensor as in fuselegs.
  The tensor is then fused, tensorsvd applied and split again.

  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

  tensorsvd(A::AbstractTensor{T,2}; svdtrunc)

  returns the svd of an AbstractTensor.
  svdtrunc is a function that has the singular values as input and returns a number specifying how many of them to keep.
  The default svdtrunc is svdtrunc_default and keeps all of them.

  Other options are: svdtruncdiscardzero svdtruncmaxχ svdtruncmaxcumerror svdtruncmaxerror
```
and `tensorqr` which returns the `QR`-decomposition of a tensor, see
```julia
help?> tensorqr
search: tensorqr tensor tensorsvd tensoradd tensorsvd! tensortrace tensorcopy tensoradd! tensortrace! tensorcopy! tensorproduct tensorproduct! tensorcontract

  tensorqr(A::AbstractTensor)

  returns tensor Q,R such that A = QR and Q is an orthogonal/unitary matrix and R is an upper triangular matrix.

  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

  tensorqr(A::AbstractTensor, inds)

  returns the tensorqr of A fused according to inds.
```

# KrylovKit
All `AbstractTensor`s work with `KrylovKit`,
a package that combines a number of Krylov-based algorithms
including _exponentiate_.

To use KrylovKit, follow the instruction for installation at the [github repository](https://github.com/Jutho/KrylovKit.jl) and do
```julia
using KrylovKit
```

As an example, consider `eigsolve` for a tensor with two independent symmetries:
```julia
julia> a = DASTensor{Complex{Float64},2}(
  NDAS(Z2(),U1()),
  (NDASCharges(Z2Charges(), U1Charges(-1:1)), NDASCharges(Z2Charges(), U1Charges(-1:1))),
  (fill(2,6), fill(2,6)),
  InOut(1,-1))
julia> initwithrand!(a)
julia> using TensorOperations
julia> @tensor a[1,2] := a[1,2] + a'[2,1];
```

Since we are not working with a regular array,
we need to provide an initial guess for an eigenvector, e.g.
```julia
julia> v0 = DASTensor{Complex{Float64},1}(
         NDAS(Z2(),U1()),
         (NDASCharges(Z2Charges(), U1Charges(-1:1)),),
         (fill(2,6),),
         InOut(1))
julia> initwithrand!(v0)
```

Then we can define a function for applying `a` to vectors like `v0`:
```julia
julia> f(v) = @tensor v2[1] :=  a[1,-1] * v[-1]
```

Then we can use `eigsolve` with `f` and `v0` to get an
eigenvector of `a` or the map that it describes:
```julia
julia> eigsolve(f,v0)
[...]
```
which returns two eigenvalues, two eigenvectors and an object containing information about convergence.

Similarly we can apply `exp(0.1*a)` to `v0` with `exponentiate`:
```julia
julia> exponentiate(f,0.1,v0, ishermitian=true)
```
