# QMSim.jl

This package solves quantum mechanics problems where the underlying description (Hamiltonian / Lindbladian) is built from a set of rules (or specified directly).  Lattices are ones example of the modality where, for example a real-space potential

$$
\hat V = \frac{V_0}{2} cos(2 k_r \hat x)
$$

for a lattice of depth $V_0$ with period $a = \pi / k_r$ becomes

$$
\hat V = \frac{V_0}{4} \sum_k \ket{k + 2 k_2}\bra{k} + \mathrm{H.c}
$$

in momentum space.  This expresses a rule that a matrix element of $V_0/4$ exists for momentum states differing by $2 k_r$.  Atomic physics coupling graphs are another example.

See `examples/examples.ipynb` for example behavior.

## Status

The minimal functionality is now in place to setup simple models, IBS will be implementing new models and refining the code base as is required for Spielman Lab research. 

## Type hierarchy

There are two basic and disconnected groups of types: those derived from `AbstractRule` and those from `AbstractMatrixWithRules`.

The rules hierarchy provides individual construction rules that can be composed to generate matrices.
```
AbstractRule
    ├─ RelativeRule 
    ├─ ElementRule
    ├─ AbsoluteRule
    └─ ExplicitRule
```

The Matrices... hierarchy collects these rules together to construct matrices, and use them to efficiently encode physical problems.
```
AbstractMatrix
    └─ AbstractMatrixWithRules [UndefinedTrait]
        ├─ MatrixWithRules [LeafTrait]
        ├─ MatricesWithRules [NodeTrait]
        └─ AbstractMatrixSolver [LinkTrait, SolverUndefinedTrait]
            ├─ QMSolver [LinkTrait, SolverFrameworkTrait]
            ... specific solvers [LinkTrait, SolverTrait]
```
Notice that I have also defined traits `[UndefinedTrait, LeafTrait, NodeTrait, LinkTrait]` to specialize dispatch on other qualities of `AbstractMatrixWithRules`, and `[SolverUndefinedTrait, SolverFrameworkTrait, SolverTrait]` to separate out solver frameworks (like `QMSolver`) from actual solvers.


