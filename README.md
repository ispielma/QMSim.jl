# QMSim.jl

This package solves quantum mechanics problems where the underlying description (Hamiltonian / Lindbladian) is built from a set of rules (or specified directly).  Lattices are ones example of the modality where, for example a real-space potential
$$
\hat V = \frac{V_0}{2} \cos(2 k_r \hat x)
$$
for a lattice of depth $V_0$ with period $a = \pi / k_r$ becomes
$$
\hat V = \frac{V_0}{4} \sum_k \ket{k + 2 k_2}\bra{k} + \mathrm{H.c}
$$
in momentum space.
This expresses a rule that a matrix element of $V_0/4$ exists for momentum states differing by $2 k_r$.

Atomic physics coupling graphs are another example.