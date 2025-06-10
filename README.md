# QMSim.jl

This package solves quantum mechanics problems where the underlying description (Hamiltonian / Lindbladian) are built from a set of rules (or specified directly).  This is targeting lattice problems such as lattices where, for example a real-space potential
\begin{equation}
\hat V = \frac{V_0}{2} \cos(2 k_r \hat x)
\end{equation}
for a lattice of depth $V_0$ with period $a = \pi / k_r$ becomes
\begin{equation}
\hat V = \frac{V_0}{4} \sum_k \ket{k}\bra{k + 2 k_2} + \mathrm{H.c}
\end{equation}
in momentum space.
This expresses a rule that a matrix element of $V_0/4$ exists for momentum states differing by $2 k_r$.
