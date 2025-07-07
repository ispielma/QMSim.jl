"""
    ERD_SOC

A subtype that setups and solves the equal Rashba-Dresselhaus spin orbit coupling problem.
"""
module ERD_SOC

using LinearAlgebra

using QGas.NumericalTools.ArrayDimensions: Dimensions
using ...QMSim

# Required overrides
import ...QMSim: set_defaults!, error_check, build_rules!

export ERD_SOC_Solver

"""
    ERD_SOC_Solver

Describes an N-level spin-orbit coupled atomic gas.

This module assumes that energy is expressed in condensed matter
friendly units:

    To be consistent with AMO band structure standards, 
    the recoil momentum is q_r = 2 π / λ.
    
    The natural unit of energy is E = ħ^2 q_r^2 / 2*m, the single
    photon recoil energy.

    * q : quasi-momentum expressed in recoil units.

    * Ω : Raman coupling strength multiplying leads F matrices.

    * ϵ : Quadratic Zeeman shift, describing a downwards shift of
    the higher mf states.

    * δ : detuning
"""
mutable struct ERD_SOC_Solver{T, M<:AbstractMatrix{T}} <: AbstractMatrixSolver{T,M}
    shared::MatrixSharedData
    mwrs::QMSolver{T,M}
end
issolver(::Type{<:ERD_SOC_Solver}) = SolverTrait()
function ERD_SOC_Solver(adims::Dimensions)

    shared = MatrixSharedData(adims)    

    mwrs = QMSolver(Matrix{ComplexF64}, shared; wrap=Hermitian)

    add_leaf!(mwrs, :kinetic)
    add_leaf!(mwrs, :potential)

    solver = ERD_SOC_Solver(shared, mwrs)

    set_defaults!(solver)

    # check for errors
    error_check(solver)

    # Setup the rules
    build_rules!(solver)

    # Generate the efficient builders
    generate_builders!(solver)
end

function set_defaults!(solver::ERD_SOC_Solver;
    q::Float64=zero(Float64),
    Ω::ComplexF64=zero(ComplexF64),
    ϵ::Float64=zero(Float64),
    Δ::Float64=zero(Float64))

    # Defaults for kinetic (diagonal matrix)
    set_default_kwargs!(solver, :kinetic; q=q, ϵ=ϵ, Δ=Δ)

    # Defaults for potential (off-diagonal matrix)
    set_default_kwargs!(solver, :potential; Ω=Ω)
end

function error_check(solver::ERD_SOC_Solver)

    if ndims(get_dimensions(solver)) != 1
        throw(ArgumentError("ERD_SOC_Solver solver must have just a single spin dimension"))
    end

    error_check(get_leaf(solver))
end

"""
    build_rules!

Build the rules using the matrix-element approach.
"""
function build_rules!(solver::ERD_SOC_Solver)

    function kinetic(x0, x1; q=0.0, ϵ=0.0, Δ=0.0, kwargs...)

        mF = x0[1]

        return (q - mF).^2 - ϵ * mF^2 + Δ * mF
    end
    add_rule!(solver, :kinetic, RelativeRule, kinetic, [0,] )

    # Total angular momentum quantum number
    f = (size(get_dimensions(solver), 1) - 1.0) / 2.0

    function potential(x0, x1; Ω=0.0, kwargs...)

        mF = x0[1]
        δmF = x1[1] - x0[1]

        # Clebsch-Gordon Coefficient
        CG = sqrt(f*(f+1) - mF*(mF+δmF) )

        # Make Hermitian
        Ω = δmF > 0 ? Ω : conj(Ω)

        return Ω * CG / 2.0
    end
    add_rule!(solver, :potential, RelativeRule, potential, [+1,] )
    add_rule!(solver, :potential, RelativeRule, potential, [-1,] )
end

end # ERD_SOC