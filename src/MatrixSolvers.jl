"""
    MatrixSolvers

types tailored for solving specific types of problems.
"""
module MatrixSolvers

# External imports
using LinearAlgebra, SparseArrays, Arpack
import QGas.NumericalTools.ArrayDimensions as AD

# Internal imports
using ..Helpers
using ..AbstractSolverTypes: AbstractMatrixSolver, IsSolverTrait, SolverTrait, SolverFrameworkTrait, SolverUndefinedTrait
using ..AbstractMatrixTypes
using ..MatrixBuilders: MatricesWithRules

export QMSolver, eigensystem!, rank_ordering!

# To add methods
import ..AbstractMatrixTypes: isleaf
import ..AbstractSolverTypes: issolver

"""
    QMSolver{T, M<:AbstractMatrix{T}}

Subtype of AbstractMatrixWithRules that is specifically for solving quantum mechanics problems.

    * `mwrs::MatricesWithRules{T,M}` : the builder for the matrix under consideration
    * `num_states::Int` : the number of states to solve for
    * `ranker::AbstractMatrix{T}` : sort eigenvalues and vectors in order of the expectation value of this matrix
    *  `wrap::Function` : a type such as Symmetric or Hermitian that the matrix can be wrapped in.

"""
mutable struct QMSolver{T, M<:AbstractMatrix{T}} <: AbstractMatrixSolver{T,M}
    shared      :: MatrixSharedData
    mwrs         :: MatricesWithRules{T,M} # TODO: change this to be any AbstractMatrixWithRules?
    eigenvalues ::Vector{T}
    eigenvectors::Matrix{T}
    num_states  ::Int
    ranker      ::Union{AbstractMatrix{T},Nothing} # TODO: change this to a symbol key in the reference matrices?
    wrap        ::Union{Type, Function}
end
issolver(::Type{<:QMSolver}) = SolverFrameworkTrait()

function QMSolver(
        ::Type{M}, 
        adims::AD.Dimensions;
        options=Dict{Symbol,Any}(), # MatrixSharedData field
        cache_kwargs=true, # MatrixSharedData field 
        num_states=nothing, 
        wrap=identity,
        ranker=nothing
    ) where {T, M<:AbstractMatrix{T}}

    shared = MatrixSharedData(adims, options, cache_kwargs)    
    mwrs = MatricesWithRules(M, shared)

    num_states = num_states === nothing ? dim(mwrs) : Int(num_states)
    # mwrs_reference = MatricesWithRules(args...; kwargs...)

    QMSolver(
        shared,
        mwrs, 
        zeros(T, num_states), 
        zeros(T, num_states, dim(mwrs)), 
        num_states, 
        ranker, 
        wrap
    )
end

#
# New methods
#

"""
    rank_ordering(qms::QMSolver)

Returns the order of the eigenvalues, either as a function
of increasing energy if qms.ranker = nothing, or based on the expectation value of `Hermitian(qms.ranker)`

in the future qms.ranker might be an MatrixWithRules in which type in will be evaluated using the currently
keyword arguments.
"""
function rank_ordering!(qms::QMSolver)
    ψ = qms.eigenvectors
    if isnothing(qms.ranker) || isnothing(ψ)
        ordering = sortperm(real.(qms.eigenvalues))
    else
        ordering = ψ' * Hermitian(qms.ranker) * ψ
    end

    # now resort the eigenvalues and vectors.
    qms.eigenvalues = qms.eigenvalues[ordering]
    qms.eigenvectors = qms.eigenvectors[:, ordering]

    qms
end

"""
In the future use ArnoldiMethod.jl, but right now this is not a complete replacement of eigs
"""
function eigensystem!(qms::QMSolver; names=nothing, which=:SR, kwargs...)
    build!(qms.mwrs; names=names, kwargs...)

    matrix = get_matrix(qms)

    if issparse(qms)
        qms.eigenvalues, qms.eigenvectors = eigs(matrix; nev=qms.num_states, which=which, maxiter=10000)
    else
        factors = eigen(matrix)
        qms.eigenvalues = factors.values
        qms.eigenvectors = factors.vectors
    end

    if issymmetric(qms) || ishermitian(qms)
        qms.eigenvalues = real(qms.eigenvalues)
    end

    rank_ordering!(qms)

    return qms
end

end # MatrixSolvers