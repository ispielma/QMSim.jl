"""
    MatrixSolvers

types tailored for solving specific types of problems.
"""
module MatrixSolvers

# External imports
using LinearAlgebra, SparseArrays, Arpack
import QGas.NumericalTools.ArrayDimensions as AD

# For overloading
import ..AbstractTypes: get_array, build_rules!, add_rule!, error_check!, generate_builders!, generate_builders
import ..MatrixBuilders: add_matrix!, build!

using ..AbstractTypes
using ..MatrixBuilders

export QMSolver, eigensystem!

"""
    QMSolver{T, M<:AbstractMatrix{T}}

Subtype of AbstractMatrixWithRules that is specifically for solving quantum mechanics problems.

    * `mwrs::MatricesWithRules{T,M}` : the builder for the matrix under consideration
    * `num_states::Int` : the number of states to solve for
    * `ranker::AbstractMatrix{T}` : sort eigenvalues and vectors in order of the expectation value of this matrix
    *  `wrap::Function` : a type such as Symmetric or Hermitian that the matrix can be wrapped in.

"""
mutable struct QMSolver{T, M<:AbstractMatrix{T}} <: AbstractMatrixSolver{T,M}
    mwrs::MatricesWithRules{T,M}
    eigenvalues::Vector{T}
    eigenvectors::Matrix{T}
    num_states::Int
    ranker::Union{AbstractMatrix{T},Nothing}
    wrap::Union{Type, Function}
end
function QMSolver(::Type{M}, adims::AD.Dimensions, args...; num_states=nothing, wrap=identity, ranker=nothing, kwargs...) where {T, M<:AbstractMatrix{T}}   M
    mwrs = MatricesWithRules(M, adims, args...; kwargs...)

    num_states = num_states === nothing ? dim(mwrs) : Int(num_states)
    # mwrs_reference = MatricesWithRules(args...; kwargs...)

    QMSolver(mwrs, 
             zeros(T, num_states), 
             zeros(T, num_states, dim(mwrs)), 
             num_states, 
             ranker, 
             wrap)
end

#
# Standard interface of parent type
#

get_array(qms::QMSolver) = qms.wrap(get_array(qms.mwrs))

function generate_builders!(qms::QMSolver)
    generate_builders!(qms.mwrs)
    return qms
end

function add_rule!(qms::QMSolver, args...; kwargs...)
    add_rule!(qms.mwrs, args...; kwargs...)
    return qms
end

function add_matrix!(qms::QMSolver, args...; kwargs...)
    add_matrix!(qms.mwrs, args...; kwargs...)
    return qms
end

function build!(qms::QMSolver, args...; kwargs...)
    build!(qms.mwrs, args...; kwargs...)
    return qms
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

    matrix = get_array(qms)

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