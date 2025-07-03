"""
    AbstractSolverTypes

This module defines all the abstract matrix types used in QMSim.jl and defines the interface for
these abstract types
"""

module AbstractSolverTypes

using ..AbstractMatrixTypes 
using ..MatrixBuilders

# for adding methods
import ..AbstractMatrixTypes: get_matrix, isleaf, get_leafs, add_leaf!

export AbstractMatrixSolver, issolver

#
# Define traits 
#

abstract type IsSolverTrait end
struct SolverTrait <: IsSolverTrait end
struct SolverFrameworkTrait <: IsSolverTrait end
struct SolverUndefinedTrait <: IsSolverTrait end

issolver(T::Type{<:Any}) = error("IsSolverTrait is not defined for type $(T)")
issolver(::T) where T = issolver(T) # value fallback

"""
    AbstractMatrixSolver

Encodes information required to actually obtain solutions to the matrix problem.  The standard interface
only assumes the existence of the field
    * mwrs :: MatricesWithRules [maybe change to AbstractMatrixWithRules, but I don't think the interface is ready yet]

    this must have the NodeTrait trait because it does not contain an array of rules.
"""

abstract type AbstractMatrixSolver{T, M} <: AbstractMatrixWithRules{T, M} end
isleaf(::Type{<:AbstractMatrixSolver}) = LinkTrait()
issolver(::Type{<:AbstractMatrixSolver}) = SolverUndefinedTrait()

# All suitable overloads go here...

end