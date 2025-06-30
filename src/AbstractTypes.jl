"""
    AbstractTypes

This module defines all the abstract types used in QMSim.jl and defines the interface for
these abstract types
"""

module AbstractTypes

# Will be adding methods to these functions
import Base: size, getindex, setindex!, length, iterate, eltype, isdone, ndims
import SparseArrays: issparse

import QGas.NumericalTools.ArrayDimensions as AD

export AbstractRule, AbstractRuleBuilder, AbstractMatrixWithRules
export get_matrix, set_matrix!
export build_rules!, error_check, set_default_kwargs!, add_rule!, generate_builders!, generate_builders, build!, build

export AbstractMatrixSolver

#=
########  ##     ## ##       ########  ######
##     ## ##     ## ##       ##       ##    ##
##     ## ##     ## ##       ##       ##
########  ##     ## ##       ######    ######
##   ##   ##     ## ##       ##             ##
##    ##  ##     ## ##       ##       ##    ##
##     ##  #######  ######## ########  ######
=#

"""
    `AbstractRule(args...; kwargs...)` → complex number (or whatever type of matrix we are looking at)
    `AbstractRuleBuilder` 

Top-level type for all rule objects.

concrete type of `AbstractRuleBuilder` will be callable with arguments:
    * mat :: AbstractMatrix matrix to add to
    * index_to_values :: Vector{Vector{Float64}} mapping from linear array indices to coordinate dimensions

Each concrete type of `AbstractRuleBuilder` is expected to provide an outer constructor with the following signature:
    ```
    AbstractRuleBuilder(
        rules::Vector{ConcereteRuleType},
        adims::AD.Dimensions,
        index_to_coords::Vector{Vector{Int}})
    ```
note that args... and kwargs... can be used in the constructors to ignore arguments that are not needed.

The following is a specific example for a rule that attaches a hard-coded matrix
```
struct ExplicitRule{M<:AbstractMatrix} <: AbstractRule
    matrix::M
end
function ExplicitRule(::Type{M}, mat) where M

    # error checking would go here...

    mat = M(mat)

    ExplicitRule(mat)
end

 # only the matrix builder for this type needs to know this call signature, so it is not part of the
 # interface rules
(a::ExplicitRule)() = a.matrix

struct ExplicitRuleBuilder <: AbstractRuleBuilder
    actions::Vector{ExplicitRule}
end
matrix_builder(::Type{ExplicitRule}, args...; kwargs...) = ExplicitRuleBuilder(args...; kwargs...)

ExplicitRuleBuilder(rules, args...; kwargs...) = ExplicitRuleBuilder(rules)

function (rb::ExplicitRuleBuilder)(mat, args...; kwargs...)
    mat += mapreduce(r -> r.matrix, +, rb.actions)

    return mat
end
```

"""
abstract type AbstractRule end

abstract type AbstractRuleBuilder end

# Introduce traits that are needed to connect the rule type with the associated matrix builder 
matrix_builder(::Type{T}) where {T<:AbstractRule} = error("No builder defined for rule‐type $T")

(rb::AbstractRuleBuilder)(mat, index_to_values; kwargs...) = mat

#=
##     ##    ###    ######## ########  #### ##     ## ##      ## #### ######## ##     ## ########  ##     ## ##       ########  ######
###   ###   ## ##      ##    ##     ##  ##   ##   ##  ##  ##  ##  ##     ##    ##     ## ##     ## ##     ## ##       ##       ##    ##
#### ####  ##   ##     ##    ##     ##  ##    ## ##   ##  ##  ##  ##     ##    ##     ## ##     ## ##     ## ##       ##       ##
## ### ## ##     ##    ##    ########   ##     ###    ##  ##  ##  ##     ##    ######### ########  ##     ## ##       ######    ######
##     ## #########    ##    ##   ##    ##    ## ##   ##  ##  ##  ##     ##    ##     ## ##   ##   ##     ## ##       ##             ##
##     ## ##     ##    ##    ##    ##   ##   ##   ##  ##  ##  ##  ##     ##    ##     ## ##    ##  ##     ## ##       ##       ##    ##
##     ## ##     ##    ##    ##     ## #### ##     ##  ###  ###  ####    ##    ##     ## ##     ##  #######  ######## ########  ######
=#

"""
    AbstractMatrixWithRules

Abstract MatrixWithRules type, used to define the interface.  The default interface assumes the fields:

* `rules` :: Vector{AbstractRule}
* `matrix` :: AbstractMatrix
* `adims` :: AD.Dimensions
* `_default_kwargs` :: Dict{Symbol, Any}

"""
abstract type AbstractMatrixWithRules{T, M} <: AbstractMatrix{T} end

get_matrix(mwr::AbstractMatrixWithRules) = mwr.matrix
set_matrix!(mwr::AbstractMatrixWithRules, matrix::AbstractMatrix) = mwr.matrix = matrix

get_rules(mwr::AbstractMatrixWithRules) = mwr.rules
set_rules!(mwr::AbstractMatrixWithRules, rules) = mwr.rules = rules

get_dimensions(mwr::AbstractMatrixWithRules) = mwr.adims
set_dimensions!(mwr::AbstractMatrixWithRules, adims) = mwr.adims = adims

get_default_kwargs(mwr::AbstractMatrixWithRules) = mwr._default_kwargs
set_default_kwargs!(mwr::AbstractMatrixWithRules, kwargs) = mwr._default_kwargs = kwargs
set_default_kwargs!(mwr::AbstractMatrixWithRules, args...; kwargs...) = set_default_kwargs!(mwr, args..., kwargs)

#
# Implement array interface
#

size(mwr::AbstractMatrixWithRules) = size(get_matrix(mwr))
getindex(mwr::AbstractMatrixWithRules, args...) = getindex(get_matrix(mwr), args...)
setindex!(mwr::AbstractMatrixWithRules, args...) = setindex!(get_matrix(mwr), args...)
length(mwr::AbstractMatrixWithRules) = length(get_matrix(mwr)) 

ndims(mwr::AbstractMatrixWithRules) = ndims(get_matrix(mwr)) 

iterate(mwr::AbstractMatrixWithRules, args...) = iterate(get_matrix(mwr), args...)
eltype(mwr::AbstractMatrixWithRules) = eltype(get_matrix(mwr))
isdone(mwr::AbstractMatrixWithRules, args...) = Base.isdone(get_matrix(mwr), args...)

issparse(mwr::AbstractMatrixWithRules) = issparse(get_matrix(mwr))

#
# New methods for AbstractMatrixWithRules
#

"""
    build_rules!(mwr::AbstractMatrixWithRules)

build the efficient build-tables based on the rules encoded in the matrix
"""
build_rules!(mwr::AbstractMatrixWithRules) = mwr

"""
    error_check(mwr::AbstractMatrixWithRules)

check for errors
"""
function error_check(mwr::AbstractMatrixWithRules) 
    
    # The only possible error at this point is a dimension mismatch between the matrix and the dimensions
    len_mwr = length(mwr)
    len_dims = length(get_dimensions(mwr))
    if len_mwr != len_dims
        throw(DimensionMismatch("Length of stored matrix $(len_mwr) does not match length of dimensions $(len_dims)"))
    end
    
end

"""
    `add_rule!(mwr::AbstractMatrixWithRules, rule::AbstractRule)`

Adds a rule of any type to `mwr`.  Methods exist that directly take the arguments for each rule type.
"""
add_rule!(mwr::AbstractMatrixWithRules, rule::AbstractRule) = push!(get_rules(mwr), rule)

"""
    `add_rule!(mwr::MatrixWithRules, RuleType::Type{R <: AbstractRule}, args...)

Adds a rule of any type to `mwr`.  Dispatch to the type provided by RuleType.
"""
add_rule!(mwr::AbstractMatrixWithRules{T, M}, RuleType::Type{R}, args...) where {T, M, R <:AbstractRule} = add_rule!(mwr, RuleType(M, args...))

"""
    generate_builders(mwr::AbstractMatrixWithRules)::Vector{AbstractRuleBuilder}()
    generate_builders!(mwr::AbstractMatrixWithRules)::AbstractMatrixWithRules

Interface for generating builders
"""
generate_builders(::AbstractMatrixWithRules) = Vector{AbstractRuleBuilder}()
generate_builders!(mwr::AbstractMatrixWithRules) = mwr

build(mwr::MatrixWithRules{T, M}, args...; kwargs...) = get_matrix(mwr)
build!(mwr::MatrixWithRules{T, M}, args...; kwargs...) = set_matrix!(mwr, get_matrix(mwr))

#=
##     ##    ###    ######## ########  ####  ######  ########  ######  ##      ## #### ######## ##     ## ########  ##     ## ##       ########  ######
###   ###   ## ##      ##    ##     ##  ##  ##    ## ##       ##    ## ##  ##  ##  ##     ##    ##     ## ##     ## ##     ## ##       ##       ##    ##
#### ####  ##   ##     ##    ##     ##  ##  ##       ##       ##       ##  ##  ##  ##     ##    ##     ## ##     ## ##     ## ##       ##       ##
## ### ## ##     ##    ##    ########   ##  ##       ######    ######  ##  ##  ##  ##     ##    ######### ########  ##     ## ##       ######    ######
##     ## #########    ##    ##   ##    ##  ##       ##             ## ##  ##  ##  ##     ##    ##     ## ##   ##   ##     ## ##       ##             ##
##     ## ##     ##    ##    ##    ##   ##  ##    ## ##       ##    ## ##  ##  ##  ##     ##    ##     ## ##    ##  ##     ## ##       ##       ##    ##
##     ## ##     ##    ##    ##     ## ####  ######  ########  ######   ###  ###  ####    ##    ##     ## ##     ##  #######  ######## ########  ######
=#

"""
    AbstractMatricesWithRules

Abstract MatrixWithRules type, used to define the interface.  The default interface assumes the fields:

* `adims` :: AD.Dimensions
* `mwrs` :: Vector{AbstractMatrixWithRules}
* `matrix` :: AbstractMatrix

"""

abstract type AbstractMatricesWithRules{T, M} <: AbstractMatrixWithRules{T} end
get_matrices(mwrs::AbstractMatricesWithRules) = mwrs.matrices

# Implement AbstractMatrixWithRules interface
get_matrix(mwrs::AbstractMatricesWithRules, name::Symbol) = get_matrices(mwrs)[name]
set_matrix!(mwrs::AbstractMatricesWithRules, name::Symbol, matrix) = set_matrix!(get_matrix(mwrs, name), matrix)

get_default_kwargs(mwrs::AbstractMatricesWithRules; kwargs) = error("Must specify which element of a MatricesWithRules is being accessed")
get_default_kwargs(mwrs::AbstractMatricesWithRules, name::Symbol) = get_default_kwargs(get_matrix(mwrs, name))

set_default_kwargs!(mwrs::AbstractMatricesWithRules, kwargs) = error("Must specify which element of a MatricesWithRules is being set")
set_default_kwargs!(mwrs::AbstractMatricesWithRules, name::Symbol, kwargs) = set_default_kwargs!(get_matrix(mwrs, name), kwargs)

add_rule!(mwrs::AbstractMatricesWithRules, name::Symbol, args...; kwargs...) = add_rule!(get_matrix(mwrs, name), args...; kwargs...)

function generate_builders!(mwrs::AbstractMatricesWithRules)
    for (_, mwr) in get_matrix(mwrs, name)
        generate_builders!(mwr)
    end
    return mwrs
end

"""
    build!(mwrs::AbstractMatricesWithRules; names=nothing, kwargs...)

method of build! for abstract type AbstractMatricesWithRules.  Adds a keyword argument `name` that
optionally specifies which of the matrices to build.
"""
function build!(mwrs::AbstractMatricesWithRules; names=nothing, kwargs...)

    matrices = get_matrices(mwrs)

    if names === nothing
        names = keys(matrices)
    end

    for name in names
        mwr = matrices[name]
        build!(mwr; kwargs...)
    end

    matrix = mapreduce(mwr -> get_matrix(mwr), +, values(matrices))
    set_matrix!(mwrs, matrix)
end

add_matrix!(mwrs::AbstractMatricesWithRules, name::Symbol, args...; kwargs...) = error("add_matrix! must be implemented for type $(typeof(mwrs))")

#=
 ######   #######  ##       ##     ## ######## ########   ######
##    ## ##     ## ##       ##     ## ##       ##     ## ##    ##
##       ##     ## ##       ##     ## ##       ##     ## ##
 ######  ##     ## ##       ##     ## ######   ########   ######
      ## ##     ## ##        ##   ##  ##       ##   ##         ##
##    ## ##     ## ##         ## ##   ##       ##    ##  ##    ##
 ######   #######  ########    ###    ######## ##     ##  ######
=#

"""
    AbstractMatrixSolver

Encodes information required to actually obtain solutions to the matrix problem
"""


abstract type AbstractMatrixSolver{T, M} <: AbstractMatricesWithRules{T, M} end
get_matrix(ms::AbstractMatrixSolver) = get_matrix(ms.mwrs)

end