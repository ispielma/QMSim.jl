"""
    AbstractTypes

This module defines all the abstract types used in QMSim.jl and defines the interface for
these abstract types
"""

module AbstractTypes

# Will be adding methods to these functions
import Base: size, getindex, setindex!, length, iterate, eltype, isdone
import SparseArrays: issparse

import QGas.NumericalTools.ArrayDimensions as AD

export AbstractRule, AbstractRuleBuilder, AbstractMatrixWithRules
export build_rules!, error_check!, dim, set_defaults!, add_rule!, generate_builders!, generate_builders, get_array

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
"""
abstract type AbstractMatrixWithRules{T, M} <: AbstractMatrix{T} end
get_array(mwr::AbstractMatrixWithRules) = mwr.matrix
_get_rules(mwr::AbstractMatrixWithRules) = mwr.rules
_set_rules!(mwr::AbstractMatrixWithRules, rules) = mwr.rules = rules

#
# Implement array interface
#

size(mwr::AbstractMatrixWithRules) = size(get_array(mwr))
getindex(mwr::AbstractMatrixWithRules, args...) = getindex(get_array(mwr), args...)
setindex!(mwr::AbstractMatrixWithRules, args...) = setindex!(get_array(mwr), args...)
length(mwr::AbstractMatrixWithRules) = length(get_array(mwr))

iterate(mwr::AbstractMatrixWithRules, args...) = iterate(get_array(mwr), args...)
eltype(mwr::AbstractMatrixWithRules) = eltype(get_array(mwr))
isdone(mwr::AbstractMatrixWithRules, args...) = Base.isdone(get_array(mwr), args...)

issparse(mwr::AbstractMatrixWithRules) = issparse(get_array(mwr))

#
# New methods for AbstractMatrixWithRules
#

"""
    build_rules!(mwr::AbstractMatrixWithRules)

build the efficient build-tables based on the rules encoded in the matrix
"""
build_rules!(mwr::AbstractMatrixWithRules) = mwr

"""
    error_check!(mwr::AbstractMatrixWithRules)

check for errors
"""
error_check!(mwr::AbstractMatrixWithRules) = mwr

"""
    dim(mwr::MatrixWithRules)

Returns the vector space dimension, so internally matrices will be dim x dim in size
"""
dim(mwr::AbstractMatrixWithRules) = length(mwr.adims)

"""
    set_defaults!(mwd::AbstractMatrixWithRules, args...; kwargs...)

Set default parameters where the matrix will be evaluated.
"""
set_defaults!(mwr::AbstractMatrixWithRules; kwargs...) = mwr._kwargs_defaults = kwargs

"""
    `add_rule!(mwr::AbstractMatrixWithRules, rule::AbstractRule)`

Adds a rule of any type to `mwr`.  Methods exist that directly take the arguments for each rule type.
"""
add_rule!(mwr::AbstractMatrixWithRules, rule::AbstractRule) = push!(_get_rules(mwr), rule)

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

# Add new new functionality
add_matrix!(mwrs::AbstractMatrixWithRules, args...; kwargs...) = mwrs

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


abstract type AbstractMatrixSolver{T, M} <: AbstractMatrixWithRules{T, M} end
get_array(ms::AbstractMatrixSolver) = get_array(ms.mwr)

end