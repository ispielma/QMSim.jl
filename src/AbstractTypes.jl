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

export AbstractMatrixWithRules, AbstractMatrixSolver

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

Abstract MatrixWithRules type, this can be organized in a tree with leafs containing rules and nodes contain other nodes or
leafs, but not rules.

and the default interface assumes the fields:

* `adims` :: AD.Dimensions
* `matrix` :: AbstractMatrix

only for leafs
* `rules` :: Vector{AbstractRule}
* `builders` :: Vector{AbstractRuleBuilder}
* `_default_kwargs` :: Dict{Symbol, Any}

only for nodes
* `mwrs` :: Dict{Symbol,AbstractMatrixWithRules{T,M}}
"""
abstract type AbstractMatrixWithRules{T, M} <: AbstractMatrix{T} end

#
# Assign traits 
#

isleaf(::AbstractMatrixWithRules) = Val(:undefined) # Val{:leaf} or Val{:node} for concrete types

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
# Define interface for all
#

get_matrix(mwr::AbstractMatrixWithRules) = mwr.matrix

set_matrix!(mwr::AbstractMatrixWithRules, matrix::AbstractMatrix) = (mwr.matrix = matrix; mwr)

get_dimensions(mwr::AbstractMatrixWithRules) = mwr.adims

set_dimensions!(mwr::AbstractMatrixWithRules, adims) = (mwr.adims = adims; mwr)


#
# defined only for leafs
#

get_default_kwargs(::Val{:leaf}, mwr::AbstractMatrixWithRules) = mwr._default_kwargs
get_default_kwargs(::Val, mwr::AbstractMatrixWithRules) = throw(ArgumentError("`get_default_kwargs` is not defined for $(typeof(mwr))"))
get_default_kwargs(mwr::AbstractMatrixWithRules) = get_default_kwargs(isleaf(mwr), mwr)

set_default_kwargs!(::Val{:leaf}, mwr::AbstractMatrixWithRules, kwargs) = (mwr._default_kwargs = kwargs; mwr)
set_default_kwargs!(::Val, mwr::AbstractMatrixWithRules, kwargs) = throw(ArgumentError("`set_default_kwargs!` is not defined for $(typeof(mwr))"))
set_default_kwargs!(mwr::AbstractMatrixWithRules, kwargs) = set_default_kwargs!(isleaf(mwr), mwr, kwargs)
set_default_kwargs!(mwr::AbstractMatrixWithRules; kwargs...) = set_default_kwargs!(mwr, Dict(kwargs))

get_rules(::Val{:leaf}, mwr::AbstractMatrixWithRules) = mwr.rules
get_rules(::Val, mwr::AbstractMatrixWithRules) = throw(ArgumentError("`get_rules` is not defined for $(typeof(mwr))"))
get_rules(mwr::AbstractMatrixWithRules) = get_rules(isleaf(mwr), mwr) 

set_rules!(::Val{:leaf}, mwr::AbstractMatrixWithRules, rules) = (mwr.rules = rules; mwr)
set_rules!(::Val, mwr::AbstractMatrixWithRules, rules) = throw(ArgumentError("`set_rules!` is not defined for $(typeof(mwr))"))
set_rules!(mwr::AbstractMatrixWithRules, rules) = set_rules!(isleaf(mwr), mwr, rules) 

#
# methods defined only by nodes 
#

get_leafs(::Val{:node}, mwr::AbstractMatrixWithRules) = mwr.mwrs
get_leafs(::Val, mwr::AbstractMatrixWithRules) = throw(ArgumentError("`get_leafs` is not defined for $(typeof(mwr))"))
get_leafs(mwr::AbstractMatrixWithRules) = get_leafs(isleaf(mwr), mwr)

# TODO: should this be implemented here for the abstract type?
add_leaf!(::Val{:node}, mwr::AbstractMatrixWithRules, name::Symbol, args...; kwargs...) = error("add_leaf! must be implemented for type $(typeof(mwr))")
add_leaf!(::Val, mwr::AbstractMatrixWithRules, name::Symbol, args...; kwargs...) = throw(ArgumentError("`add_leaf!` is not defined for $(typeof(mwr))"))

add_leaf!(mwr::AbstractMatrixWithRules, name::Symbol, args...; kwargs...) = add_leaf!(isleaf(mwr), mwr, name, args...; kwargs...)

get_matrix(::Val{:node}, mwr::AbstractMatrixWithRules, name::Symbol) = get_matrix(get_leafs(mwr)[name])
get_matrix(::Val, mwr::AbstractMatrixWithRules, name::Symbol) = throw(ArgumentError("`get_matrix` accepts no `name` field for $(typeof(mwr))"))
get_matrix(mwr::AbstractMatrixWithRules, name::Symbol) = get_matrix(isleaf(mwr), mwr, name)

set_matrix!(::Val{:node}, mwr::AbstractMatrixWithRules, name::Symbol, matrix) = set_matrix!(get_matrix(mwr, name), matrix)
set_matrix!(::Val, mwr::AbstractMatrixWithRules, name::Symbol, matrix) = throw(ArgumentError("`set_matrix!` accepts no `name` field for $(typeof(mwr))"))
set_matrix!(mwr::AbstractMatrixWithRules, name::Symbol, matrix) = set_matrix!(isleaf(mwr), mwr, name, matrix)

get_default_kwargs(::Val{:node}, mwr::AbstractMatrixWithRules, name::Symbol) = get_default_kwargs(get_leafs(mwr)[name])
get_default_kwargs(::Val, mwr::AbstractMatrixWithRules, name::Symbol) = throw(ArgumentError("`get_default_kwargs` accepts no `name` field for $(typeof(mwr))"))
get_default_kwargs(mwr::AbstractMatrixWithRules, name::Symbol) = get_default_kwargs(isleaf(mwr), mwr, name)

set_default_kwargs!(::Val{:node}, mwr::AbstractMatrixWithRules, name::Symbol, kwargs) = set_default_kwargs!(get_leafs(mwr)[name], kwargs)
set_default_kwargs!(::Val, mwr::AbstractMatrixWithRules, name::Symbol, kwargs) = throw(ArgumentError("`set_default_kwargs!` accepts no `name` field for $(typeof(mwr))"))
set_default_kwargs!(mwr::AbstractMatrixWithRules, name::Symbol, kwargs) = set_default_kwargs!(isleaf(mwr), mwr, name, kwargs)
set_default_kwargs!(mwr::AbstractMatrixWithRules, name::Symbol; kwargs) = set_default_kwargs!(mwr, name, Dict(kwargs))

#
# Methods for AbstractMatrixWithRules that perform non-trivial actions either here or when implemented
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

error_check(::Val{:undefined}, mwr::AbstractMatrixWithRules) = throw(ArgumentError("`isleaf` not overridden for $(typeof(mwr))"))
error_check(::Val{:leaf}, mwr::AbstractMatrixWithRules) = nothing
error_check(::Val{:node}, mwr::AbstractMatrixWithRules) = foreach(error_check, values(get_leafs(mwr)))

function error_check(mwr::AbstractMatrixWithRules) 
    
    len_mwr = length(mwr)
    len_dims = length(get_dimensions(mwr))
    if len_mwr != len_dims
        throw(DimensionMismatch("Length of stored matrix $(len_mwr) does not match length of dimensions $(len_dims)"))
    end

    # now do a trait-by-trait check
    error_check(isleaf(mwr), mwr) 
end

"""
    `add_rule!(mwr::AbstractMatrixWithRules, rule::AbstractRule)`

Adds a rule of any type to `mwr`.  Methods exist that directly take the arguments for each rule type.
"""
add_rule!(::Val{:leaf}, mwr::AbstractMatrixWithRules, rule::AbstractRule) = (push!(get_rules(mwr), rule); mwr)
add_rule!(::Val, mwr::AbstractMatrixWithRules, rule::AbstractRule) = throw(ArgumentError("`add_rule!`: invalid call signature for $(typeof(mwr))"))

add_rule!(mwr::AbstractMatrixWithRules, rule::AbstractRule) = add_rule!(isleaf(mwr), mwr, rule)

"""
    `add_rule!(mwr::MatrixWithRules, RuleType::Type{R <: AbstractRule}, args...)

Adds a rule of any type to `mwr`.  Dispatch to the type provided by RuleType.
"""
add_rule!(mwr::AbstractMatrixWithRules{T, M}, RuleType::Type{R}, args...) where {T, M, R <:AbstractRule} = add_rule!(mwr, RuleType(M, args...))

"""
    `add_rule!(mwr::MatrixWithRules, name::Symbol, args...; kwargs...)

Dispatches add rule to a leaf.
"""
add_rule!(::Val{:node}, mwr::AbstractMatrixWithRules, name::Symbol, args...; kwargs...) = add_rule!(get_leafs(mwr)[name], args...; kwargs...)
add_rule!(::Val, mwr::AbstractMatrixWithRules, name::Symbol, args...; kwargs...) = throw(ArgumentError("`add_rule!`: name field not supported for $(typeof(mwr))"))

add_rule!(mwr::AbstractMatrixWithRules, name::Symbol, args...; kwargs...) = add_rule!(isleaf(mwr), mwr, name, args...; kwargs...)


"""
    generate_builders(mwr::AbstractMatrixWithRules)::Vector{AbstractRuleBuilder}()
    generate_builders!(mwr::AbstractMatrixWithRules)::AbstractMatrixWithRules

Interface for generating builders
"""
generate_builders(::Val{:leaf}, mwr::AbstractMatrixWithRules) = Vector{AbstractRuleBuilder}()

function generate_builders!(::Val{:node}, mwr::AbstractMatrixWithRules)
    for (_, leaf) in get_leafs(mwr)
        generate_builders!(leaf)
    end
    return mwr
end

generate_builders!(::Val, mwr::AbstractMatrixWithRules) = throw(ArgumentError("`generate_builders!`: not supported for $(typeof(mwr))"))

generate_builders!(mwr::AbstractMatrixWithRules) = generate_builders!(isleaf(mwr), mwr)

"""
    build(mwr::AbstractMatrixWithRules)
    build!(mwr::AbstractMatrixWithRules; names=nothing, kwargs...)

Interface for generating builders.

    * `names` :: Vector{Symbol} : only valid for node types where it optionally specifies which of the matrices to build.
"""

# Based on the pattern above, I should actually move the now generic code that performs the building for leafs to here 
# from the concrete type definitions

build(::Val{:leaf}, mwr::AbstractMatrixWithRules, args...; kwargs...) = get_matrix(mwr)


function build(::Val{:node}, mwr::AbstractMatrixWithRules; names=nothing, kwargs...)

    leafs = get_leafs(mwr)

    if names === nothing
        names = keys(leafs)
    end

    for name in names
        leaf = leafs[name]
        build!(leaf; kwargs...)
    end

    matrix = mapreduce(mwr -> get_matrix(mwr), +, values(leafs))
    set_matrix!(mwr, matrix)
end
build!(::Val, mwr::AbstractMatrixWithRules) = throw(ArgumentError("`build!`: not supported for $(typeof(mwr))"))

build(mwr::AbstractMatrixWithRules, args...; kwargs...) = build(isleaf(mwr), mwr, args...; kwargs...)

build!(mwr::AbstractMatrixWithRules, args...; kwargs...) = set_matrix!(mwr, build(mwr, args...; kwargs...))

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

Encodes information required to actually obtain solutions to the matrix problem.  The standard interface
only assumes the existence of the field
    * mwr :: AbstractMatricesWithRules 
this can have either the Val(:leaf) or Val(:node) flag
"""

abstract type AbstractMatrixSolver{T, M} <: AbstractMatrixWithRules{T, M} end
isleaf(ams::AbstractMatrixSolver) = isleaf(ams.mwr)

get_matrix(ms::AbstractMatrixSolver, args...; kwargs...) = get_matrix(ms.mwr, args...; kwargs...)

end