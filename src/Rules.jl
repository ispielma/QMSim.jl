"""
    Rules

This module defines the abstract type AbstractRule and the associated concrete types.
"""
module Rules

using QGas.NumericalTools.ArrayDimensions: Dimensions, valid_coords, coords_to_index  

export AbstractRule, AbstractRuleBuilder, RelativeRule, ElementRule, AbsoluteRule, ExplicitRule
export matrix_builder

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
        adims::Dimensions,
        index_to_coords::Vector{Vector{Int}})
    ```
note that args... and kwargs... can be used in the constructors to ignore arguments that are not needed.
"""
abstract type AbstractRule end
abstract type AbstractRuleBuilder end

# Introduce traits that are needed to connect the rule type with the associated matrix builder 
matrix_builder(::Type{T}) where {T<:AbstractRule} = error("No builder defined for rule‐type $T")

(rb::AbstractRuleBuilder)(mat, index_to_values; kwargs...) = mat

#=
########  ######## ##          ###    ######## #### ##     ## ########
##     ## ##       ##         ## ##      ##     ##  ##     ## ##
##     ## ##       ##        ##   ##     ##     ##  ##     ## ##
########  ######   ##       ##     ##    ##     ##  ##     ## ######
##   ##   ##       ##       #########    ##     ##   ##   ##  ##
##    ##  ##       ##       ##     ##    ##     ##    ## ##   ##
##     ## ######## ######## ##     ##    ##    ####    ###    ########
=#
"""
    `RelativeRule{F} <: AbstractRule`
    `RelativeRuleBuilder <: AbstractRuleBuilder`
    
A rule and binder for functions of the displacement coordinate (currently fixed as an integer, but this need not be the case).

    `RelativeRuleBuilder`

Encodes the relative rules that are to be run each time a full matrix is assembled.  
This pre-computes all calls to index_to_coords and valid_coords, increasing
performance, at the expense of memory.
"""
struct RelativeRule{F} <: AbstractRule
    func::F
    Δ::Vector{Int}
end
RelativeRule(::Type{M}, func, Δ) where M = RelativeRule(func, Δ)

(a::RelativeRule)(x0::Vector, x1::Vector; kwargs...) = a.func(x0, x1; kwargs...)

struct RelativeRuleBuilder <: AbstractRuleBuilder
    # the actual matrix elements i, j, being written to
    actions::Vector{Tuple{Tuple{Int,Int}, RelativeRule}}
end
matrix_builder(::Type{RelativeRule}, args...; kwargs...) = RelativeRuleBuilder(args...; kwargs...)

function RelativeRuleBuilder(rules::Vector{RelativeRule}, adims::Dimensions, index_to_coords::Vector{Vector{Int}})

    rb = RelativeRuleBuilder([])

    if length(rules) != 0
        for (i, coords) in enumerate(index_to_coords), rule in rules
            new_coords = coords .+ rule.Δ

            if valid_coords(adims, new_coords)
                j = coords_to_index(adims, new_coords)
                push!(rb.actions, ((i, j), rule))
            end
        end
    end
    
    return rb
end

function (rb::RelativeRuleBuilder)(mat, index_to_values; kwargs...)

    for ((i,j), rule) in rb.actions
        mat[i,j] += rule(index_to_values[i], index_to_values[j]; kwargs...)
    end
    
    return mat
end

#=
######## ##       ######## ##     ## ######## ##    ## ########
##       ##       ##       ###   ### ##       ###   ##    ##
##       ##       ##       #### #### ##       ####  ##    ##
######   ##       ######   ## ### ## ######   ## ## ##    ##
##       ##       ##       ##     ## ##       ##  ####    ##
##       ##       ##       ##     ## ##       ##   ###    ##
######## ######## ######## ##     ## ######## ##    ##    ##
=#
"""
    `ElementRule{F} <: AbstractRule`
    `ElementRuleBuilder <: AbstractRuleBuilder`
    
A rule and binder for functions of actual matrix elements.

    `ElementRuleBuilder`

Encodes the element rules that are to be run each time a full matrix is assembled.  
This precomputes all calls to index_to_coords and valid_coords, increasing
performance, at the expense of memory.
"""

struct ElementRule{F} <: AbstractRule
    func::F
    x0::Vector{Int}
    x1::Vector{Int}
end
ElementRule(::Type{M}, func, x0, x1) where M = ElementRule(func, x0, x1)

(a::ElementRule)(x0::Vector, x1::Vector; kwargs...) = a.func(x0, x1; kwargs...)

struct ElementRuleBuilder <: AbstractRuleBuilder
    # the actual matrix elements i, j, being written to
    actions::Vector{Tuple{Tuple{Int,Int}, ElementRule}}
end
matrix_builder(::Type{ElementRule}, args...; kwargs...) = ElementRuleBuilder(args...; kwargs...)

function ElementRuleBuilder(rules::Vector{ElementRule}, adims::Dimensions, ::Vector{Vector{Int}})

    rb = ElementRuleBuilder([])

    if length(rules) != 0
        for rule in rules

            if valid_coords(adims, rule.x0) && valid_coords(adims, rule.x1)
                i = coords_to_index(adims, rule.x0)
                j = coords_to_index(adims, rule.x1)
                push!(rb.actions, ((i, j), rule))
            end
        end
    end
    
    return rb
end

# This is actually the same as the RelativeRuleBuilder since we iterate over the from-to rules in the end of the day
function (rb::ElementRuleBuilder)(mat, index_to_values; kwargs...)

    for ((i,j), rule) in rb.actions
        mat[i,j] += rule(index_to_values[i], index_to_values[j]; kwargs...)
    end
    
    return mat
end

#=
   ###    ########   ######   #######  ##       ##     ## ######## ########
  ## ##   ##     ## ##    ## ##     ## ##       ##     ##    ##    ##
 ##   ##  ##     ## ##       ##     ## ##       ##     ##    ##    ##
##     ## ########   ######  ##     ## ##       ##     ##    ##    ######
######### ##     ##       ## ##     ## ##       ##     ##    ##    ##
##     ## ##     ## ##    ## ##     ## ##       ##     ##    ##    ##
##     ## ########   ######   #######  ########  #######     ##    ########
=#

"""
    AbsoluteRule{F} <: AbstractRule
    AbsoluteRuleBuilder <: AbstractRuleBuilder

A rule and builder for functions that return a complete matrix
"""
struct AbsoluteRule{F} <: AbstractRule
    func::F
end
AbsoluteRule(::Type{M}, func) where M = AbsoluteRule(func)
(a::AbsoluteRule)(kwargs...) = a.func(;kwargs...)

struct AbsoluteRuleBuilder <: AbstractRuleBuilder
    actions::Vector{AbsoluteRule}
end
matrix_builder(::Type{AbsoluteRule}, args...; kwargs...) = AbsoluteRuleBuilder(args...; kwargs...)

AbsoluteRuleBuilder(rules::Vector{AbsoluteRule}, ::Dimensions, ::Vector{Vector{Int}}) = AbsoluteRuleBuilder(rules)

function (rb::AbsoluteRuleBuilder)(mat, index_to_values; kwargs...) 
    mat += mapreduce(r -> r.func(;kwargs...), +, rb.actions)
    return mat
end

#=
######## ##     ## ########  ##       ####  ######  #### ########
##        ##   ##  ##     ## ##        ##  ##    ##  ##     ##
##         ## ##   ##     ## ##        ##  ##        ##     ##
######      ###    ########  ##        ##  ##        ##     ##
##         ## ##   ##        ##        ##  ##        ##     ##
##        ##   ##  ##        ##        ##  ##    ##  ##     ##
######## ##     ## ##        ######## ####  ######  ####    ##
=#

"""
    ExplicitRule{M<:AbstractMatrix} <: AbstractRule
    ExplicitRuleBuilder <: AbstractRuleBuilder

A rule and builder for for hard-coded matrices
"""
struct ExplicitRule{M<:AbstractMatrix} <: AbstractRule
    matrix::M
end
function ExplicitRule(::Type{M}, mat) where M

    try
        mat = convert(M, mat)
    catch
        throw(ArgumentError("Unable to convert passed matrix type $(eltype(mat)) to $(M) "))
    end

    ExplicitRule(mat)
end

(a::ExplicitRule)() = a.matrix

struct ExplicitRuleBuilder <: AbstractRuleBuilder
    actions::Vector{ExplicitRule}
end
matrix_builder(::Type{ExplicitRule}, args...; kwargs...) = ExplicitRuleBuilder(args...; kwargs...)

ExplicitRuleBuilder(rules::Vector{ExplicitRule}, ::Dimensions, ::Vector{Vector{Int}}) = ExplicitRuleBuilder(rules)

function (rb::ExplicitRuleBuilder)(mat, index_to_values; kwargs...)
    mat += mapreduce(r -> r.matrix, +, rb.actions)

    return mat
end

end # Rules