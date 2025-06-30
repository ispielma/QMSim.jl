"""
    MatrixBuilders

A module for creating matrices using a mixture of rules or direct pre-computed matrices
"""

module MatrixBuilders
#
# In general I follow a naming convention for variables as follows:
#
# `mwd` - `MatrixWithDimensions`
# `mb` - `MatrixBuilder`
# `mwr` - `MatrixWithRules`
# and ...
#
# where the rule is that for these CamelCase types I am combining the first letter of each word.
#
using ..AbstractTypes
import ..AbstractTypes: build_rules!, build, build!, add_rule!, generate_builders!, generate_builders, add_matrix!, get_default_kwargs, set_default_kwargs! # To be overloaded

using ..Helpers: DimensionWithSpace, base_typeof

import QGas.NumericalTools.ArrayDimensions as AD


export DimensionWithSpace, RelativeRule, ElementRule, AbsoluteRule, ExplicitRule, MatrixWithRules, MatricesWithRules


#=
########  ##     ## ##       ########  ######
##     ## ##     ## ##       ##       ##    ##
##     ## ##     ## ##       ##       ##
########  ##     ## ##       ######    ######
##   ##   ##     ## ##       ##             ##
##    ##  ##     ## ##       ##       ##    ##
##     ##  #######  ######## ########  ######
=#


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
This precomputes all calls to index_to_coords and valid_coords, increasing
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

function RelativeRuleBuilder(rules::Vector{RelativeRule}, adims::AD.Dimensions, index_to_coords::Vector{Vector{Int}})

    rb = RelativeRuleBuilder([])

    if length(rules) != 0
        for (i, coords) in enumerate(index_to_coords), rule in rules
            new_coords = coords .+ rule.Δ

            if AD.valid_coords(adims, new_coords)
                j = AD.coords_to_index(adims, new_coords)
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
    actions::Vector{Tuple{Tuple{Int,Int}, RelativeRule}}
end
matrix_builder(::Type{ElementRule}, args...; kwargs...) = ElementRuleBuilder(args...; kwargs...)

function ElementRuleBuilder(rules::Vector{RelativeRule}, adims::AD.Dimensions, index_to_coords::Vector{Vector{Int}})

    rb = ElementRuleBuilder([])

    if length(rules) != 0
        for rule in rules

            if AD.valid_coords(adims, rule.x0) && AD.valid_coords(adims, rule.x1)
                i = AD.coords_to_index(adims, rule.x0)
                j = AD.coords_to_index(adims, rule.x1)
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
matrix_builder(::Type{AbsoluteRule}) = AbsoluteRuleBuilder(args...; kwargs...)

AbsoluteRuleBuilder(actions, args...; kwargs...) = AbsoluteRuleBuilder(actions)

function (rb::AbsoluteRuleBuilder)(mat, args...; kwargs...) 
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
        throw(ArgumentError("Unable to convert passed matrix type $(eltype(matrix)) to $(T) "))
    end

    mat = M(mat)

    ExplicitRule(mat)
end

(a::ExplicitRule)() = a.matrix

struct ExplicitRuleBuilder <: AbstractRuleBuilder
    actions::Vector{ExplicitRule}
end
matrix_builder(::Type{ExplicitRule}, args...; kwargs...) = ExplicitRuleBuilder(args...; kwargs...)

ExplicitRuleBuilder(actions, args...; kwargs...) = ExplicitRuleBuilder(actions)

function (rb::ExplicitRuleBuilder)(mat, args...; kwargs...)
    mat += mapreduce(r -> r.matrix, +, rb.actions)

    return mat
end

#=
##     ##    ###    ######## ########  #### ##     ##       ###    ##    ## ########     ########  ##     ## ##       ########  ######
###   ###   ## ##      ##    ##     ##  ##   ##   ##       ## ##   ###   ## ##     ##    ##     ## ##     ## ##       ##       ##    ##
#### ####  ##   ##     ##    ##     ##  ##    ## ##       ##   ##  ####  ## ##     ##    ##     ## ##     ## ##       ##       ##
## ### ## ##     ##    ##    ########   ##     ###       ##     ## ## ## ## ##     ##    ########  ##     ## ##       ######    ######
##     ## #########    ##    ##   ##    ##    ## ##      ######### ##  #### ##     ##    ##   ##   ##     ## ##       ##             ##
##     ## ##     ##    ##    ##    ##   ##   ##   ##     ##     ## ##   ### ##     ##    ##    ##  ##     ## ##       ##       ##    ##
##     ## ##     ##    ##    ##     ## #### ##     ##    ##     ## ##    ## ########     ##     ##  #######  ######## ########  ######
=#

"""
    MatrixWithRules

This a container type that contains rules used to build matrices.

rules: Each rule is a dictionary, and each rule specifies an individual matrix element.

* `adims`: physical dimensions of the system
* `rules`: rules to be applied given relative coordinates
* `builders`: efficiently build matrices
* `matrix`: Most recent matrix
* `cache_kwargs::Bool`: whether to cache the kwargs of the rules
* `options`: Dictionary of options

Private fields (prefixed `_`) should only be touched by helpers.
"""
mutable struct MatrixWithRules{T, M<:AbstractMatrix{T}} <: AbstractMatrixWithRules{T, M}
    adims::AD.Dimensions # Spatial / internal dimensions of system
    rules::Vector{AbstractRule}
    builders::Vector{AbstractRuleBuilder}
    matrix::M
    cache_kwargs::Bool
    options::Dict{Symbol, Any}

    _index_to_coords::Vector{Vector{Int}}
    _index_to_values::Vector{Vector{Float64}}
    _kwargs::Dict{Symbol, Any} # Most recent kwargs
    _default_kwargs::Dict{Symbol, Any} # Default kwargs
    _matrix_cached::Bool # whether the matrix is currently cached
end
function MatrixWithRules(
        ::Type{M}, 
        adims::AD.Dimensions,
        index_to_coords,
        index_to_values; 
        cache_kwargs::Bool=true, 
        options=Dict{Symbol, Any}()
    ) where M

    MatrixWithRules{eltype(M), M}(
        adims,
        AbstractRule[],
        AbstractRuleBuilder[],
        zeros(M, eltype(M), length(adims), length(adims) ), 
        cache_kwargs, 
        options,
        index_to_coords,
        index_to_values,
        Dict(),
        Dict(),
        false
    )
end

function MatrixWithRules(::Type{M}, adims::AD.Dimensions; kwargs...) where M
    
    # define the mapping from matrix index to (i,j,k, ...) coordinates
    index_to_coords = AD.index_to_coords(Vector, adims)

    # define the mapping from linear array indices scaled values
    index_to_values = AD.index_to_values(Vector, adims)

    MatrixWithRules(M, adims, index_to_coords, index_to_values; kwargs...)
end

"""
    generate_builders(mwr::MatrixWithRules)::Vector{AbstractRuleBuilder}()
    generate_builders!(mwr::MatrixWithRules)::MatrixWithRules

Methods to construct MatrixBuilder that directly uses the MatrixWithRules
"""
function generate_builders(mwr::MatrixWithRules)

    # Get all rule types
    rule_types = Set(base_typeof(rule) for rule in mwr.rules)

    builders =  Vector{AbstractRuleBuilder}()

    for T in rule_types

        # I do need to declare the type here to make sure that we don't
        # get the specialized type in the case of a length 1 vector.
        rules::Vector{T} = [rule for rule in mwr.rules if rule isa T]

        push!(builders, matrix_builder(T, rules, mwr.adims, mwr._index_to_coords))
    end 

    return builders
end
function generate_builders!(mwr::MatrixWithRules)
    mwr.builders = generate_builders(mwr)
    return mwr
end

"""
    build(mwr::MatrixWithRules; kwargs...)

* `kwargs`: kwargs to be passed to rules 
"""
function build(mwr::MatrixWithRules{T, M}; kwargs...) where {T, M}

    # First combine provided kwargs with the defaults as backups
    default_kwargs = get_default_kwargs(mwr)
    kwargs = merge(default_kwargs, kwargs)

    # Now check if we can use the cached matrix
    use_cached = mwr.cache_kwargs && mwr._matrix_cached

    if use_cached
        for (k, cached_kwarg) in mwr._kwargs

            if !(k in keys(default_kwargs))
                throw(ArgumentError("Default does not exist for key $(k)."))
            end

            kwarg = get(kwargs, k, default_kwargs[k])
                   
            if kwarg != cached_kwarg
                mwr._kwargs[k] = copy(kwarg)
                use_cached = false
            end
        end
    end

    if !use_cached
        mat = zeros(M, T, dim(mwr), dim(mwr) ) 

        for builder in mwr.builders

            mat = builder(mat, mwr._index_to_values; kwargs...)
        end

        mwr._kwargs = kwargs
    else
        mat = mwr.matrix
    end        

    return mat
end
function build!(mwr::MatrixWithRules; kwargs...)
    mwr.matrix = build(mwr; kwargs...)

    mwr._matrix_cached = true

    return mwr
end

"""
    MatricesWithRules{T,M<:AbstractMatrix{T}} <: AbstractMatrixWithRules

A collection of MatrixWithRules each labeled by a symbol that share the same coordinate system.
"""
mutable struct MatricesWithRules{T,M<:AbstractMatrix{T}} <: AbstractMatricesWithRules{T, M}
    adims      :: AD.Dimensions
    matrices   :: Dict{Symbol,MatrixWithRules{T,M}}
    matrix     :: M
    cache_kwargs::Bool
    options    :: Dict{Symbol,Any}         # global options (rarely needed)

    # one set of maps is enough for the whole family
    _index_to_coords :: Vector{Vector{Int}}
    _index_to_values :: Vector{Vector{Float64}}
end
function MatricesWithRules(::Type{M},
                           adims::AD.Dimensions,
                           index_to_coords,
                           index_to_values;
                           cache_kwargs::Bool = true,
                           options = Dict{Symbol,Any}()) where M

    return MatricesWithRules{eltype(M),M}(
        adims,
        Dict{Symbol,MatrixWithRules{eltype(M),M}}(),
        zeros(M, eltype(M), length(adims), length(adims) ),
        cache_kwargs,
        options,
        index_to_coords,
        index_to_values
        )
end
function MatricesWithRules(::Type{M}, adims::AD.Dimensions; kwargs...) where M

    return MatricesWithRules(
        M,
        adims,
        AD.index_to_coords(Vector, adims),
        AD.index_to_values(Vector, adims);
        kwargs...
        )
end

# Add new new functionality
function add_matrix!(mwrs::AbstractMatricesWithRules{T, M}, name::Symbol; kwargs...) where {T, M}
    matrix = MatrixWithRules(
        M, 
        get_dimensions(mwrs),
        mwrs._index_to_coords,  
        mwrs._index_to_values; 
        cache_kwargs=mwrs.cache_kwargs, 
        options=mwrs.options, 
        kwargs...
    )

    return set_matrix!(mwrs, name, matrix)
end

end # MatrixBuilders