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
using QGas.NumericalTools.ArrayDimensions: Dimensions
using ..Helpers: MatrixSharedData
using ..Helpers: base_typeof, get_index_to_coords, get_index_to_values, get_options, set_options!, get_cache_kwargs, set_cache_kwargs!

using ..Rules
using ..AbstractMatrixTypes

# To be overloaded for new types
import ..AbstractMatrixTypes: isleaf
import ..AbstractMatrixTypes: dim, get_default_kwargs, set_default_kwargs!, get_rules, set_rules!, add_rule!
import ..AbstractMatrixTypes: build_rules!, error_check, build, build!, generate_builders!, generate_builders

export MatrixWithRules, MatricesWithRules

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

* `shared`: shared properties
* `rules`: rules to be applied given relative coordinates
* `builders`: efficiently build matrices
* `matrix`: Most recent matrix
* `cache_kwargs::Bool`: whether to cache the kwargs of the rules
* `options`: Dictionary of options

Private fields (prefixed `_`) should only be touched by helpers.
"""
mutable struct MatrixWithRules{T, M<:AbstractMatrix{T}} <: AbstractMatrixWithRules{T, M}
    shared::MatrixSharedData
    rules::Vector{AbstractRule}
    builders::Vector{AbstractRuleBuilder}
    matrix::M

    _kwargs::Dict{Symbol, Any} # Most recent kwargs
    _default_kwargs::Dict{Symbol, Any} # Default kwargs
    _matrix_cached::Bool # whether the matrix is currently cached
end
isleaf(::Type{<:MatrixWithRules}) = LeafTrait()

function MatrixWithRules(
        ::Type{M}, 
        shared::MatrixSharedData
    ) where M

    MatrixWithRules{eltype(M), M}(
        shared,
        AbstractRule[],
        AbstractRuleBuilder[],
        zeros(M, eltype(M), length(shared), length(shared) ), 
        Dict(),
        Dict(),
        false
    )
end

MatrixWithRules(::Type{M}, adims::Dimensions; options=Dict{Symbol,Any}(), cache_kwargs=true) where M = MatrixWithRules(M,  MatrixSharedData(adims, options, cache_kwargs))

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

        push!(builders, matrix_builder(T, rules, get_dimensions(mwr), get_index_to_coords(mwr) ) )
    end 

    return builders
end
function generate_builders!(mwr::MatrixWithRules)
    mwr.builders = generate_builders(mwr)
    return mwr
end

"""
    build(mwr::MatrixWithRules; kwargs...)

* `kwargs`: kwargs to be passed to rules.  

We will only accept kwargs that are also in get_default_kwargs(mwr).
"""
function build(mwr::MatrixWithRules{T, M}; kwargs...) where {T, M}

    # First get default kwargs and replace with any elements provided by kwargs
    default_kwargs = get_default_kwargs(mwr)
    kwargs = Dict(k => get(kwargs, k, v) for (k, v) in default_kwargs)

    # Now check if we can use the cached matrix
    use_cached = get_cache_kwargs(mwr) && mwr._matrix_cached

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

            mat = builder(mat, get_index_to_values(mwr); kwargs...)
        end

        mwr._kwargs = kwargs
    else
        mat = get_matrix(mwr)
    end        

    return mat
end
function build!(mwr::MatrixWithRules; kwargs...)
    set_matrix!(mwr, build(mwr; kwargs...))

    mwr._matrix_cached = true

    return mwr
end

"""
    MatricesWithRules{T,M<:AbstractMatrix{T}} <: AbstractMatrixWithRules

A collection of MatrixWithRules each labeled by a symbol that share the same coordinate system.
"""
mutable struct MatricesWithRules{T,M<:AbstractMatrix{T}} <: AbstractMatrixWithRules{T, M}
    shared     :: MatrixSharedData
    mwrs       :: Dict{Symbol,MatrixWithRules{T,M}}
    matrix     :: M
end
isleaf(::Type{<:MatricesWithRules}) = NodeTrait()

function MatricesWithRules(::Type{M}, shared::MatrixSharedData) where M

    return MatricesWithRules{eltype(M),M}(
        shared,
        Dict{Symbol,MatrixWithRules{eltype(M),M}}(),
        zeros(M, eltype(M), length(shared), length(shared) )
        )
end

MatricesWithRules(::Type{M}, adims::Dimensions; options=Dict{Symbol,Any}(), cache_kwargs=true) where M = MatricesWithRules(M, MatrixSharedData(adims, options, cache_kwargs))
MatrixWithRules(mwrs::MatricesWithRules{T, M}) where {T, M} = MatrixWithRules(M, mwrs.shared)

end # MatrixBuilders