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
using QGas.NumericalTools.ArrayDimensions: Dimensions, index_to_coords, index_to_values
using ..Helpers: DimensionWithSpace, base_typeof
using ..Rules

using ..AbstractTypes
import ..AbstractTypes: build_rules!, build, build!, add_rule!, generate_builders!, generate_builders, add_leaf!, get_default_kwargs, set_default_kwargs! # To be overloaded

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

* `adims`: physical dimensions of the system
* `rules`: rules to be applied given relative coordinates
* `builders`: efficiently build matrices
* `matrix`: Most recent matrix
* `cache_kwargs::Bool`: whether to cache the kwargs of the rules
* `options`: Dictionary of options

Private fields (prefixed `_`) should only be touched by helpers.
"""
mutable struct MatrixWithRules{T, M<:AbstractMatrix{T}} <: AbstractMatrixWithRules{T, M}
    adims::Dimensions # Spatial / internal dimensions of system
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
isleaf(::MatricesWithRules) = Val{:leaf}

function MatrixWithRules(
        ::Type{M}, 
        adims::Dimensions,
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

function MatrixWithRules(::Type{M}, adims::Dimensions; kwargs...) where M
    
    # define the mapping from matrix index to (i,j,k, ...) coordinates
    index_to_coords = index_to_coords(Vector, adims)

    # define the mapping from linear array indices scaled values
    index_to_values = index_to_values(Vector, adims)

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
mutable struct MatricesWithRules{T,M<:AbstractMatrix{T}} <: AbstractMatricesWithRules{T, M}
    adims      :: Dimensions
    mwrs       :: Dict{Symbol,MatrixWithRules{T,M}}
    matrix     :: M
    cache_kwargs::Bool
    options    :: Dict{Symbol,Any}         # global options (rarely needed)

    # one set of maps is enough for the whole family
    _index_to_coords :: Vector{Vector{Int}}
    _index_to_values :: Vector{Vector{Float64}}
end
function MatricesWithRules(::Type{M},
                           adims::Dimensions,
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
isleaf(::MatricesWithRules) = Val{:node}

function MatricesWithRules(::Type{M}, adims::Dimensions; kwargs...) where M

    return MatricesWithRules(
        M,
        adims,
        index_to_coords(Vector, adims),
        index_to_values(Vector, adims);
        kwargs...
        )
end

# Add new new functionality
function add_leaf!(Val{:node}, mwrs::MatrixWithRules{T, M}, name::Symbol; kwargs...) where {T, M}
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
add_leaf!(Val{:node}, args...; kwargs...) = throw(ArgumentError("`add_leaf!` not valid for $(typeof(mwrs))"))

add_leaf!(mwrs::MatrixWithRules, name::Symbol; kwargs...) = add_leaf!(isleaf(mwrs), mwrs, name; kwargs...)

end # MatrixBuilders