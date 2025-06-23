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
using LinearAlgebra, SparseArrays
import QGas.NumericalTools.ArrayDimensions as AD

import Base: zeros

export DimensionWithSpace, RelativeRule, AbsoluteRule, ExplicitRule
export MatrixWithRules, add_rule!, build_matrix!
export MatrixWithDimensions
export finalize!


# Zeros methods for currently supported matrix types.
zeros(::Type{Matrix}, ::Type{T}, args...) where T = zeros(T, args...)
zeros(::Type{Matrix{T}}, ::Type{T}, args...) where T = zeros(T, args...)

zeros(::Type{SparseMatrixCSC}, ::Type{T}, args...) where T = spzeros(T, args...)
zeros(::Type{SparseMatrixCSC{T}}, ::Type{T}, args...) where T = spzeros(T, args...)

"""
    `DimensionWithSpace`

A concrete type of `AbstractDimension` that has a `spatial` flag.
"""
Base.@kwdef mutable struct DimensionWithSpace <: AD.AbstractDimension
    x0::Float64 = 0.0
    dx::Float64 = 1.0
    npnts::Int64 = 0
    unit::String = ""
    symmetric::Bool = false
    periodic::Bool = false
    spatial::Bool = true
end

#=
########  ##     ## ##       ########    ######## ##    ## ########  ########  ######
##     ## ##     ## ##       ##             ##     ##  ##  ##     ## ##       ##    ##
##     ## ##     ## ##       ##             ##      ####   ##     ## ##       ##
########  ##     ## ##       ######         ##       ##    ########  ######    ######
##   ##   ##     ## ##       ##             ##       ##    ##        ##             ##
##    ##  ##     ## ##       ##             ##       ##    ##        ##       ##    ##
##     ##  #######  ######## ########       ##       ##    ##        ########  ######
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
This precomputes all calls to index_to_coords and valid_coords, increasing
performance, at the expense of memory.
"""
struct RelativeRule{F} <: AbstractRule
    func::F
    Δ::Vector{Int}
end
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

"""
   
"""
function (rb::RelativeRuleBuilder)(mat, index_to_values; kwargs...)

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
(a::AbsoluteRule)(kwargs...) = a.func(;kwargs...)

struct AbsoluteRuleBuilder <: AbstractRuleBuilder
    actions::Vector{AbsoluteRule}
end
matrix_builder(::Type{AbsoluteRule}) = AbsoluteRuleBuilder(args...; kwargs...)

AbsoluteRuleBuilder(rules, args...; kwargs...) = AbsoluteRuleBuilder(rules)

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
(a::ExplicitRule)() = a.matrix

struct ExplicitRuleBuilder <: AbstractRuleBuilder
    actions::Vector{ExplicitRule}
end
matrix_builder(::Type{ExplicitRule}) = ExplicitRuleBuilder(args...; kwargs...)

ExplicitRuleBuilder(rules, args...; kwargs...) = ExplicitRuleBuilder(rules)

function (rb::ExplicitRuleBuilder)(mat, args...; kwargs...) 
    mat += reduce(+, rb.actions)
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

* `rules`: rules to be applied given relative coordinates.  
* `cache_kwargs::Bool`: whether to cache the kwargs of the rules
* `options`: Dictionary of options
* `matrix`: Most recent matrix

Private fields (prefixed `_`) should only be touched by helpers.
"""

mutable struct MatrixWithRules{T, M<:AbstractMatrix{T}}
    adims::AD.Dimensions # Spatial / internal dimensions of system
    rules::Vector{AbstractRule}
    builders::Vector{AbstractRuleBuilder}
    matrix::M
    cache_kwargs::Bool
    options::Dict{Symbol, Any}

    _index_to_coords::Vector{Vector{Int}}
    _index_to_values::Vector{Vector{Float64}}
    _kwargs::Dict{Symbol, Any} # Most recent kwargs
    _kwargs_defaults::Dict{Symbol, Any} # Default kwargs
    _matrix_cached::Bool # whether the matrix is currently cached
end
function MatrixWithRules(M::Type, adims::AD.Dimensions; cache_kwargs=true, option=Dict())
    
    # define the mapping from matrix index to (i,j,k, ...) coordinates
    index_to_coords = AD.index_to_coords(Vector, adims)

    # define the mapping from linear array indices scaled values
    index_to_values = AD.index_to_values(Vector, adims)

    MatrixWithRules{eltype(M), M}(
        adims,
        [],
        [],
        M(undef, 0, 0), 
        cache_kwargs, 
        options,
        index_to_coords,
        index_to_values,
        Dict(),
        Dict(),
        false
    )
end

"""
    dim(mwr::MatrixWithRules)

Returns the vector space dimension, so internally matrices will be dim x dim in size
"""
dim(mwr::MatrixWithDimensions) = length(mwr.adims)

"""
    `add_rule!(mwr::MatrixWithRules, rule::AbstractRule)`

Adds a rule of any type to `mwr`.  Methods exist that directly take the arguments for each rule type.
"""
add_rule!(mwr::MatrixWithRules, rule::AbstractRule) = mwr.rules = vcat(mwr.rules, rule)

"""
    `add_rule!(mwr::MatrixWithRules, f::Function Δ::AbstractVector{Int})`

This method assumes that we are adding an `RelativeRule` type which when called returns a matrix element for a specific transition.
"""
add_rule!(mwr::MatrixWithRules, f::Function, Δ::AbstractVector{Int}) = add_rule!(mwr, RelativeRule(f, Δ))

"""
    `add_rule!(mwr::MatrixWithRules, f::Function)`

This method assumes that we are adding an `AbsoluteRule` type which when called returns a matrix
"""
add_rule!(mwr::MatrixWithRules, f::Function) = add_rule!(mwr, AbsoluteRule(f))

"""
    `add_rule!(mwr::MatrixWithRules{T,M}, matrix::AbstractMatrix) where {T,M}`

This method assumes that we are adding an `ExplicitRule` type which when called returns a matrix
"""
function add_rule!(mwr::MatrixWithRules{T, M}, matrix::AbstractMatrix) where {T, M}

    try
        matrix = convert(M, matrix)
    catch
        throw(ArgumentError("Unable to convert passed matrix type $(eltype(matrix)) to $(T) "))
    end

    matrix = M(matrix)

    add_rule!(mwr, ExplicitRule(matrix))
end

"""
    generate_builders(mwr::MatrixWithRules)::MatrixBuilder
    generate_builders!(mwr::MatrixWithRules)::MatrixWithRules

Methods to construct MatrixBuilder that directly uses the MatrixWithRules
"""
function generate_builders(mwr::MatrixWithRules, dims)

    # Get all rule types
    rule_types = Set([typeof(rule) for rule in mwr.rules])

    builders = AbsoluteRuleBuilder[]

    for T in rule_types

        rules = [rule for rule in mwr.rules if rule isa T]

        push!(builders, matrix_builder(T, dims, mwr.index_to_coords))
    end 

    return builders
end
function generate_builders!(mwr::MatrixWithRules, dims)
    mwr.builders = generate_builders(mwr, dims)
    return mwr
end

"""
    build(mwr::MatrixWithRules; kwargs...)

* `kwargs`: kwargs to be passed to rules 
"""
function build(mwr::MatrixWithRules; kwargs...)

    use_cached = mwr.cache_kwargs && mwr._matrix_cached

    if use_cached
        for (k, cached_kwarg) in mwr._kwargs

            if !(k in keys( mwr._kwargs_defaults))
                throw(ArgumentError("Default does not exist for key $(k)."))
            end

            kwarg = get(kwargs, k, mwr._kwargs_defaults[k])
                   
            if kwarg != cached_kwarg
                mwr._kwargs[k] = copy(kwarg)
                use_cached = false
            end
        end
    end

    if !use_cached
        mat = zeros(M, T, dim(mwr), dim(mwr) ) 

        for builder! in mwr.builders
            builder!(mat, mwr.index_to_values; kwargs...)
        end

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


#=
##     ##    ###    ######## ########  #### ##     ##    ##      ## #### ######## ##     ##    ########  ##     ## #### ##       ########  ######## ########
###   ###   ## ##      ##    ##     ##  ##   ##   ##     ##  ##  ##  ##     ##    ##     ##    ##     ## ##     ##  ##  ##       ##     ## ##       ##     ##
#### ####  ##   ##     ##    ##     ##  ##    ## ##      ##  ##  ##  ##     ##    ##     ##    ##     ## ##     ##  ##  ##       ##     ## ##       ##     ##
## ### ## ##     ##    ##    ########   ##     ###       ##  ##  ##  ##     ##    #########    ########  ##     ##  ##  ##       ##     ## ######   ########
##     ## #########    ##    ##   ##    ##    ## ##      ##  ##  ##  ##     ##    ##     ##    ##     ## ##     ##  ##  ##       ##     ## ##       ##   ##
##     ## ##     ##    ##    ##    ##   ##   ##   ##     ##  ##  ##  ##     ##    ##     ##    ##     ## ##     ##  ##  ##       ##     ## ##       ##    ##
##     ## ##     ##    ##    ##     ## #### ##     ##     ###  ###  ####    ##    ##     ##    ########   #######  #### ######## ########  ######## ##     ##
=#
"""
    AbstractMatrixWithDimensions

An abstract interface for “builder” types that

  * expose `build_rules!(mwd; …)`  
  * expose `prep_matrix!(mwd; …)`  
  * expose `finalize!(mwd; …)`  

and store their rule state in a `mwr` field.
"""
abstract type AbstractMatrixWithDimensions end

"""
    set_defaults!(mwd::AbstractMatrixWithDimensions, args...; kwargs...)

Set default parameters where the matrix will be evaluated.
"""
set_defaults!(mwd::AbstractMatrixWithDimensions, args...; kwargs...) = mwd

"""
    error_check(mwd::AbstractMatrixWithDimensions, args...; kwargs...) = false

Implement error checking.
"""
error_check(mwd::AbstractMatrixWithDimensions, args...; kwargs...) = false

"""
     build_rules!(mwd::AbstractMatrixWithDimensions, args...; empty=false, kwargs...) 

Build the rules required to create the Hamiltonian.  This is interface is expected
to be implemented for each concrete type.
"""
function build_rules!(mwd::AbstractMatrixWithDimensions, args...; kwargs...)

    return mwd
end

"""
    prep_matrix!(mwd::AbstractMatrixWithDimensions, args...; kwargs...)

Initializes the matrix to be solved.  This step is based on the idea
that each time that matrices are needed we update all of the values
in them that need changing, and the size never changes.

Must be defined for each concrete type
"""
function prep_matrix!(mwd::AbstractMatrixWithDimensions, args...; kwargs...) 
    throw(MethodError(prep_matrix!, (mwd,))) 
end

"""
    finalize!(mwd::AbstractMatrixWithDimensions, args...; kwargs...)

Finalize the setup.  This does not need to be defined for each concrete type (but it can)
"""
function finalize!(mwd::AbstractMatrixWithDimensions, args...; kwargs...)

    # Error check using child class provided error checker
    error_check(mwd, args...; kwargs...)
    
    # Define all the rules
    build_rules!(mwd, args...; kwargs...)
    
    prep_matrix!(mwd, args...; kwargs...)

    return mwd
end

"""
    MatrixWithDimensions{T, M<:AbstractMatrix{T}}

Encodes the information regarding the dimensions the system is defined in.
This could be spatial dimensions, or some internal structure.

* `T` : element type of matrix
* `M` : type of matrix

* `_index_to_coords`: vector mapping indices to coordinates
* `_coords_to_index`: vector mapping coordinates to indices
"""
mutable struct MatrixWithDimensions{T, M<:AbstractMatrix{T}} <: AbstractMatrixWithDimensions
    mwr::MatrixWithRules{T, M}
    num_states::Int
end
MatrixWithDimensions(M, adims::AD.Dimensions) = MatrixWithDimensions{eltype(M), M}(
    adims, 
    MatrixWithRules(M),
    0,
    Vector{Int}[], 
    Vector{Float64}[]
)

#
# Add interface methods
#

"""
    dim(mwd::MatrixWithDimensions)

Returns the vector space dimension, so internally matrices will be dim x dim in size
"""
dim(mwd::MatrixWithDimensions) = dim(mwd.mwr)

"""
    `add_rule!(mwd::MatrixWithDimensions, args...; kwargs...)`

Adds a rule to `mwd`.  This is dispatched via the `add_rule!` method for `mwd.mwr`
"""
add_rule!(mwd::MatrixWithDimensions, args...; kwargs...) = add_rule!(mwd.mwr, args...; kwargs...)

"""
    prep_matrix!(mwd::MatrixWithDimensions; precompute_index=true)

Method specialized for MatrixWithDimensions
"""
function prep_matrix!(mwd::MatrixWithDimensions)
    
    # If we have not set the number of states, set it to the full system size
    if mwd.num_states == 0
        mwd.num_states = dim(mwd)
    end

    # Initialize the build rules
    return generate_builders!(mwd)
end

generate_builders(mwd::MatrixWithDimensions) = generate_builders!(mwd.mwr, mwd.adims, mwd._index_to_coords)

function generate_builders!(mwd::MatrixWithDimensions)
    mwd.mwr.builders = generate_builders(mwd)
    return mwd
end


end # MatrixBuilders