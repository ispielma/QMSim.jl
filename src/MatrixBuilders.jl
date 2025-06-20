"""
    MatrixBuilders

A module for creating matrices using a mixture of rules or direct pre-computed matrices
"""
module MatrixBuilders

using LinearAlgebra, StaticArrays
import QGas.NumericalTools.ArrayDimensions as AD

import Base: empty!

export DimensionWithSpace, RelativeRule
export MatrixAndRules, add_rule!
export MatrixWithBuilder, MatrixBuilder, MatrixBuilder!
export finalize!

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
    `AbstractRule(args...; kwargs...)` → complex number

Top-level type for all rule objects.  Sub-types are either

* **ElementRule** – returns a while matrix element per call.
* **MatrixRule**  – returns a whole matrix.

"""
abstract type AbstractRule end

abstract type AbstractElementRule <: AbstractRule end # Returns matrix element
abstract type AbstractMatrixRule  <: AbstractRule end # Returns whole matrix

"""
A rule that is a function of the displacement coordinate (currently fixed as an integer, but this need not be the case).
"""
struct RelativeRule{F} <: AbstractElementRule
    func::F
    Δ::Vector{Int}
end

(a::RelativeRule)(x0::Vector, x1::Vector; kwargs...) = a.func(x0, x1; kwargs...)

"""
A rule that is a function that returns individual matrix elements
"""
struct ElementRule{F} <: AbstractElementRule
    func::F
end
(a::ElementRule)(x0, x1; kwargs...) = a.func(x0, x1; kwargs...)

"""
A rule that is a function and returns a complete matrix
"""
struct AbsoluteRule{F} <: AbstractMatrixRule
    func::F
end
(a::AbsoluteRule)(kwargs...) = a.func(;kwargs...)

"""
A rule  that contains a matrix outright
"""
struct ExplicitRule{M<:AbstractMatrix} <: AbstractMatrixRule
    mat::M
end
(a::ExplicitRule)(kwargs...) = a.mat


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
    MatrixAndRules

This a container type that contains rules used to build matrices.

rules: Each rule is a dictionary, and each rule specifies an individual matrix element.

* `rules`: rules to be applied given relative coordinates.  
* `cache_kwargs::Bool`: whether to cache the kwargs of the rules
* `options`: Dictionary of options
* `matrix`: Most recent matrix

Private fields (prefixed `_`) should only be touched by helpers.
"""

mutable struct MatrixAndRules{T, M<:AbstractMatrix{T}}
    rules::Vector{AbstractRule}
    cache_kwargs::Bool
    options::Dict{Symbol, Any}
    matrix::M
    _kwargs::Dict{Symbol, Any} # Most recent kwargs
    _kwargs_defaults::Dict{Symbol, Any} # Default kwargs
    _matrix_cached::Bool # whether the matrix is currently cached
end
MatrixAndRules(M::Type, cache_kwargs, options) = MatrixAndRules{eltype(M), M}(
    AbstractRule[],
    cache_kwargs, 
    options,
    M(undef, 0, 0), 
    Dict(),
    Dict(),
    false
)
MatrixAndRules(M, cache_kwargs) = MatrixAndRules(M, cache_kwargs, Dict{Symbol, Any}())
MatrixAndRules(M) = MatrixAndRules(M, true)

"""
    `empty!(mat_rules::MatrixAndRules)`

Clears all rules from `mat_rules` and resets all values to their defaults.  This adds a method to the existing
`empty!(...)` family of functions.
"""
function empty!(mat_rules::MatrixAndRules{T, M}) where {T, M}
    mat_rules.rules = AbstractRule[]
    mat_rules._kwargs = Dict{Symbol, Any}()
    mat_rules.matrix = M(undef, 0, 0)

    mat_rules._kwargs = Dict{Symbol, Any}()
    mat_rules._kwargs_defaults = Dict{Symbol, Any}()
    mat_rules._matrix_cached = false

    return mat_rules
end

"""
    `add_rule!(mat_rules::MatrixAndRules, rule::AbstractRule)`

Adds a rule of any type to `mat_rules`.  Methods exist that directly take the arguments for each rule type.
"""
add_rule!(mat_rules::MatrixAndRules, rule::AbstractRule) = mat_rules.rules = vcat(mat_rules.rules, rule)

"""
    `add_rule!(mat_rules::MatrixAndRules, f::Function Δ::AbstractVector{Int})`

This method assumes that we are adding an `RelativeRule` type which when called returns a matrix element for a specific transition.
"""
add_rule!(mat_rules::MatrixAndRules, f::Function, Δ::AbstractVector{Int}) = add_rule!(mat_rules, RelativeRule(f, Δ))

"""
    `add_rule!(mat_rules::MatrixAndRules, f::Function)`

This method assumes that we are adding an `AbsoluteRule` type which when called returns a matrix
"""
add_rule!(mat_rules::MatrixAndRules, f::Function) = add_rule!(mat_rules, AbsoluteRule(f))

"""
    `add_rule!(mat_rules::MatrixAndRules{T,M}, matrix::AbstractMatrix) where {T,M}`

This method assumes that we are adding an `ExplicitRule` type which when called returns a matrix
"""
function add_rule!(mat_rules::MatrixAndRules{T, M}, matrix::AbstractMatrix) where {T, M}

    try
        matrix = convert(M, matrix)
    catch
        throw(ArgumentError("Unable to convert passed matrix type $(eltype(matrix)) to $(T) "))
    end

    add_rule!(mat_rules, ExplicitRule(matrix))
end

#=
##     ##    ###    ######## ########  #### ##     ##    ########  ##     ## #### ##       ########  ######## ########
###   ###   ## ##      ##    ##     ##  ##   ##   ##     ##     ## ##     ##  ##  ##       ##     ## ##       ##     ##
#### ####  ##   ##     ##    ##     ##  ##    ## ##      ##     ## ##     ##  ##  ##       ##     ## ##       ##     ##
## ### ## ##     ##    ##    ########   ##     ###       ########  ##     ##  ##  ##       ##     ## ######   ########
##     ## #########    ##    ##   ##    ##    ## ##      ##     ## ##     ##  ##  ##       ##     ## ##       ##   ##
##     ## ##     ##    ##    ##    ##   ##   ##   ##     ##     ## ##     ##  ##  ##       ##     ## ##       ##    ##
##     ## ##     ##    ##    ##     ## #### ##     ##    ########   #######  #### ######## ########  ######## ##     ##
=#

raw"""
    MatrixBuilder

Encodes the relative rules that are to be run each time a full matrix is assembled.  
This precomputes all calls to index_to_coords and valid_coords, increasing
performance, at the expense of memory.

The memory scaling is $N \times \mathrm{rules}$ (where $N$ is the vector size, not $N^2$
the matrix size).
"""
struct MatrixBuilder
    actions::Vector{Tuple{Tuple{Int,Int}, AbstractElementRule}}
end
(a::MatrixBuilder)(args...; kwargs...) = nothing

MatrixBuilder() = MatrixBuilder(Tuple{Tuple{Int,Int}, AbstractElementRule}[])


function MatrixBuilder(mwr::MatrixAndRules, adims::AD.Dimensions, index_to_coords::Vector{NTuple{D, Int}}) where D

    actions = Tuple{Tuple{Int,Int}, AbstractElementRule}[]   

    rules = [rule for rule in mwr.rules if rule isa RelativeRule]

    if length(rules) != 0

        for (i, coords) in enumerate(index_to_coords), rule in rules
            new_coords = coords .+ rule.Δ

            if AD.valid_coords(adims, new_coords)
                j = AD.coords_to_index(adims, new_coords)
                push!(actions, ((i, j), rule))
            end
        end
    end
    
    return MatrixBuilder(actions)
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
    AbstractMatrixWithBuilder

An abstract interface for “builder” types that

  * expose `empty!(mwb)`  
  * expose `build_rules!(mwb; …)`  
  * expose `prep_matrix!(mwb; …)`  
  * expose `finalize!(mwb; …)`  

and store their rule state in a `mat_rules` field.
"""
abstract type AbstractMatrixWithBuilder end

"""
    empty!(mwb::AbstractMatrixWithBuilder)

Reset the builder’s rules to an empty state.  Must be defined for each concrete type
"""
function empty!(mwb::AbstractMatrixWithBuilder)
    throw(MethodError(empty!, (mwb,))) 
end

"""
    set_defaults!(mwb::AbstractMatrixWithBuilder, args...; kwargs...)

Set default parameters where the matrix will be evaluated.
"""
set_defaults!(mwb::AbstractMatrixWithBuilder, args...; kwargs...) = mwb

"""
    error_check(mwb::AbstractMatrixWithBuilder, args...; kwargs...) = false

Implement error checking.
"""
error_check(mwb::AbstractMatrixWithBuilder, args...; kwargs...) = false

"""
     build_rules!(mwb::AbstractMatrixWithBuilder, args...; empty=false, kwargs...) 

Build the rules required to create the Hamiltonian. 
"""
function build_rules!(mwb::AbstractMatrixWithBuilder, args...; empty=false, kwargs...)

    if empty
        empty!(mwb)
    end

    return mwb
end

"""
    prep_matrix!(mwb::AbstractMatrixWithBuilder, args...; kwargs...)

Initializes the matrix to be solved.  This step is based on the idea
that each time that matrices are needed we update all of the values
in them that need changing, and the size never changes.

Must be defined for each concrete type
"""
function prep_matrix!(mwb::AbstractMatrixWithBuilder, args...; kwargs...) 
    throw(MethodError(prep_matrix!, (mwb,))) 
end

"""
    finalize!(mwb::AbstractMatrixWithBuilder, args...; kwargs...)

Finalize the setup.  This does not need to be defined for each concrete type (but it can)
"""
function finalize!(mwb::AbstractMatrixWithBuilder, args...; kwargs...)

    # Error check using child class provided error checker
    error_check(mwb, args...; kwargs...)
    
    # BuildRules Needs to be before Prep matrices, which makes the tables 
    # required to actually assign all the matrix elements
    build_rules!(mwb, args...; kwargs...)
    
    prep_matrix!(mwb, args...; kwargs...)

    return mwb
end

"""
    MatrixWithBuilder{D, T, M<:AbstractMatrix{T}}

Encodes the functionality to actually build matrices.

* `T` : element type of matrix
* `M` : type of matrix
* `D` : number of physical dimensions

* `_index_to_coords`: vector mapping indices to coordinates
* `_coords_to_index`: vector mapping coordinates to indices
"""
mutable struct MatrixWithBuilder{D, T, M<:AbstractMatrix{T}} <: AbstractMatrixWithBuilder
    adims::AD.Dimensions # Spatial dimensions of system
    mat_rules::MatrixAndRules{T, M}
    mat_builder::MatrixBuilder
    num_states::Int

    _index_to_coords::Vector{NTuple{D, Int}}
    _index_to_values::Vector{NTuple{D, Float64}}
end
MatrixWithBuilder(M, adims::AD.Dimensions) = MatrixWithBuilder{ndims(adims), eltype(M), M}(
    adims, 
    MatrixAndRules(M),
    MatrixBuilder(),
    0,
    NTuple{ndims(adims), Int}[], 
    NTuple{ndims(adims), Float64}[]
)

#
# Add interface methods
#

# Now create new methods for the standard interface that need to be changed
empty!(mwb::MatrixWithBuilder) = empty!(mat_builder.mat_rules)

"""
    dim(mwb::MatrixWithBuilder)

Returns the vector space dimension, so internally matrices will be dim x dim in size
"""
dim(mwb::MatrixWithBuilder) = length(mwb.adims)

"""
    `add_rule!(mwb::MatrixWithBuilder, args...; kwargs...)`

Adds a rule to `mwb`.  This is dispatched via the `add_rule!` method for `mwb.mat_rules`
"""
add_rule!(mwb::MatrixWithBuilder, args...; kwargs...) = add_rule!(mwb.mat_rules, args...; kwargs...)


function precompute_index_mapping(dims::AD.Dimensions)
    """
    define the mapping from linear array indices to matrix location
    """
    index_to_coords = AD.index_to_coords(dims)

    # define the mapping from linear array indices scaled values
    index_to_values = AD.index_to_values(dims)

    (index_to_coords = index_to_coords, index_to_values = index_to_values)
end
function precompute_index_mapping!(mwb::MatrixWithBuilder)
    mwb._index_to_coords, mwb._index_to_values = precompute_index_mapping(mwb.adims)
    return mwb
end

"""
    MatrixBuilder(mwb::MatrixWithBuilder)

A method to construct MatrixBuilder that directly uses the MatrixWithBuilder
"""
function MatrixBuilder!(mwb::MatrixWithBuilder)
    mwb.mat_builder = MatrixBuilder(mwb.mat_rules, mwb.adims, mwb._index_to_coords)
    return mwb
end

"""
    prep_matrix!(mwb::MatrixWithBuilder; precompute_index=true)

Method specialized for MatrixWithBuilder

precompute_index : precompute all of the mappings between array indices and physical values
"""
function prep_matrix!(mwb::MatrixWithBuilder{D, T, M}, precompute_index=true) where {D, T, M}
    
    if precompute_index
        precompute_index_mapping!(mwb::MatrixWithBuilder)
    end   
    
    mwb.mat_rules.matrix = M(undef, dim(mwb), dim(mwb))

    # If we have not set the number of states, set it to the full system size
    if mwb.num_states == 0
        mwb.num_states = dim(mwb)
    end

    # Initialize the build rules
    return MatrixBuilder!(mwb)
end

end