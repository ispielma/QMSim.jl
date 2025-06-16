"""
    MatrixBuilders

A module for creating matrices using a mixture of rules or direct pre-computed matrices
"""
module MatrixBuilders

using LinearAlgebra, SparseArrays, StaticArrays

export MatrixAndRules, add_rule!

"""
    AbstractRule(args...; kwargs...) → complex number

Top-level type for all rule objects.  Sub-types are either

* **ElementRule** – returns a while matrix element per call.
* **MatrixRule**  – returns a whole matrix.

"""
abstract type AbstractRule end

abstract type AbstractElementRule <: AbstractRule end # Returns matrix element
abstract type AbstractMatrixRule  <: AbstractRule end # Returns whole matrix

"""
A rule that is a function of the displacement coordinate (the type of which is unknown.  Usually integer, but doesn't have to be)
"""
struct RelativeRule{D, T, F} <: AbstractElementRule
    Δ::SVector{D, T}
    func::F
end
RelativeRule(Δ, func) = RelativeRule(SVector(Δ), func)
(a::RelativeRule)(x0, x1; kwargs...) = a.func(x0, x1; kwargs...)

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


"""
    MatrixBuilder

A type that efficiently builds a matrix from rules.
"""
# TODO: I am currently just a stub
struct MatrixBuilder{T, M<:AbstractMatrix{T}} end
(a::MatrixBuilder)(args...; kwargs...) = nothing

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
    _matrix_builder::MatrixBuilder{T,M} # function to efficiently build the matrix from the rules
    _kwargs::Dict{Symbol, Any} # Most recent kwargs
    _kwargs_defaults::Dict{Symbol, Any} # Default kwargs
    _matrix_cached::Bool # whether the matrix is currently cached
end
MatrixAndRules(M::Type, cache_kwargs, options) = MatrixAndRules{eltype(M), M}(
    AbstractRule[],
    cache_kwargs, 
    options,
    M(undef, 0, 0), 
    MatrixBuilder{eltype(M), M}(), 
    Dict(),
    Dict(),
    false
)
MatrixAndRules(M, cache_kwargs) = MatrixAndRules(M, cache_kwargs, Dict{Symbol, Any}())

"""
    add_rule!(mat_rules::MatrixAndRules, rule::AbstractRule)

Adds a rule of any type to `mat_rules`.
"""
add_rule!(mat_rules::MatrixAndRules, rule::AbstractRule) = mat_rules.rules = vcat(mat_rules.rules, rule)

"""
    add_rule!(mat_rules::MatrixAndRules, f::Function)

This method assumes that we are adding an `RelativeRule` type which when called returns a matrix element for a specific transition.
"""
add_rule!(mat_rules::MatrixAndRules, f::Function, Δ::AbstractVector) = add_rule!(mat_rules, RelativeRule(f, Δ))

"""
    add_rule!(mat_rules::MatrixAndRules, f::Function)

This method assumes that we are adding an `AbsoluteRule` type which when called returns a matrix
"""
add_rule!(mat_rules::MatrixAndRules, f::Function) = add_rule!(mat_rules, AbsoluteRule(f))

"""
    add_rule!(mat_rules::MatrixAndRules, f::Function)

This method assumes that we are adding an `ExplicitRule` type which when called returns a matrix
"""
function add_rule!(mat_rules::MatrixAndRules{T,M}, matrix::AbstractMatrix) where {T,M}
    if T != eltype(matrix)
        throw(ArgumentError("MatrixAndRules type $(T) does not match passed matrix type $(eltype(matrix))"))
    end

    add_rule!(mat_rules, ExplicitRule(matrix))
end

"""
    MatrixWithBuilder

Encodes the functionality to actually build matrices.

* `_index_to_coords`: vector mapping indices to coordinates
* `_coords_to_index`: vector mapping coordinates to indices
"""
struct MatrixWithBuilder{T, M<:AbstractMatrix{T}}
    mat_rules::MatrixAndRules{T,M}

    _index_to_coords
    _coords_to_index
end


end