module Helpers

    # To add methods
    import Base: zeros

    using SparseArrays
    import QGas.NumericalTools.ArrayDimensions as AD

    export DimensionWithSpace

    # Zeros methods for currently supported matrix types.
    zeros(::Type{Matrix}, ::Type{T}, args...) where T = zeros(T, args...)
    zeros(::Type{Matrix{T}}, ::Type{T}, args...) where T = zeros(T, args...)

    zeros(::Type{SparseMatrixCSC}, ::Type{T}, args...) where T = spzeros(T, args...)
    zeros(::Type{SparseMatrixCSC{T}}, ::Type{T}, args...) where T = spzeros(T, args...)

    """
        `base_typeof(t)`

    Return the base type of a type, so Matrix{Float64} -> Matrix
    """
    function base_typeof(t)
        t = Base.unwrap_unionall(typeof(t))    # peel off any UnionAll layers
        return t.name.wrapper
    end

    """
        `DimensionWithSpace`

    A concrete type of `AbstractDimension` that has a `spatial` flag.
    """
    Base.@kwdef mutable struct DimensionWithSpace <: AD.AbstractDimension
        x0::Float64 = 0.0
        dx::Float64 = 1.0
        npnts::Int = 0
        unit::String = ""
        symmetric::Bool = false
        periodic::Bool = false
        spatial::Bool = true
    end

end