module Helpers

    using SparseArrays
    using QGas.NumericalTools.ArrayDimensions: AbstractDimension, Dimensions, index_to_coords, index_to_values

    # To add methods
    import Base: zeros, length

    export DimensionWithSpace, MatrixSharedData
    export get_dimensions, set_dimensions!, get_options, set_options!, get_cache_kwargs, set_cache_kwargs!
    export get_index_to_coords, get_index_to_values

    # Zeros methods for currently supported matrix types.
    zeros(::Type{Matrix}, ::Type{T}, args...) where T = zeros(T, args...)
    zeros(::Type{Matrix{T}}, ::Type{T}, args...) where T = zeros(T, args...)

    zeros(::Type{SparseMatrixCSC}, ::Type{T}, args...) where T = spzeros(T, args...)
    zeros(::Type{SparseMatrixCSC{T}}, ::Type{T}, args...) where T = spzeros(T, args...)

    """
        `base_typeof(t)`

    Return the base type of a type, so Matrix{Float64} -> Matrix
    """
    base_typeof(t) = base_typeof(typeof(t))
    base_typeof(T::Type) = Base.unwrap_unionall(T).name.wrapper


    """
        `DimensionWithSpace`

    A concrete type of `AbstractDimension` that has a `spatial` flag.
    """
    Base.@kwdef mutable struct DimensionWithSpace <: AbstractDimension
        x0::Float64 = 0.0
        dx::Float64 = 1.0
        npnts::Int = 0
        unit::String = ""
        symmetric::Bool = false
        periodic::Bool = false
        spatial::Bool = true
    end

    """
        MatrixSharedData

        * `adims`             : spatial / internal dimensions
        * `idx_to_coord`  : cached coordinate lookup
        * `index_to_values`  : cached value lookup
        * `options`           : Dict of user-tunable flags
        * `cache_kwargs`      : whether to remember last kwargs set
    """
    mutable struct MatrixSharedData
        adims           :: Dimensions
        idx_to_coord    :: Vector{Vector{Int}}
        idx_to_val      :: Vector{Vector{Float64}}
        options         :: Dict{Symbol,Any}
        cache_kwargs    :: Bool
    end    
    function MatrixSharedData(adims::Dimensions, options, cache_kwargs)
        idx_to_coord = index_to_coords(Vector, adims)
        idx_to_val = index_to_values(Vector, adims)
        MatrixSharedData(adims, idx_to_coord, idx_to_val, options, cache_kwargs)
    end
    MatrixSharedData(adims::Dimensions; options=Dict{Symbol,Any}(), cache_kwargs=true) = MatrixSharedData(adims, options, cache_kwargs)

    length(shared::MatrixSharedData) = length(shared.adims)

    get_dimensions(shared::MatrixSharedData) = shared.adims

    function set_dimensions!(shared::MatrixSharedData, adims)

        shared.adims
        shared.idx_to_coord = index_to_coords(Vector, adims)
        shared.idx_to_val = index_to_values(Vecrtor,adims)
    
        shared
    end

    get_options(shared::MatrixSharedData) = shared.options
    set_options!(shared::MatrixSharedData, options::Dict{Symbol, Any}) = (shared.options = options; shared)

    get_cache_kwargs(shared::MatrixSharedData) = shared.cache_kwargs
    set_cache_kwargs!(shared::MatrixSharedData, cache_kwargs::Bool) = (shared.cache_kwargs = cache_kwargs; shared)

    # overload methods from ArrayDimensions
    get_index_to_coords(shared::MatrixSharedData) = shared.idx_to_coord
    get_index_to_values(shared::MatrixSharedData) = shared.idx_to_val

end