__precompile__()

#=
#  Module for specifying functions present in the Sturm-Louiville form of
#  the L operators. Simplification of the ratios of f(x)/w(x) is assumed to
#  have been already performed (functions will not be scaled by a density)
=#

module slf

using ThreadsX

export p, pp, V

    function p(x::Array, rh::Float64)
        z = x ./ 2 .+ (1/2)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> -4 * z[i]^2 * (1 - z[i]) * ((1 + (rh)^(-2)) * z[i]^2 + z[i] + 1), foo, eachindex(x))
        return foo
    end
    
    function pp(x::Array, rh::Float64)
        z = x ./ 2 .+ (1/2)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> 2 * z[i]^2 * (3 * (1 + (rh)^(-2)) * z[i]^2 - 2 * z[i] / (rh)^2), foo, eachindex(x))
        return foo
    end
    
    function V(x::Array, l::Int, rh::Float64)
        z = x ./ 2 .+ (1/2)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> 2 + z[i]^2 * ((1 + rh^2)* z[i] + l * (l + 1)) / (rh^2), foo, eachindex(x))
        return foo
    end

end

