__precompile__()

#=
#  Module for specifying functions present in the Sturm-Louiville form of
#  the L operators
=#

module slf

using ThreadsX

export p, pp, V

    function p(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> - x[i]^2 * (1 - x[i]^4), foo, eachindex(x))
        return foo
    end
    
    function pp(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> 4 * x[i]^5, foo, eachindex(x))
        return foo
    end
    
    function V(x::Array, q::Float64)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> x[i]^2 * q^2 + 3 * (1 - x[i]^4) / 4 + 3 * (1 + x[i]^4), foo, eachindex(x))
        return foo
    end

end

