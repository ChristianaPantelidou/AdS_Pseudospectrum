__precompile__()

#=
#  Module for specifying functions present in the Sturm-Louiville form of
#  the L operators
=#

module slf

using ThreadsX

export w, p, pp, V, gamma, gammap

    function zeta(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> exp(atanh(x[i])), foo, eachindex(x))
        return foo
    end

    function w(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> (-1/2)*((-1 + x[i]^2)*(2 - sqrt(1 - x[i]^2) + x[i]^2*(2 + sqrt(1 - x[i]^2))))/(1 + x[i]^2)^2, foo, eachindex(x))
        return foo
    end

    function p(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> 2*sqrt(1 - x[i]^2), foo, eachindex(x))
        return foo
    end
    
    function pp(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> (-2*x[i])/sqrt(1 - x[i]^2), foo, eachindex(x))
        return foo
    end
    
    function V(x::Array, l::Int64)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> (-(l*(1 + l)*(-1 + x[i])) + 2*(1 + x[i]))/(1 - x[i]^2)^(3/2), foo, eachindex(x))
        return foo
    end

    function gamma(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> 1 + sqrt(1 - x[i]^2) - (2*sqrt(1 - x[i]^2))/(1 + x[i]^2), foo, eachindex(x))
        return foo
    end

    function gammap(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> (x[i]*sqrt(1 - x[i]^2)*(5 + x[i]^2))/(1 + x[i]^2)^2, foo, eachindex(x))
        return foo
    end

end

