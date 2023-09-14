__precompile__()

#=
#  Module for specifying functions present in the Sturm-Louiville form of
#  the L operators
=#

module slf

using ThreadsX

export p, pp, V, w, gamma, gammap

    function p(rho::Array)
        foo = Vector{eltype(rho)}(undef, length(rho))
        ThreadsX.map!(i -> -1 + (-1 + rho[i])^4, foo, eachindex(rho))
        return foo
    end
    
    function pp(rho::Array)
        foo = Vector{eltype(rho)}(undef, length(rho))
        ThreadsX.map!(i -> (4 - 20*rho[i] + 30*rho[i]^2 - 20*rho[i]^3 + 5*rho[i]^4)/(-1 + rho[i]), foo, eachindex(rho))
        return foo
    end
    
    function V(rho::Array, q::Float64, m::Float64)
        foo = Vector{eltype(rho)}(undef, length(rho))
        ThreadsX.map!(i -> (m^2 + q^2*(-1 + rho[i])^2 + 4*(2 - 4*rho[i] + 6*rho[i]^2 - 4*rho[i]^3 + rho[i]^4))/(-1 + rho[i])^2, foo, eachindex(rho))
        return foo
    end

    function w(rho::Array)
        foo = Vector{eltype(rho)}(undef, length(rho))
        ThreadsX.map!(i -> -1 - (-1 + rho[i])^4, foo, eachindex(rho))
        return foo
    end

    function gamma(rho::Array)
        foo = Vector{eltype(rho)}(undef, length(rho))
        ThreadsX.map!(i -> -(-1 + rho[i])^4, foo, eachindex(rho))
        return foo
    end

    function gammap(rho::Array)
        foo = Vector{eltype(rho)}(undef, length(rho))
        ThreadsX.map!(i -> -5*(-1 + rho[i])^3, foo, eachindex(rho))
        return foo
    end

end

