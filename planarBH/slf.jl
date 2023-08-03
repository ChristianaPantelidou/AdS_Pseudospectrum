__precompile__()

#=
#  Module for specifying functions present in the Sturm-Louiville form of
#  the L operators
=#

module slf

using ThreadsX

export w, p, pp, V, gamma, gammap

    function f_p(x::Array)
        rho = x ./2 .+ (1/2)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> 1 - (1 - rho[i])^4, foo, eachindex(rho))
        return foo
    end

    function w(x::Array)
        return f_p(x) .- 2
    end

    function p(x::Array)
        return -f_p(x)
    end
    
    function pp(x::Array)
        rho = x ./ 2 .+ (1/2)
        f = f_p(x)
        foo = Vector{eltype(x)}(undef, length(rho))
        ThreadsX.map!(i -> (f[i] / (1 - rho[i])) - 4 * (1 - rho[i])^3, foo, eachindex(rho))
        return foo
    end
    
    function V(x::Array, m::Float64, q::Float64)
        rho = x ./ 2 .+ (1/2)
        f = f_p(x)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> (2 * f[i] - m^2 - 6) / (1 - rho[i])^2 - q^2 
            - 2 * (1 - rho[i])^2 , foo, eachindex(x))
        return foo
    end

    function gamma(x::Array)
        return f_p(x) .- 1
    end

    function gammap(x::Array)
        rho = x ./ 2 .+ (1/2)
        f = f_p(x)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> (1 - rho[i])^3 - 4 * (f[i] - 1) / (1 - rho[i]), foo, eachindex(rho))
        return foo
    end

end

