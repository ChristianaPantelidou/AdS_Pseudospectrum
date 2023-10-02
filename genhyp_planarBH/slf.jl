__precompile__()

#=
#  Module for specifying functions present in the Sturm-Louiville form of
#  the L operators
=#

module slf

using ThreadsX
using Roots

export p, pp, V, w, gamma, gammap

    #=
    function p_scale(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> (-1 + x[i]^2)^2*(1 + atanh(x[i])^2), foo, eachindex(x))
        # Vanishes at boundaries
        foo[1] = foo[length(x)] = 0
        println("p_scale(x) = ", foo)
        return foo
    end

    function pp_scale(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> 2*x[i]*(-1 + x[i]^2)*(1 + atanh(x[i])^2), foo, eachindex(x))
        # Vanishes at boundaries
        foo[1] = foo[length(x)] = 0
        println("pp_scale(x) = ", foo)
        return foo
    end

    function V_scale(x::Array, q::Float64, m::Float64)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.foreach(eachindex(x)) do I
            # Potential vanishes at the horizon
            if x[I] == -1
                foo[I] = 0
            # g'(x)/w(x) = 0 at the AdS boundary
            elseif x[I] == 1
                foo[I] = 0
            else
                r = root!(x[I])
                if r < 1
                    println("Root solver returned invalid radial position: ", r)
                    foo[I] = Nan
                end
                println("Check: x = ", x[I], " calculated r = ", r)
                foo[I] =(1 + atanh(x[I])^2) * (r^4 - 1) * ((q/r)^2 + 15/4 + m^2 + 9/(4*r^4))
            end
        end
        println("q_scale(x) = ", foo)
        return foo
    end

    function gamma_scale(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> -((-1 + x[i]^2)*(atanh(x[i]) + atanh(x[i])^3)), foo, eachindex(x))
        # Vanishes at boundaries
        foo[1] = foo[length(x)] = 0
        println("gamma_scale(x) = ", foo)
        return foo
    end

    function gammap_scale(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> 1/sqrt(1 + atanh(x[i])^2), foo, eachindex(x))
        # Vanishes at boundaries
        foo[1] = foo[length(x)] = 0
        println("gammap_scale(x) = ", foo)
        return foo
    end
    =#

    function p(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> 1 - x[i]^2, foo, eachindex(x))
        println("p(x) = ", foo)
        return foo
    end
    
    function pp(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> -2 * x[i], foo, eachindex(x))
        println("p'(x) = ", foo)
        return foo
    end
    
    # Calculate r for the given x. Potential is g'(x)V(r)
    function V(x::Array, q::Float64, m::Float64)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.foreach(eachindex(x)) do I
            # Potential vanishes at the horizon
            if x[I] == -1
                foo[I] = 0
            # Potential is infinite at the boundary
            elseif x[I] == 1
                foo[I] = NaN
            else
                r = root!(x[I])
                if r < 1
                    println("Root solver returned invalid radial position: ", r)
                    foo[I] = Nan
                end
                println("Check: x = ", x[I], " calculated r = ", r)
                foo[I] = gp(x[I]) * (r^4 - 1) * ((q/r)^2 + 15/4 + m^2 + 9/(4*r^4))
            end
        end
        println("q(x) = ", foo)
        return foo
    end

    function w(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> 1, foo, eachindex(x))
        return foo
    end

    function gamma(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> -x[i], foo, eachindex(x))
        return foo
    end

    function gammap(x::Array)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> -1, foo, eachindex(x))
        # Limit of 0 as x-> \pm 1
        foo[1]=foo[length(x)]=0
        return foo
    end

    # Spatial compactification: may be used for calculating roots
    function g(x)
        return atanh(x)
    end

    # Spatial compactification derivative: may be used for calculating 
    # roots and/or potential
    function gp(x)
        return 1/(1 - x^2)
    end

    # Calculate root of equation to solve for r(x)
    function root!(x)
        f(r,X=0) = 2*r + log((r-1)/(r+1)) - g(X)
        fp(r,X=0) = 2*r^2 / (r^2-1)
        fpp(r,X=0) = -4*r / (r^2-1)^2
        try
            return convert(eltype(x), find_zero(f,(1,10^10), p=x, atol=0.,rtol=0.))
        catch e
            println(e)
            println("Switching to higher-order root finder...")
            try
                foo = find_zero((f,fp,fpp),(1,10^10), p=x, Roots.Halley())
                println("Success!")
                return convert(eltype(x), foo)
            catch ee
                println(ee); println("No root found for x = ", x)
                println("Brackets: f(a) = ", f(fa,x), " f(b) = ", f(fb,x));
                return NaN
            end
        end
    end
end

