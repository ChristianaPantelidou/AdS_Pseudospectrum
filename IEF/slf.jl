__precompile__()

#=
#  Module for specifying functions present in the Sturm-Louiville form of
#  the L operators
=#

module slf

using ThreadsX

export V

    function V(x::Array, k::Int64)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> x[i]^2*k^2+3*(1-x[i]^4)/4+3*(1+x[i]^4), foo, eachindex(x))
        return foo
    end


end

