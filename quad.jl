__precompile__()

#=
#  Perform Clenshaw-Curtis quadrature on a function: calculate the quadrature points
#  and weights, then return a matrix whose diagonal is the partial sum
=#

module quad

export Gram
export quadrature_GC
export quadrature_GL

using LinearAlgebra
using ThreadsX
using BlockArrays

    # Returns weights and points for Clenshaw-Curtis quadrature
    function quadrature_GC(x::Vector)
        # Quadrature points
        tquad = Vector{eltype(x)}(undef, length(x))
        @inbounds ThreadsX.foreach(eachindex(x)) do i
            tquad[i] = 0.5*pi*(2*i - 1)/length(x)
        end
        # Quadrature weights
        Nfl = trunc(Int, floor(length(x))/2)
        wts = Vector{eltype(x)}(undef, length(x))
        sums = Vector{eltype(x)}(undef, length(x))
        # Use the identity T_n(cos x) = cos(n x) to simplify the calculation of weights
        ThreadsX.foreach(eachindex(x)) do i
            sums[i] = ThreadsX.sum(cos(2 * j * tquad[i])/(4*j^2 - 1) for j in 1:Nfl)
            wts[i] = 2 * (1 - 2 * sums[i]) / length(x)
        end
        # Partial sums are the weights times the function evaluated at the 
        # quadrature points
        return wts, cos.(tquad)
    end

    # Returns weights and points for Clenshaw-Curtis quadrature
    function quadrature_GL(x::Vector)
        # Quadrature points
        tquad = Vector{eltype(x)}(undef, length(x))
        @inbounds ThreadsX.foreach(eachindex(x)) do i
            tquad[i] = cos(pi * i / length(x))
        end
        kappa = [i == 1 ? 2 : i == length(x) ? 2 : 1 for i in eachindex(x)]
        # Quadrature weights
        Nfl = trunc(Int, floor(length(x))/2)
        wts = Vector{eltype(x)}(undef, length(x))
        sums = Vector{eltype(x)}(undef, length(x))
        # Use the identity T_n(cos x) = cos(n x) to simplify the calculation of weights
        ThreadsX.foreach(eachindex(x)) do i
            sums[i] = ThreadsX.sum(2*j == length(x) ? cos(2 * j * tquad[i])/(4*j^2 - 1) : 2 * cos(2 * j * tquad[i])/(4*j^2 - 1) for j in 1:Nfl)
            wts[i] = 2 * (1 - 2 * sums[i]) / (kappa[i] * length(x))
        end
        # Partial sums are the weights times the function evaluated at the 
        # quadrature points
        return wts, cos.(tquad)
    end
    
    # Use quadrature to construct Gram matrices an inverses
    # Function V is from the Sturm Louiville formulation and 
    # takes an array of points. D is the derivative matrix from
    # Chebyshev discretization
    function Gram(D::Matrix, x::Vector, rh::Float64, l::Int, abscissa::String)
        # Construct empty matrix
        G = similar(D)
        if abscissa == "GC"
            wts, pts = quadrature_GC(x)
            C1 = reshape(Diagonal(wts .* f1(pts, rh)), (length(x), length(x)))
            C2 = reshape(Diagonal(wts .* f2(pts, rh, l)), (length(x), length(x)))
            G = D' * (C1 ./ (8 * rh)) * D + C2 .* (rh / 4)
        elseif abscissa == "GL"
            # Gridpoints can't contain the boundaries -- reconstruct
            # the collocation points for the integration and evaluate
            # on that grid. Then use barycentric interpolation to
            # produce the functions at the requested points
            nothing
        else
            println("\nERROR: Unrecognized abscissa in Gram(). Must be one of 'GC', or 'GL'")
        end
    return G, inv(G)
    end

    # Derivative integral in Gram matrix
    function f1(x::Array, rh::Float64)
        return ThreadsX.map(i -> (1 - x[i]) * ((x[i] + 1)^2 + rh^2 * (x[i]^2 + 4 * x[i] + 7)), eachindex(x))
    end

    # Potential integral in Gram matrix
    function f2(x::Array, rh::Float64, l::Int)
        return ThreadsX.map(i -> 8 / (1 + x[i])^2 + l * (l + 1) / rh^2 + 
        (1 + rh^2) * (1 + x[i]^2) / (2 * rh^2), eachindex(x))
        return foo
    end
end