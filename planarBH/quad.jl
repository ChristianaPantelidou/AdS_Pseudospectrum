__precompile__()

#=
#  Perform Clenshaw-Curtis quadrature on a function: calculate the quadrature points
#  and weights, then return a matrix whose diagonal is the partial sum
=#

module quad

export Gram
export quadrature_GC
export quadrature_GL
export f1, f2

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

    function f_p(x::Array)
        rho = -x ./2 .+ (1/2)
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> 1 - (1 - rho[i])^4, foo, eachindex(rho))
        return foo
    end

    # Calculate Guassian quadrature weights assuming the endpoint grid
    function quadrature(x::Array)
        Nfl = trunc(Int, floor(length(x))/2)
        kappa = [i == 1 ? 2 : i == length(x) ? 2 : 1 for i in eachindex(x)]
        foo = Vector{eltype(x)}(undef, length(x))
        ThreadsX.foreach(eachindex(x)) do I
            mysum = ThreadsX.sum(2*j == length(x) ? cos(2 * j * x[I])/(4*j^2 - 1) : 2 * cos(2*j*x[I])/(4*j^2-1) for j in 1:Nfl)
            foo[I] = 2 * (1 - mysum) / (kappa[I] * length(x))
        end
        return foo
    end
    
    # Construct the first Gram matrix. Note that factors of (-2) come
    # from changing derivative matrices from x to rho
    function G1(x::Array, D::Matrix, m::Float64, q::Float64)
        rho = -x ./ 2 .+ (1/2)
        f = f_p(x)
        wts = quadrature(x)
        foo = Matrix{eltype(x)}(undef, (length(x), length(x)))
        bar = similar(foo)
        ThreadsX.foreach(eachindex(x)) do i
            foo[i,i] = wts[i] * (f[i]/(1 - rho[i])^3)
            bar[i,i] = wts[i] * (m^2 / (1- rho[i])^5 + q^2 / (1-rho[i])^3)
        end
        return bar + D' * (foo ./ 4) * D
    end

    # Construct the first Gram matrix. Note that factors of (-2) come
    # from changing derivative matrices from x to rho
    function G2(x::Array)
        rho = -x ./ 2 .+ (1/2)
        f = f_p(x)
        wts = quadrature(x)
        foo = Matrix{eltype(x)}(undef, (length(x), length(x)))
        ThreadsX.foreach(eachindex(x)) do i
            foo[i,i] = wts[i] * (2 - f[i]) / (1 - rho[i])^3
        end
        return foo
    end
    
    # Use quadrature to construct Gram matrices at double the spectral
    # density, then use interpolation to bring the result back to the
    # proper shape
    function Gram(D::Matrix, x::Vector, m::Float64, q::Float64)
        # Interpolation from high res to low res (2N+2)x(2N+2)
        Imat = interpolator(x)
        # Hi res grid
        y = Vector{eltype(x)}(undef, 2 * length(x))
        ThreadsX.map!(i -> cos(pi*i / (length(y)-1)), y, 0:length(y)-1)
        # Use high res grid to calculate G1, then interpolate down
        G1_int = Imat' * G1(y,D,m,q) * Imat
        Gup = reduce(hcat, [G1_int, zeros(eltype(x), size(D))])
        # Same for G2
        G2_int = Imat' * G2(y) * Imat
        Glow = reduce(hvat, [zeros(eltype(x), size(D)), G2_int])
        G = vcat(Gup, Glow)
    return G, inv(G)
    end

    # Interpolation between high and low resolution Gram matrices
    function interpolator(x::Vector)
        Nbar = 2 * length(x)
        xbar = Vector{eltype(x)}(undef, Nbar)
        ThreadsX.map!(i->cos(i*pi/Nbar), xbar, 0:Nbar-1)
        Imat = Matrix{eltype(x)}(undef, (Nbar,Nbar))
        kappa = [i == 1 ? 2 : i == Nbar ? 2 : 1 for i in eachindex(xbar)]
        println(kappa)
        ThreadsX.foreach(Iterators.product(eachindex(xbar), eachindex(x))) do (i,j)
            sum = ThreadsX.sum(k == length(x) ? cos(k * xbar[i]) * cos(k * x[j]) : 2 * cos(k * xbar[i]) * cos(k * x[j]) for k in 1:length(x))
            Imat[i,j] = (1 + sum) / (kappa[i] * length(x))
        end
        Iup = reduce(hcat,[Imat, zeros(eltype(Imat), size(Imat))])
        Idown = reduce(hcat, [zeros(eltype(Imat), size(Imat)), Imat])
        BigI = vcat(Iup, Idown)
        return BigI
    end

end