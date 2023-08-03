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
    
    # Construct the first Gram matrix from the phi_1 phi_2 term . 
    # Note that factors of (2) come
    # from changing derivative matrices from x to rho
    function G1(x::Array, D::Matrix, m::Float64, q::Float64)
        rho = x ./ 2 .+ (1/2)
        wts = quadrature(x)
        #print("High res quadrature weights: "); show(wts); println("")
        foo = Matrix{eltype(x)}(undef, size(D))
        fdiag = diagm(wts .* f_p(x))
        # Terms with derivatives
        foo = D' * (diagm(ThreadsX.map(i -> fdiag[i,i] * (2)^2 * (1 - rho[i]), eachindex(x))) * D) - D' * (4 .* fdiag) - (4 .* fdiag) * D
        foo += diagm(ThreadsX.map(i -> 4 * fdiag[i,i] / (1 - rho[i]) + wts[i] * (m^2 /(1 - rho[i]) + q^2 * (1 - rho[i])), eachindex(x)))
        return foo
    end

    # Construct the first Gram matrix. Note that factors of (2) come
    # from changing derivative matrices from x to rho
    function G2(x::Array)
        rho = x ./ 2 .+ (1/2)
        f = f_p(x)
        wts = quadrature(x)
        return diagm(ThreadsX.map(i -> wts[i] * (2 - f[i]) * (1 - rho[i]), eachindex(x)))
    end
    
    # Use quadrature to construct Gram matrices at double the spectral
    # density, then use interpolation to bring the result back to the
    # proper shape
    function Gram(x::Vector, D::Matrix, y::Vector, Dy::Matrix, m::Float64, q::Float64)
        # Interpolation from high res (2N+2)x(2N+2) to low res (N+1)x(N+1)
        Imat = interpolator(x,y)
        ImatT = Imat'
        # Use high res grid to calculate G1, then interpolate down
        #print("G1 high res: "); show(G1(y,Dy,m,q)); println("")
        temp = Matrix{eltype(x)}(undef, (length(y), length(x)))
        G1_int = Matrix{eltype(x)}(undef, size(D))
        # Safe matrix product to handle Infs
        G1mat = G1(y,Dy,m,q)
        # G(1,1) is Inf but only multiplies a non-zero value once
        @views ThreadsX.foreach(Iterators.product(eachindex(y), eachindex(x))) do (i,j)
            if i == 1 && j == 1
                temp[i,j] = Inf
            elseif i == 1
                temp[i,j] = dot(G1mat[i,2:end], Imat[2:end,j])
            else
                temp[i,j] = dot(G1mat[i,:], Imat[:,j])
            end
        end
        #print("Intermediate: "); show(temp); println("")
        @views ThreadsX.foreach(Iterators.product(eachindex(x), eachindex(x))) do (i,j)
            # Intermediate matrix has a row of Infs that multiplies either 0 or 1
            if isinf(temp[begin,j])
                if i != 1
                    # Infinite value is multiplied by 0 so does not contribute
                    G1_int[i,j] = dot(ImatT[i,begin+1:end], temp[begin+1:end,j])
                else
                    # Execpt the last row where it is multiplied by 1
                    G1_int[i,j] = Inf
                end
            else
                G1_int[i,j] = dot(ImatT[i,:], temp[:,j])
            end
        end
        #print("G1 interpolated: "); show(G1_int); println("")
        # Remove rows and columns corresponding to the rho = 1 boundary
        Gup = reduce(hcat, [view(G1_int, 2:length(x), 2:length(x)), zeros(eltype(x), (length(x)-1, length(x)-1))])
        # Same for G2
        #print("G2 high res: "); show(G2(y)); println("")
        G2_int = Imat' * G2(y) * Imat
        #print("G2 interpolated: "); show(G2_int); println("")
        # Remove rows and columns corresponding to the rho = 1 boundary
        Glow = reduce(hcat, [zeros(eltype(x), (length(x)-1, length(x)-1)), view(G2_int, 2:length(x), 2:length(x))])
        G = vcat(Gup, Glow)
    return G
    end

    function delta(x, y)
        return x == y ? 1 : 0
    end

    # Interpolation between high (y) and low (x) resolution Gram matrices
    function interpolator(x::Vector, y::Vector)
        tx = Vector{eltype(x)}(undef, length(x))
        ty = Vector{eltype(y)}(undef, length(y))
        ThreadsX.map!(i -> i * pi / (length(x) - 1), tx, 0:length(x)-1)
        ThreadsX.map!(i -> i * pi / (length(y) - 1), ty, 0:length(y)-1)
        Imat = Matrix{eltype(x)}(undef, (length(y),length(x)))
        kappa = [i == 1 ? 2 : i == length(x) ? 2 : 1 for i in eachindex(x)]
        ThreadsX.foreach(Iterators.product(eachindex(y), eachindex(x))) do (i,j)
            sum = ThreadsX.sum((2 - delta(k, length(x)-1)) * cos(k * ty[i]) * cos(k * tx[j]) for k in 1:length(x)-1)
            Imat[i,j] = (1 + sum) / (kappa[j] * (length(x)-1))
        end
        return Imat
    end

end