__precompile__()

#=
#  Perform Clenshaw-Curtis quadrature on a function: calculate the quadrature points
#  and weights then return a matrix whose diagonal is the partial sum
=#

module quad

export Gram
export quadrature_GC
export quadrature_GL
export quadrature

using LinearAlgebra
using ThreadsX
using BlockArrays

    # Return the quadrature weights assuming the endpoint grid
    function quadrature(x::Vector)
        nn = length(x) - 1
        tquad = ThreadsX.map(i->pi*i/nn, 0:nn)
        wts = Vector{eltype(x)}(undef, length(x))
        kappa = [i == 1 ? 2 : i == length(x) ? 2 : 1 for i in eachindex(x)]
        Nfl = trunc(Int, floor(nn/2))
        # T_n(cos x) = cos(n x)
        @inbounds ThreadsX.foreach(eachindex(wts)) do I
            mysum = ThreadsX.sum(k == Nfl ? cos(2*k * tquad[I]) / (4*k^2 - 1) : 2 * cos(2*k * tquad[I]) / (4*k^2 - 1) for k in 1:Nfl)
            wts[I] = 2 * (1 - mysum) / (kappa[I] * nn)
        end
        return wts
    end

    # Matrix factorization for matrices that are only approximately Hermitian (up to
    # machine precision)
    function factorize!(A::Matrix)
        # Test for symmetric up to machine precision
        if !(isapprox(A, A'))
            println("ERROR: Couldn't perform decomposition");
            nothing
        elseif ishermitian(A)
            println("factorize! detected a hermitian matrix")
            F = GenericLinearAlgebra.cholesky(A)
            return F.L
        else
            # Factorization should be of the BunchKaufman type. Produces F.U  
            # (upper-triangular matrix), F.P (permutation vector), F.D (tridiagonal) 
            A_bc = LinearAlgebra.bunchkaufman!(Hermitian(Matrix{Float64}(A)))
            # Perform Cholesky decomposition on tridiagonal portion
            L = LinearAlgebra.cholesky!(A_bc.D)
            # Construct the scaling matrix 
            F = L.L * A_bc.U' * A_bc.P
            if eltype(A) == BigFloat
                return Matrix{BigFloat}(F)
            else    
                return F
            end
        end
    end

    # Construct the first Gram matrix from the phi_1 phi_2 term . 
    # Note that factors of (2) come
    # from changing derivative matrices from x to rho
    function G1(x::Array, D::Matrix, m::Float64, q::Float64)
        qwts = quadrature(x)
        rho = (1/2) .- (x ./ 2)
        f_p = Vector{eltype(x)}(undef, length(x))
        ThreadsX.map!(i -> 1 - (1 - rho[i])^4, f_p, eachindex(rho))
        #print("High res quadrature weights: "); show(qwts); println("")
        # Terms with derivatives
        foo = diagm(ThreadsX.map(i -> qwts[i] * (4 * f_p[i]), eachindex(qwts))) * D + D' * diagm(ThreadsX.map(i -> qwts[i] * (4 * f_p[i]), eachindex(rho))) + D' * diagm(ThreadsX.map(i -> 4 * qwts[i] * (1 - rho[i]) * f_p[i], eachindex(rho))) * D
        #println(size(foo), foo)
        # Terms without derivatives
        foo += diagm(ThreadsX.map(i -> qwts[i] * ((4 * f_p[i] + m^2)/(1 - rho[i]) + q^2 * (1 - rho[i])), eachindex(rho)))
        #println(""); show(foo); println("")
        return foo
    end

    # Construct the first Gram matrix. Note that factors of (2) come
    # from changing derivative matrices from x to rho
    function G2(x::Array)
        qwts = quadrature(x)
        rho = (1/2) .- (x ./ 2)
        return diagm(ThreadsX.map(i -> qwts[i] * (1 - rho[i]) * (1 + (rho[i] - 1)^4), eachindex(x)))
    end
    
    # Use quadrature to construct Gram matrices at double the spectral
    # density, then use interpolation to bring the result back to the
    # proper shape
    function Gram(x::Vector, D::Matrix, y::Vector, Dy::Matrix, m::Float64, q::Float64)
        # Interpolation from high res (2N+2)x(2N+2) to low res (N+1)x(N+1)
        Imat = interpolator(x,y)
        ImatT = Imat'
        # Use high res grid to calculate G1, then interpolate down
        temp = Matrix{eltype(x)}(undef, (length(y), length(x)))
        G1_int = Matrix{eltype(x)}(undef, size(D))
        # Safe matrix product to handle Infs
        G1mat = G1(y,Dy,m,q)
        #print("\nG1 full res: ", size(G1mat), " "); show(G1mat); println("")
        # G(N,N) is Inf but only multiplies a non-zero value once
        @views ThreadsX.foreach(Iterators.product(eachindex(y), eachindex(x))) do (i,j)
            if i == length(y) && j == length(x)
                temp[i,j] = Inf
            elseif i == length(y)
                temp[i,j] = dot(G1mat[i,1:end-1], Imat[1:end-1,j])
            else
                temp[i,j] = dot(G1mat[i,:], Imat[:,j])
            end
        end
        #print("\nPartial interpolation: ", size(temp), " "); show(temp); println("")
        @views ThreadsX.foreach(Iterators.product(eachindex(x), eachindex(x))) do (i,j)
            # Intermediate matrix has an Inf that multiplies either 0 or 1
            if isinf(temp[end,j])
                if i != length(x)
                    # Infinite value is multiplied by 0 so does not contribute
                    G1_int[i,j] = dot(ImatT[i,1:end-1], temp[1:end-1,j])
                else
                    # Execpt the last row where it is multiplied by 1
                    G1_int[i,j] = Inf
                end
            else
                G1_int[i,j] = dot(ImatT[i,:], temp[:,j])
            end
        end
        #print("\nG1 low res: ", size(G1_int), " "); show(G1_int); println("")
        nrows, ncols = size(G1_int)
        # Remove rows and columns corresponding to the rho = 1 boundary
        Fup = factorize!(Matrix(view(G1_int, 1:nrows-1, 1:ncols-1)))
        Gup = reduce(hcat, [view(G1_int, 1:nrows-1, 1:ncols-1), zeros(eltype(x), size(G1_int) .- 1)])
        #print("\nG_upper: ", size(Gup), " "); show(Gup); println("")

        # Same for G2
        #print("G2 high res: "); show(G2(y)); println("")
        G2_int = ImatT * G2(y) * Imat
        # Remove rows and columns corresponding to the rho = 1 boundary and factor
        Flow = factorize!(Matrix(view(G2_int, 2:length(x), 2:length(x))))
        Glow = reduce(hcat, [zeros(eltype(x), (length(x)-1, length(x)-1)), view(G2_int, 2:length(x), 2:length(x))])

        # Stack and return
        G = vcat(Gup, Glow)
        F = vcat(reduce(hcat, [Fup, zeros(eltype(x), size(Fup))]),
                reduce(hcat, [zeros(eltype(x), size(Flow)), Flow]))
    return G, F
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
        #=
        println(""); print("Interpolation matrix: "); show(Imat); println("")
        testm = [0.015873 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 
        0.0 0.146219 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 
        0.0 0.0 0.279365 0.0 0.0 0.0 0.0 0.0 0.0; 
        0.0 0.0 0.0 0.361718 0.0 0.0 0.0 0.0 0.0; 
        0.0 0.0 0.0 0.0 0.393651 0.0 0.0 0.0 0.0; 
        0.0 0.0 0.0 0.0 0.0 0.361718 0.0 0.0 0.0; 
        0.0 0.0 0.0 0.0 0.0 0.0 0.279365 0.0 0.0; 
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.146219 0.0; 
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.015873]
        println(""); print("Test interpolation: "); show(Imat' * testm * Imat); println("")
        =#
        return Imat
    end

end