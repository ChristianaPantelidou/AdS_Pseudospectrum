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
        else
            # Factorization should be of the BunchKaufman type. Produces F.U  
            # (upper-triangular matrix), F.P (permutation vector), F.D (tridiagonal) 
            A_bc = LinearAlgebra.bunchkaufman!(Hermitian(copy(A)))
            #println(A_bc.D, size(A_bc.D))
            # Perform Cholesky decomposition on tridiagonal portion
            L = LinearAlgebra.cholesky!(-A_bc.D)
            # Construct the scaling matrix 
            F = L.L * A_bc.U' * A_bc.P 
            return F
        end
    end

   # the function f here includes the factor of r^2
   function f_p(x::Array)
        z = x ./ 2 .+ (1/2)
        return ThreadsX.map(i -> (1 - z[i]^4)/z[i]^2, eachindex(z))
    end
    
##############change################
    # Construct the first Gram matrix from the phi_1 phi_2 term . 
    # Note that factors of (2) come
    # from changing derivative matrices from z to x
    function G1(x::Array, D::Matrix)
        qwts = quadrature(x)
        z = x ./ 2 .+ (1/2)
        f = f_p(x)
        #print("High res quadrature weights: "); show(wts); println("")
        foo = Matrix{eltype(x)}(undef, size(D))
        # Terms with derivatives
        foo = D' * (diagm(ThreadsX.map(i -> qwts[i] *4* z[i]^5*(z[i]^4-1),eachindex(z)))) * D
    # Terms with 1 derivative
        foo +=   D' * (diagm(ThreadsX.map(i -> qwts[i]*5*z[i]^4*(z[i]^4-1),eachindex(z))) + diagm(ThreadsX.map(i -> qwts[i]*5*z[i]^4*(z[i]^4-1),eachindex(z)))) * D

        # Terms without derivatives
        foo += diagm(ThreadsX.map(i -> qwts[i]*2*z[i]^3 *(2*z[i]^4-5), eachindex(z)))
        #println("foo")
        #println(size(foo))
        return foo
    end

    
    # Use quadrature to construct Gram matrices at double the spectral
    # density, then use interpolation to bring the result back to the
    # proper shape
    function Gram(x::Vector, D::Matrix, y::Vector, Dy::Matrix)
        # Interpolation from high res (2N+2)x(2N+2) to low res (N+1)x(N+1)
        Imat = interpolator(x,y)
        ImatT = Imat'
        # Use high res grid to calculate G1, then interpolate down
        temp = Matrix{eltype(x)}(undef, (length(y), length(x)))
        G1_int = Matrix{eltype(x)}(undef, size(D))
        # Safe matrix product to handle Infs
        G1mat = G1(y,Dy)
	#println("G1mat", size(G1mat))
	#println(G1mat)

        @views ThreadsX.foreach(Iterators.product(eachindex(y), eachindex(x))) do (i,j)
            
            temp[i,j] = dot(G1mat[i,:], Imat[:,j])
        end
        @views ThreadsX.foreach(Iterators.product(eachindex(x), eachindex(x))) do (i,j)
            # Intermediate matrix has a row of Infs that multiplies either 0 or 1
            G1_int[i,j] = dot(ImatT[i,:], temp[:,j])
        end
	
	#Gup = reduce(hcat, [view(G1_int, 1:length(x), 1:length(x)), zeros(eltype(x), (length(x), length(x)))])
	#G = vcat(Gup, Gup)
	G=G1_int

    #println("G", size(G))
    #println(G)
    #println(temp)
    
    Fup = factorize!(Matrix(view(G, 1:length(x), 1:length(x))))
    #F = vcat(reduce(hcat, [Fup, zeros(eltype(x), size(Fup))]), reduce(hcat, [zeros(eltype(x), size(Fup)), Fup]))
	F=Fup
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
