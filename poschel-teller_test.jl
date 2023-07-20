using Base.Threads
using LinearAlgebra
using ThreadsX
using Base.MPFR
using GenericLinearAlgebra


function BF_basis(N::Integer)::Vector{BigFloat} # BigFloat version: Gauss-Chebyshev collocation points
    foo = Vector{BigFloat}(undef, N)
    ThreadsX.map!(i->BigFloat(cos(pi*(2*i-1)/(2*N))),foo,1:N)
    return foo
end

function BF_derivative(i::Integer, j::Integer, x::Array)::BigFloat # BigFloat version: Calculate an element of the first derivative matrix
    if i != j
        return BigFloat((-1.)^(i+j) * sqrt((1. - x[j] * x[j]) /
        (1. - x[i] * x[i])) / (x[i] - x[j]))
    else
        return BigFloat(0.5 * x[i] / (1. - x[i] * x[i]))
    end
end

function BF_dderivative(i::Integer, j::Integer, x::Array, D::Matrix)::BigFloat # BigFloat version: Calculate an element of the second derivative matrix
    if i == j
        return BigFloat(x[j] * x[j] / (1. - x[j] * x[j]) ^ 2 - (N * N - 1.) / (3. * (1. - x[j] * x[j])))
    else
        return BigFloat(D[i,j] * (x[i] / (1. - x[i] * x[i]) - 2. / (x[i] - x[j])))
    end
end


function BF_make_DD(x::Array, N::Integer, D::Matrix)::Matrix{BigFloat} # BigFloat version: Make the second derivative matrix
    foo = Matrix{BigFloat}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(Iterators.product(1:N, 1:N)) do (i,j)
        foo[i,j] = BF_dderivative(i, j, x, D)
    end
    return foo
end


function BF_make_D(x::Array, N::Integer)::Matrix{BigFloat} # BigFloat version: Make the derivative matrix
    foo = Matrix{BigFloat}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(Iterators.product(1:N, 1:N)) do (i,j)
        foo[i,j] = BF_derivative(i, j, x)
    end
    return foo
end


function BF_L1(x::Array, D::Matrix, DD::Matrix)::Matrix{Complex{BigFloat}} # BigFloat version: Make the L1 operator
    N = length(x)
    foo = Matrix{Complex{BigFloat}}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(1:N) do i
        foo[i,:] = BigFloat(-2 * x[i]) .* D[i,:] + BigFloat(1 - x[i] ^ 2) .* DD[i,:] # Dot operator applies addition to every element
        foo[i,i] -= BigFloat(1)
    end
    return foo
end

function BF_L2(x::Array, D::Matrix)::Matrix{Complex{BigFloat}} # BigFloat version: Make the L2 operator
    N = length(x)
    foo = Matrix{Complex{BigFloat}}(undef, (N,N))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(1:N) do i
        foo[i,:] = BigFloat(-2 * x[i]) .* D[i,:] # Dot operator applies addition to every element
        foo[i,i] -= BigFloat(1)
    end
    return foo
end

############################################################################
############################################################################

N = 75
P = 1024
setprecision(P)

x = BF_basis(N)
#print("x = "); show(x); println("")

D = BF_make_D(x,N)
#print("D = "); show(D); println("")

DD = BF_make_DD(x,N,D)
#print("DD = "); show(DD); println("")

L = BF_L1(x, D, DD)
#print("L1 = "); show(L); println("")

LL = BF_L2(x,D)
#print("L2 = "); show(LL); println("")

# Stack the matrices
Lupper = reduce(hcat, [zeros(eltype(x), (N,N)), Matrix{Complex{eltype(x)}}(I,N,N)]) # Match the data type of the collocation array
Llower = reduce(hcat, [L, LL])
BigL = vcat(Lupper, Llower)
BigL = BigL .* -1im # Automatic type matching
#print("L operator = "); show(BigL); println("")

vals = GenericLinearAlgebra._eigvals!(BigL)
print("eigenvalues = "); show(sort(vals, by=abs)); println("")