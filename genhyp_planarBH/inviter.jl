__precompile__()
#=
# Apply inverse iteration to determine smallest eigenvalue, deflate matrix
# and repeat for as many eigenvalues are required
=#

module inviter

export invit

using ThreadsX
using GenericLinearAlgebra
using Preconditioners
using LinearAlgebra
using LinearSolve
using IncompleteLU
using IterativeSolvers

const radix = 2.0
# From https://en.wikipedia.org/wiki/GNU_MPFR, BigFloat uses radix = 2
const EPS = 1.0*10^(-10)

    function mmul!(A::Matrix, x::Array)
        if size(A)[2] != length(x)
            println("\nERROR: sizes in mmul! did not match\n")
            return nothing
        end
        foo = Vector{eltype(x)}(undef, length(x))
        @views ThreadsX.foreach(eachindex(x)) do I
            foo[I] = dot(A[I,:], x)
        end
        return foo
    end

    function mmul!(A::Matrix, B::Matrix)
        if size(A)[1] != size(B)[2]
            println("\nERROR: sizes in mmul! did not match\n")
            return nothing
        end
        foo = reshape(similar(A), (size(A)[1], size(B)[2]))
        @views ThreadsX.foreach(Iterators.product(1:size(A)[1], 1:size(B)[2])) do (i,j)
            foo[i,j] = dot(A[i,:], B[:,j])
        end
        return foo
    end

    function deflate(A::Matrix, u::Array, lambda)
        # Ensure u is normalized
        normalize!(u)
        return A - (lambda * dot(u,ones(eltype(u),length(u)))) .* I
    end

    # Balance the matrix. To recover the matrix, multiply the ith 
    # column of the matrix by the ith scale 
    function mbal!(A::Matrix)
        RADIX = 2.0
        done = false
        count = 1
        scale = ones(eltype(A[1]), size(A)[1])
        while (!done) && (count < 100)
            done=true
            @views for i in 1:size(A)[1]
                # Calculate row and column norms
                r = sqrt(ThreadsX.sum(x^2 for x in A[i,:] if x != A[i,i]))
                c = sqrt(ThreadsX.sum(x^2 for x in A[:,i] if x != A[i,i]))
                if (r != 0.) && (c != 0.)
                    f = 1.
                    s = c^2 + r^2
                    while (c < r / RADIX)
                        f *= RADIX
                        c *= RADIX^2
                    end
                    while (c > r * RADIX) 
                        c /= (RADIX^2)
                        f /= RADIX
                    end
                    if ((c^2 + r^2)/f < 0.95 * s)
                        done=false
                        scale[i] *= f
                        # Apply similarity transformation
                        A[i,:] = A[i,:] ./ f
                        A[:,i] = A[:,i] .* f
                    end
                end
                count += 1
            end
        end
        return scale
    end

    function R2(R::Matrix)
        return LinearAlgebra.sqrt(mmul!(transpose(R) * R))
    end

    # Higher order difference (x - y) for increased precision
    function mdiff(x, y)
        if (x == convert(eltype(x), 0.0)) && (y == convert(eltype(y), 0.0))
            return convert(eltype(x), 0.0)
        else
            return (x^2 - y^2)/(x + y)
        end
    end

    # Preconditioned Conjugate Gradient method to find the eigenvector
    # of A (balanced) corresponding to lambda given the initial guess v
    function mPBCG(A::Matrix, Pl, v::Array, scale::Array, lambda=1, tol=EPS)
        iter = 0
        IMAX = 100
        bkden = convert(eltype(A[1]), 1.0)
        # Factor A for easy backward solve
        Q, R = LinearAlgebra.qr(A) 
        # Factor preconditioner for easy backward solve
        #PQ, PR = LinearAlgebra.qr(MatrixPl.D)
        D = Pl.D
        r = mmul!(A, v)
        println("Initial LHS: ", r)
        b = lambda .* v
        println("Initial RHS: ", b)
        r = b - r
        println("Initial relative difference: ", r ./ norm(r))
        rr = r
        p = r
        pp = rr
        # Convergence condition is |Ax - b|/|b| < tol
        z = [b[i]/D[i] for i in eachindex(b)]
        bnorm = norm(z)
        z = [r[i]/D[i] for i in eachindex(r)]
        println(z)
        # Main loop
        while (iter < IMAX )
            iter += 1
            zz = [rr[i]/D[i] for i in eachindex(rr)]
            bknum = ThreadsX.sum(dot(z, rr))
            # Calculate coefficient bk and direction vectors p, pp
            bk = bknum / bkden
            p = z + bk .* p
            pp = zz + bk .* pp
            bkden = bknum
            # Calculate coefficent ak, new iterate v, new residuals 
            r = mmul!(A, z)
            akden = ThreadsX.sum(dot(z, pp))
            ak = bknum / akden
            zz = mmul!(Matrix(transpose(A)), pp)
            v += ak .* p
            r -= ak .* z
            rr -= ak .* zz
            # Solve \tilde A z = r and check stopping criteria
            z = [r[i]/D[i] for i in eachindex(r)]
            err = norm(z)/bnorm
            if (err < tol)
                break
            end
        end
        return v
    end


    function invit(A::Matrix, n=1) 
        nsize = size(A)[1]
        A_cpy = copy(A)
        # Step 1: balance matrix in place and return scaling vector
        scale = mbal!(A)
        println("Matrix after balancing: ", A)
        println("Scaling vector: ", scale)
        tau = rand(eltype(A[1]))
        eig = [tau]
        # Step 2: get preconditioner 
        # NB. CholeskyPreconditioner requires the matrix before
        # positive definite; AMGPreconditioner requires the matrix
        # to be sparse. Methods in ilu require A to be sparse.
        P = DiagonalPreconditioner(A - tau .* I)
        println(P) 
        # Step 3: QR decomposition to shifted & preconditioned system
        Q, R = LinearAlgebra.qr(P \ (A - tau .* I))
        println("QR decomposition of preconditioned shift: ", Q, R)
        # Step 4: inverse iteration to find the n smallest eigenvalues
        b = rand(eltype(A[1]), nsize)
        normalize!(b)
        MAX = 1000
        ii = 0
        while (ii < MAX)
            Z = adjoint(Q) * (P \ b)
            y = R \ Z
            push!(eig, (adjoint(y) * (P \ (A - tau .* I)) * y) / norm(y))
            b = y ./ norm(y)
            if (norm(b - y)/max(norm(y)) < EPS) || (abs(eig[end-1] - eig[end])/max(abs(eig[end-1]), abs(eig[end])) < EPS)
                println("Convergence after ", ii, " iterations")
                break
            end
            if (ii + 1 == MAX)
                println("\nERROR: Inverse iteration did not converge after ",
                MAX, " iterations.")
                println(eig)
            end
            ii += 1
        end
        # Eigenvalues have been shifted by the choice of tau; 
        # shift back to return true eigenvalues
        eig = eig .+ tau
        println("Eigenvalue ", n, ": ", eig[end])
        println(eig)
        # Eigenvectors have been scaled during balancing; rescale
        # to return true eigenvectors
        #b = b .* scale[1]
        normalize!(b) 
        println("Eigenvector ", n, ": ", b)
        # Step 4.5: Use existing methods to refine b
        b = bicgstabl(A, b, 2, Pl=P, max_mv_products=2000, log=true)
        #b = mPBCG(A, P, eig[end] .* b, scale, eig[end])
        println("After PBCG: ", b)
        # Step 5: deflate system and repeat for the desired number
        # of eigenvalues
        A = deflate(A, b .* scale[n], eig[end])
        println("Deflated: ", A)
    end

end