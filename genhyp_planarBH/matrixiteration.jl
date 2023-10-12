__precompile__()
#=
# Apply inverse iteration to determine smallest eigenvalue, deflate matrix
# and repeat for as many eigenvalues are required
=#

module matrixiteration

export invit, mbal!, findshift, ericsonn

using ThreadsX
using GenericLinearAlgebra
using Preconditioners
using LinearAlgebra
using LinearSolve
using IncompleteLU
using IterativeSolvers
using KrylovKit
using Arpack
using ArnoldiMethod

const radix = 2.0
# From https://en.wikipedia.org/wiki/GNU_MPFR, BigFloat uses radix = 2
const EPS = 1.0*10^(-12)

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
        # Subtract off outer product times eigenvalue
        return A - lambda .* (u * u')
    end

    # Balance the matrix. To recover the matrix, multiply the ith 
    # column of the matrix by the ith scale 
    function mbal!(A::Matrix)
        A_cpy = similar(A)
        copyto!(A_cpy, A)
        RADIX = 2.0
        convert(eltype(A[1]), RADIX)
        done = false
        count = 1
        scale = ones(eltype(A[1]), size(A)[1])
        while (!done) && (count < 100)
            done=true
            @views for i in 1:size(A_cpy)[1]
                # Calculate row and column norms
                r = sqrt(ThreadsX.sum(x^2 for x in A_cpy[i,:] if x != A_cpy[i,i]))
                c = sqrt(ThreadsX.sum(x^2 for x in A_cpy[:,i] if x != A_cpy[i,i]))
                if (r != 0.) && (c != 0.)
                    f = 1.
                    s = c^2 + r^2
                    while (abs(c) < abs(r / RADIX))
                        f *= RADIX
                        c *= RADIX^2
                    end
                    while (abs(c) > abs(r * RADIX)) 
                        c /= (RADIX^2)
                        f /= RADIX
                    end
                    if (abs(c^2 + r^2)/abs(f) < 0.95 * abs(s))
                        done=false
                        scale[i] *= f
                        # Apply similarity transformation
                        A_cpy[i,:] = A_cpy[i,:] ./ f
                        A_cpy[:,i] = A_cpy[:,i] .* f
                    end
                end
                count += 1
            end
        end
        if cond(A_cpy) < cond(A)
            println("Rebalance decreased condition number from ", cond(A),
            " to ", cond(A_cpy))
            copyto!(A, A_cpy)
            return scale
        else
            return 1.0
        end
    end

    function R2(R::Matrix)
        return LinearAlgebra.sqrt(mmul!(transpose(R) * R))
    end

    # Subtraction that avoids ROE
    function mdiff(x, y)
        if (x == convert(eltype(x), 0.0)) && (y == convert(eltype(y), 0.0))
            return convert(eltype(x), 0.0)
        else
            if isreal(x) && isreal(y)
                return (x^2 - y^2)/(x + y)
            else
                return mdiff(real(x), real(y)) + mdiff(imag(x), imag(y)) * 1im
            end
        end
    end

    # Use random seeds to find an advantageous shift that decreases
    # the condition number of the matrix
    function findshift(A::Matrix, Nshifts=100)
        icond = abs(cond(A))
        shifts = Vector{eltype(A[1])}(undef, Nshifts)
        conds = Vector{Real}(undef, Nshifts)
        ThreadsX.foreach(eachindex(shifts)) do i
            # Real shifts based on matrix data type
            if isreal(A)
                # Shifts in [-1,1]
                shifts[i] = 2.0 * rand(eltype(A[1])) - 1.
            # Complex shifts
            else
                shifts[i] = 2.0 * rand(eltype(A[1])) - 1.0 - 1im
            end
            conds[i] = cond(A + shifts[i] .* I)
        end
        #println("Shifts: ", shifts)
        min_shift = findmin(abs.(conds))[2]
        #println("Condition numbers: ", conds)
        #println("Condition number of best shift: ", conds[min_shift])
        if conds[min_shift] < icond
            return shifts[min_shift]
        else
            return nothing
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
        #scale = mbal!(A)
        #println("Matrix after balancing: ", A)
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

    function ecount(D::Matrix)
        lnum = 0
        for i in eachindex(diag(D))
            if D[i,i] > convert(eltype(D[1]), 0.0)
                lnum += 1
            else
                nothing
            end
        end
        return lnum
    end

    # Return true if imaginary parts are equal and real parts 
    # are conjugates
    function realpair(x, y)
        if isreal(x) || isreal(y)
            #println("\nERROR: eigenvalue pairs must be complex\n")
            return nothing
        elseif isapprox(imag(x), imag(y))
            if isapprox(real(x), -real(y))
                println("Real conjugates: ", x, " and ", y)
                return true
            else
                #println("Not a conjugates: ", x, " and ", y)
                return false
            end
        else
            # Not a real pair
            #println("Imaginary parts are not equal: ", x, " and ", y)
            return false
        end
    end

    # Return true if real parts are equal and imaginary parts 
    # are conjugates
    function ispair(x, y)
        if isreal(x) || isreal(y)
            #println("\nERROR: eigenvalue pairs must be complex\n")
            return nothing
        elseif isapprox(real(x), real(y))
            if isapprox(imag(x), -imag(y))
                println("Imaginary conjugates: ", x, " and ", y)
                return true
            else
                #println("Not a conjugates: ", x, " and ", y)
                return false
            end
        else
            # Not a complex pair
            #println("\nERROR: real parts are not equal: ", x, " and ", y)
            return false
        end
    end


    # Find and return eigenvalue pairs, sorted by size. Remaining values
    # are returned as 'shifts'
    function getpairs(v::Vector)
        p = Vector{eltype(v)}(undef,1)
        s = Vector{eltype(v)}(undef,1)
        for i in eachindex(v)
            # Look through remaining values for a pair
            for j in i:length(v)
                if ispair(v[i], v[j])
                    push!(p, [v[i], v[j]])
                else
                    push!(s, v[i])
                end
            end
        end
        # Return sorted pairs 
        return ThreadsX.sort(p[2:end], alg=ThreadsX.StableQuickSort, 
            by = x -> sqrt(real(x)^2 + imag(x)^2)), ThreadsX.sort(unique(s[2:end]), 
            alg=ThreadsX.StableQuickSort, by = x -> sqrt(real(x)^2 + imag(x)^2))
    end

    # Ericsson algorithm: https://doi.org/10.2307/2006390
    # Note: Arnoldi iteration is used in leiu of Lanczos to handle non-Hermitian
    # matrices. Iteration provided by KrylovKit underlying methods.
    # Schur factoring provided by GLA

    function ericsson(A::Matrix, neig=1)

        nrows = size(A)[1]
        if neig > nrows
            neig = nrows
        end
    
        A_bk = copy(A)
        
        eigs = GenericLinearAlgebra.eigvals(A)
        println("Eigenvalues by direct solve: ", ThreadsX.sort(eigs,
                alg=ThreadsX.StableQuickSort, by = x -> sqrt(real(x)^2 + imag(x)^2))[1:10])
        
    
        # Apply the Ericsson algorithm to the matrix A and extract 
        # the leading neig eigenvalues
    
        # Step 0: Rebalance and shift initial matrix
        #println("Intial condition number: ", cond(A))
        scale = mbal!(A)
        #println("Condition number after balancing: ", cond(A))
        # Find good shift 
        best_shift = findshift(A)
        if !isnothing(best_shift)
            #println("Best shift value: ", best_shift)
            A = A + best_shift .* I
            #println("Condition number after applying best shift: ", cond(A))
        else
            nothing
        end
    
        # Convergence criteria
        ATOL = 1.0*10^(-16)
        # Maximum Rayleigh iteration number
        IMAX = 300
        # Converged eigenvalues
        C = Vector(undef, 1)
        # Starting shift
        μ = convert(eltype(best_shift), 0.0)
        # Maximum eigenvalue in the current shift window
        emax = convert(eltype(μ), 0.0)
    
        # Random seed for Kyrlov space
        r = rand(eltype(μ), nrows)
    
        # t, z, foo = GenericLinearAlgebra.schur(A)
        #println("Schur decomposition: ", t, z, foo)
    
        # Step 3: Create and initialize Arnoldi iterator from KrylovKit
        Ait = KrylovKit.ArnoldiIterator(A, r)
        Afactor = KrylovKit.initialize(Ait)
    
        # Step 4: Expand Arnoldi iterator until tolerance is reached
        while normres(Afactor) > sqrt(eps())
            expand!(Ait, Afactor)
        end
    
        # Step 5: Find eigenvalues of the Hessenburg 
        V, B, r, bar = Afactor
        # Subdiagonal of the Hessenburg
        v = diag(B, -1)
        #println("Hesseburg subdiagonal of the inverse-shift: ", v)
        #println("Norm residual of the Arnoldi iteration: ", normres(Afactor))
        #println("Residual of the Arnoldi iteration: ", r)
        eigs = GenericLinearAlgebra.eigvals(B)
        println("Eigenvalues by Arnoldi iteration: ", ThreadsX.sort(eigs .- best_shift,
                alg=ThreadsX.StableQuickSort, by = x -> sqrt(real(x)^2 + imag(x)^2))[1:10])
        # Selection criteria: looking for eigenvalue pairs 
        pairs, shifts = getpairs(eigs)
        #println("Paired eigenvalues: ", pairs)
        #println("Number of non-paired eigenvalues: ", length(shifts))
        if length(pairs) >= 1
            C = vcat(C, pairs)
        end
        # If all the desired eigenvalues are found, then return the 
        # properly sorted eigenvalues 
        if length(C) + 1 > neig
            A = A_bk
            return ThreadsX.sort!(C[2:neig] .- best_shift, alg=ThreadsX.StableQuickSort, 
                by = x -> sqrt(real(x)^2 + imag(x)^2))
        end
    
        #println("Condition number of Hessenburg matrix: ", cond(B))
        Hess_shift = findshift(Matrix(B))
        if !isnothing(Hess_shift)
            #println("Shifted Hessenburg by ", Hess_shift)
            B = B + Hess_shift .* I
            #println("Shifted Hessenburg condition number: ", cond(B))
        end
    
        
        t, z, eigs = GenericLinearAlgebra.schur(copy(B))
        println("Full Schur eigenvalues: ", ThreadsX.sort(eigs .- best_shift .- Hess_shift,
                alg=ThreadsX.StableQuickSort, by = x -> sqrt(real(x)^2 + imag(x)^2))[1:10])
        
    
        # Largest eigenvalues of inverse matrix are the smallest of the matrix
        PS, history = ArnoldiMethod.partialschur(inv(B), nev=size(B)[1], tol=eps(), which=LM())
        if history.converged
            println("Partial Schur eigenvalues: ", ThreadsX.sort(1 ./ PS.eigenvalues .- best_shift .- Hess_shift,
                alg=ThreadsX.StableQuickSort, by = x -> sqrt(real(x)^2 + imag(x)^2))[1:10])
            #C = vcat(C, 1 ./ PS.eigenvalues)
        end
    
        # Single-shift QR: Wilkinson shift
        #=
            - Start with the trailing 2x2 submatrix:
                        a_{n-1,n-1} a_{n-1,n}
                        a_{n,n-1}   a_{n,n}
            - Shift is the eigenvalue e of the submatrix closest to a_{n,n}
            - Factor the shifted matrix A - e .* I = QR
            - Update the matrix A = RQ + e .* I
            - Repeat until the subdiagonal entry, a_{n,n-1} -> 0
        =#
        ii = 0
        nrows, ncols = size(B)
        # Apply Wilkinson single-shift QR to determine eigenvalues of B
        for j in nrows:-1:2
            #println("Size of B: ", size(B))
            while ii < IMAX
                B_sub = view(B, j-1:j, j-1:j)
                eig_sub = GenericLinearAlgebra.eigvals(B_sub)
                shift = abs(mdiff(B_sub[2,2], eig_sub[1])) < abs(mdiff(B_sub[2,2], eig_sub[2])) ? eig_sub[1] : eig_sub[2]
                ThreadsX.foreach(eachindex(diag(B))) do i
                    B[i,i] = mdiff(B[i,i], shift)
                end
                Q, R = GenericLinearAlgebra.qr(B)
                B = R * Q + shift .* I
                if abs(B[j,j-1]) < ATOL
                    # This line should be all zeros except for a single 
                    #println(B[j,:])
                    #println("Single-shift QR eigenvalue: ", B[j,j] - best_shift - Hess_shift, 
                    #    " converged after ", ii, " iterations")
                    C = vcat(C, B[j,j])
                    # Deflate the system
                    B = view(B, 1:j-1, 1:j-1)
                    ii = 0
                    break
                else
                    ii += 1
                end
            end
        end
    
        #=
        ii = 0
        while ii < 500 && norm(diag(B,-1)) > 10^(-5)
            Q, R = GenericLinearAlgebra.qr(B)
            B = R * Q
            ii += 1
            if ii % 10 == 0
                println("Subdiagonal norm: ", norm(diag(B,-1)))
            end
        end
        #println("Schur decomposition subdiagonal: ", diag(M[end], -1), "\n", norm(diag(B,-1)))
    
    
        eigs = GenericLinearAlgebra.eigvals(B)
        println("Eigenvalues of shifted Hessenburg: ", ThreadsX.sort(eigs, alg=ThreadsX.StableQuickSort, by = x -> sqrt(real(x)^2 + imag(x)^2)))
        t, z, eigs = GenericLinearAlgebra.schur(B)
        println("Eigenvalues of shifted Hessenburg Schur: ", ThreadsX.sort(eigs, alg=ThreadsX.StableQuickSort, by = x -> sqrt(real(x)^2 + imag(x)^2)))
        eigs = diag(GenericLinearAlgebra.lu(B).L)
        println("Eigenvalues of shifted Hessenburg: ", ThreadsX.sort(eigs, alg=ThreadsX.StableQuickSort, by = x -> sqrt(real(x)^2 + imag(x)^2)))
        =#
        #println("After QR shifts, the subdiagonal is: ", diag(B, -1))
        #C = vcat(C, GenericLinearAlgebra.eigvals(B))
    #=
        # Iterate through successive shifts until the desired 
        # number of eigenvalues are found 
        ii = 0
        while (length(C) + 1 <= neig) && ii <= IMAX
    
            μ = 2.0 * rand(eltype(eigs[1])) - 1.0 - 1im
            λ = [convert(eltype(eigs[1]), 0.0)]
            P = DiagonalPreconditioner(B - μ .* I)
            # Step 3: QR decomposition to shifted & preconditioned system
            Q, R = LinearAlgebra.qr(P \ (B - μ .* I))
    
            # Step 4: inverse iteration to find the n smallest eigenvalues
            b = rand(eltype(A[1]), nrows)
            normalize!(b)
            jj = 0
            # Shift-inverse method
            while jj < IMAX
    
                Z = adjoint(Q) * (P \ b)
                y = R \ Z
    
                push!(λ, μ)
    
                if (norm(b - y)/norm(y) < 10^(-12)) || (abs(λ[end-1] - λ[end])/max(abs(λ[end-1]),abs(λ[end])) < 10^(-12))
                    println("\nConvergence after ", jj, " iterations: ", μ + 1/λ[end])
                    # Add converged eigenvalue to list 
                    C = vcat(C, μ + 1/λ[end])
                    # Deflate matrix
                    B = deflate(B, y, μ + 1/λ[end])
                    break
                end
                if (jj + 1 == IMAX)
                    println("ERROR: Inverse iteration did not converge after ",
                    IMAX, " iterations.")
                    #println(λ)
                end
    
                # Update the eigenvector
                b = y ./ abs(maximum(y))
                # Repeat
                jj += 1
    
            end
    
            # Increment and repeat
            ii += 1
        end
    =#
    
    #=
        while(length(C) + 1 <= neig)
    
            break
            # Step 1: Factorize shifted system
            # factor(K - mu M) = V T V'
            println("Shift value: ", μ)
    
    
            # Step 2: Construct inverse
            zinv = inv(z)
            Ainv = zinv' * inv(t) * zinv
            if !isapprox(Ainv * (A + μ .* I), I)
                println("Inverse check falied. Residual is ", Ainv * (A + μ .* I) - I)
            end
    
    
    
            # Step 6: Add converged eigenvalues to list
            for i in eachindex(eigs)
                println(abs(r[i]), " ", abs(eigs[i]))
                if abs(r[i])/abs(eigs[i]) < ATOL
                    println("Converged eigenvalue: ", -μ + 1/eigs[i])
                    C = append!(C, -μ + 1/eigs[i])
                end
            end
    
            #println("Converged eigenvalues: ", C)
            emax = maximum([abs(eigs[i]) for i in eachindex(eigs)])
    
            # Step 8: If more eigenvalues are required, adjust the shift
            # and repeat the iteration
            if emax > abs(μ) 
                μ += 2.0 * (emax - μ)
            end
    
        end
    =#
        # Return the desired eigenvalues
        #println("All eigenvalues before shifts: ", C[2:end])
        C = ThreadsX.sort!(C[2:end] .- best_shift .- Hess_shift, alg=ThreadsX.StableQuickSort, by = x -> sqrt(real(x)^2 + imag(x)^2))
        println("Single-shift QR eigenvalues: ", C[1:10])
        if neig > length(C)
            return C
        else
            return C[begin:neig]
        end
    end

end