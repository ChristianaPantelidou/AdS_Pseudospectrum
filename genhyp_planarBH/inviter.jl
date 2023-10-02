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

const radix = 2.0
# From https://en.wikipedia.org/wiki/GNU_MPFR, BigFloat uses radix = 2

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

    function deflate(A::Matrix, u::Array, lambda)
        # Ensure u is normalized
        normalize!(u)
        return A - (lambda * dot(u,ones(eltype(u),length(u)))) .* I
    end

    # Balance the matrix
    function mbal!(A::Matrix)
        RADIX = 2.0
        done = false
        count = 1
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
                        # Apply similarity transformation
                        A[i,:] = A[i,:] ./ f
                        A[:,i] = A[:,i] .* f
                    end
                end
                count += 1
            end
        end
    end


    function invit(A::Matrix, n=1) 
        # Step 1: balance matrix
        mbal!(A)
        # Step 2: get preconditioner 
        P = DiagonalPreconditioner(A)
        println(P)
        # Step 3: QR decomposition to shifted & preconditioned system
        tau = convert(eltype(A[1]), 1.)
        Q, R = LinearAlgebra.qr(P \ (A - tau .* I))
        println(Q, R)
        # Step 4: inverse iteration to find the n smallest eigenvalues
        tau = convert(eltype(A[1]), 1.)
        b = rand(eltype(A[1]), size(A)[1])
        Z = Q' * (P \ b)
        println(Z)
        y = R \ Z
        println(y)
    end

end