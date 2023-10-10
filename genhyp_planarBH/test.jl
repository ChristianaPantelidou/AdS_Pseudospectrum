using LinearAlgebra
using GenericLinearAlgebra

include("./matrixiteration.jl")
import .matrixiteration

# Example matrix
A = Matrix{Float64}([(4.0/5.0)  (-3.0/5.0)  0.0; (3.0/5.0)  (4.0/5.0)  0.0; 1.  1.  2.])
# Eigenvalues are 2, (4 \pm 3i)/5
println("Test matrix eigenvalues: ", GenericLinearAlgebra.eigvals(A))
println("Matrix before balancing: ", A)
eigvals = matrixiteration.ericsson(A, size(A)[1])
println("Done!\nFound the eigenvalues ", eigvals)
