using LinearAlgebra

include("./inviter.jl")
import .inviter

# Example matrix
A = Matrix{Float64}([1.  2.  3.0; 4.  5.  6.0; 7.  8.  9.])
println("Matrix before balancing: ", A)
inviter.invit(A, 1)
println("Done!")
