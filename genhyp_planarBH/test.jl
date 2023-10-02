using LinearAlgebra

include("./inviter.jl")
import .inviter

# Make a random matrix
A = Matrix{Float64}([1.  100.  10000.; .01  1.  100.; .0001  .01  1.])
println("Matrix before balancing: ", A)
inviter.invit(A, 1)
println("Done!")
