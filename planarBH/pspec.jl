using Base.Threads
using BenchmarkTools
using DelimitedFiles
using ThreadsX
using Base.MPFR
using ProgressMeter
using Parameters
using Distributed
using Random
using LoopVectorization
using Preconditioners
@everywhere using LinearAlgebra
@everywhere using GenericLinearAlgebra

include("./gpusvd.jl") 
include("./quad.jl")
include("./slf.jl")
include("./vpert.jl")
include("./io.jl")
import .gpusvd, .quad, .slf, .pert, .io

#####################
#= Debug Verbosity =#
#####################

# Debug 0: no debugging information
# Debug 1: function timings and matrix inversion check
# Debug 2: outputs from 1 plus matrix outputs and quadrature check
const debug = 1

#######################################################
#= Psuedospectrum calculation leveraging parallelism =#
#######################################################

####################
#= Inputs =#
####################

@with_kw mutable struct Inputs
    N::Int64 = 4
    m::Float64 = 0
    q::Float64 = 1
    xmin::Float64 = -1
    xmax::Float64 = 1
    ymin::Float64 = -1
    ymax::Float64 = 1
    xgrid::Int64 = 2
    ygrid::Int64 = 2
    basis::String = "GL"
end

# Read parameters from input file and store in Inputs struct
# Perform checks on initial values when necessary
function readInputs(f::String)::Inputs
    # Open the specified file and read the inputs into a dictionary
    data = Dict{SubString, Any}()
    if isfile(f)
        open(f) do file
            for line in readlines(file)
                data[split(chomp(line),"=")[1]] = split(chomp(line), "=")[2]
            end
        end
    else
        println(""); println("ERROR: couldn't find input file ", f)
    end
    # Create struct input values
    inpts = Inputs()
    for k in collect(keys(data)) 
        if k == "spectral_N" 
            inpts.N = parse(Int64, get(data, k,nothing))
            if inpts.N < 4
                println("WARNING: number of spectral modes must be N > 3. " *
                "Defaulting to N = 4.")
                inpts.N = 4
            # Bad singular value decomposition with odd numbers of N
            elseif isodd(inpts.N)
                inpts.N += 1
            end
        elseif k == "m"
            inpts.m = parse(Float64, get(data, k, nothing))
        elseif k == "q"
            inpts.q = parse(Float64, get(data, k, nothing))
        elseif k == "p_gridx" 
            inpts.xgrid = parse(Int64, get(data, k,nothing))
            if inpts.xgrid < 1
                inpts.xgrid = 1
            end
        elseif k == "p_gridy"
            inpts.ygrid = parse(Int64, get(data, k, nothing))
            if inpts.ygrid < 1
                inpts.ygrid = 1
            end
        elseif k == "xgrid_min"
            inpts.xmin = parse(Float64, get(data, k, nothing))
        elseif k == "xgrid_max"
            inpts.xmax = parse(Float64, get(data, k, nothing))
        elseif k == "ygrid_min"
            inpts.ymin = parse(Float64, get(data, k, nothing))
        elseif k == "ygrid_max"
            inpts.ymax = parse(Float64, get(data, k, nothing))           
        elseif k == "basis"
            nothing
        else
            println(""); println("\nERROR: unexpected entry in input file: ", k)
        end
    end
    return inpts
end

#####################
#= Basis functions =#
#####################

# Take inputs to determine the type of collocation grid, grid size,
# desired precision
function make_basis(inputs::Inputs, P::Int)

    # Determine data types and vector lengths
    if P > 64
        x = Vector{BigFloat}(undef, inputs.N)
    else
        x = Vector{Float64}(undef, inputs.N)
    end

    println("Using the ", inputs.basis, " collocation grid.")
    # Algorithms for different collocation sets
    if inputs.basis == "GC"
        x = push!(x, 0)
        D = Matrix{eltype(x)}(undef, (length(x), length(x)))
        DD = similar(D)
        # Collocation points
        ThreadsX.map!(i -> cos(pi*(2*i + 1)/(2*(inputs.N + 1))), x, 0:inputs.N)
        # First derivative matrix
        ThreadsX.foreach(Iterators.product(0:inputs.N, 0:inputs.N)) do (i,j)
            if i != j
                D[i+1,j+1] = (-1)^(i-j) * sqrt((1 - x[j+1]^2) /
                (1 - x[i+1]^2)) / (x[i+1] - x[j+1])
            else
                D[i+1,i+1] = x[i+1] / (2*(1 - x[i+1]^2))
            end
        end
        # Second derivative matrix
        ThreadsX.foreach(Iterators.product(0:inputs.N, 0:inputs.N)) do (i,j)
            if i != j
                DD[i+1,j+1] = D[i+1,j+1] * (x[i+1] / (1 - x[i+1]^2) - 2 / (x[i+1] - x[j+1]))
            else
                DD[j+1,j+1] = x[j+1] * x[j+1] / (1 - x[j+1]^2)^2 - inputs.N * (inputs.N + 2) / (3 * (1. - x[j+1]^2))
            end
        end
        return x, D, DD

    elseif inputs.basis::String == "GL"
        # Collocation points
        ThreadsX.map!(i-> cos(pi*i / inputs.N), x, 0:inputs.N-1)
        x = push!(x, -1)
        # Reference the data type of the collocation vector 
        # for the other matrices
        D = Matrix{eltype(x)}(undef, (length(x), length(x)))
        DD = similar(D)
        kappa = [i == 1 ? 2 : i == length(x) ? 2 : 1 for i in eachindex(x)]
        # First derivative matrix
        ThreadsX.foreach(Iterators.product(0:inputs.N, 0:inputs.N)) do (i,j)
            if i == j
                if i == 0 && j == 0
                    D[i+1,j+1] = (2 * inputs.N^2 + 1) / 6
                elseif i == inputs.N && j == inputs.N
                    D[i+1,j+1] = -(2 * inputs.N^2 + 1) / 6
                else
                    D[i+1,i+1] = -x[i+1] / (2*(1 - x[i+1]^2))
                end
            else
                D[i+1,j+1] = kappa[i+1] * (-1)^(i-j) / (kappa[j+1] * (x[i+1] - x[j+1]))
            end
        end
        # Second derivative matrix
        ThreadsX.foreach(Iterators.product(0:inputs.N, 0:inputs.N)) do (i,j)
            if i == j
                if i == 0 && j == 0
                    DD[i+1,i+1] = (inputs.N^4 - 1) / 15
                elseif i == inputs.N && j == inputs.N
                    DD[i+1,i+1] = (inputs.N^4 - 1) / 15
                else
                    DD[i+1,i+1] = -1 / (1 - x[i+1]^2)^2 - (inputs.N^2 - 1) / (3 * (1 - x[i+1]^2))
                end
            elseif i == 0 && j != 0
                DD[i+1,j+1] = 2 * (-1)^(j) * ((2*inputs.N^2 + 1) * (1 - x[j+1]) - 6) / (3 * kappa[j+1] * (1 - x[j+1])^2)
            elseif i == inputs.N && j != inputs.N
                DD[i+1,j+1] = 2 * (-1)^(inputs.N + j) * ((2*inputs.N^2 + 1) * (1 + x[j+1]) - 6) / (3 * kappa[j+1] * (1 + x[j+1])^2)
            else
                DD[i+1,j+1] = (-1)^(i-j) * (x[i+1]^2 + x[i+1]*x[j+1] - 2) / 
                            (kappa[j+1] * (1 - x[i+1]^2) * (x[j+1] - x[i+1])^2)
            end
        end
        return x, D, DD

    elseif inputs.basis::String == "LGR"
        # Size of abscissa is N + 1
        n = length(x) - 1
        mrange = [pi*(2*n + 1 - 2*i) / (2*n + 1) for i in 0:n]
        # Collocation points
        x = @tturbo @. cos(mrange)
        # Reference the data type of the collocation vector 
        # for the other matrices
        D = Matrix{eltype(x)}(undef, (length(x), length(x)))
        DD = similar(D)
        # First derivative matrix
        ThreadsX.foreach(Iterators.product(1:n+1, 1:n+1)) do (i,j)
            if i == j
                # NB. Arrays start at 1
                if i == 1 && j == 1
                    D[i,j] = -n * (n + 1) / 3
                else
                    D[i,j] = 1 / (2 * (1 - x[i] * x[i]))
                end
            elseif i == 1 && j != 1
                D[i,j] = (-1)^(j) * sqrt(2*(1 + x[j])) / (1 - x[j])
            elseif i != 1 && j == 1
                D[i,j] = (-1)^(i-1) / (sqrt(2*(1 + x[i])) * (1 - x[i]))
            else
                D[i,j] = (-1)^(i-j) * sqrt((1 + x[j]) / (1 + x[i])) / (x[j] - x[i])
            end
        end
        # Second derivative matrix
        ThreadsX.foreach(Iterators.product(1:n+1, 1:n+1)) do (i,j)
            if i == j
                # NB. Arrays start at 1
                if i == 1 && j == 1
                    DD[i,j] = n * (n - 1) * (n + 1) * (n + 2) / 15
                else
                    DD[i,j] = -n * (n + 1) / (3 * (1 - x[i]^2))
                                - x[i] / (1 - x[i]^2)^2
                end
            elseif i == 1 && j != 1
                DD[i,j] = (-1)^(j-1) * 2 * sqrt(2 * (1 + x[j])) * (n * (n+1) * (1 - x[j]) - 3) / (3 * (1 - x[j])^2)
            elseif i != 1 && j == 1
                DD[i,j] = (-1)^i * (2 * x[i] + 1) / (sqrt(2) * (1 - x[i])^2 * (1 + x[i])^(3/2))
            else
                DD[i,j] = (-1)^(i+j) * (2 * x[i]^2 - x[i] + x[j] - 2) * sqrt((1 + x[j]) / (1 + x[i])) / ((x[i] - x[j])^2 * (1 - x[i]^2))
            end
        end
        return x, D, DD

    elseif inputs.basis::String == "RGR"
        # Size of abscissa is N + 1
        n = length(x) - 1
        mrange = [2* pi * i / (2 * n + 1) for i in 0:n]
        # Collocation points
        x = @tturbo @. -cos(mrange)
        # Reference the data type of the collocation vector 
        # for the other matrices
        D = Matrix{eltype(x)}(undef, (length(x), length(x)))
        DD = similar(D)
        # First derivative matrix
        ThreadsX.foreach(Iterators.product(1:n+1, 1:n+1)) do (i,j)
            if i == j
                # NB. Arrays start at 1
                if i == 1 && j == 1
                    D[i,j] = n * (n + 1) / 3
                else
                    D[i,j] = -1 / (2 * (1 - x[i] * x[i]))
                end
            elseif i == 1 && j != 1
                D[i,j] = (-1)^(j-1) * sqrt(2*(1 + x[j])) / (1 - x[j])
            elseif i != 1 && j == 1
                D[i,j] = (-1)^(i) / (sqrt(2*(1 + x[i])) * (1 - x[i]))
            else
                D[i,j] = (-1)^(i-j) * sqrt((1 + x[j]) / (1 + x[i])) / (x[i] - x[j])
            end 
        end
        # Second derivative matrix
        ThreadsX.foreach(Iterators.product(1:n+1, 1:n+1)) do (i,j)
            if i == j
                # NB. Arrays start at 1
                if i == 1 && j == 1
                    DD[i,j] = n * (n - 1) * (n + 1) * (n + 2) / 15
                else
                    DD[i,j] = -n * (n + 1) / (3 * (1 - x[i]^2))
                                - x[i] / (1 - x[i]^2)^2
                end
            elseif i == 1 && j != 1
                DD[i,j] = (-1)^(j-1) * 2 * sqrt(2 * (1 + x[j])) * (n * (n+1) * (1 - x[j]) - 3) / (3 * (1 - x[j])^2)
            elseif i != 1 && j == 1
                DD[i,j] = (-1)^i * (2 * x[i] + 1) / (sqrt(2) * (1 - x[i])^2 * (1 + x[i])^(3/2))
            else
                DD[i,j] = (-1)^(i+j) * (2 * x[i]^2 - x[i] + x[j] - 2) * sqrt((1 + x[j]) / (1 + x[i])) / ((x[i] - x[j])^2 * (1 - x[i]^2))
            end
        end
        return x, D, DD
    
    end
end

###############
#= Operators =#
###############

# SL operator format
function L1(x::Array, D::Matrix, DD::Matrix, p, pp, V, w, inpts::Inputs)
    rho = (1/2) .- (x ./ 2)
    p_v = p(rho)
    pp_v = pp(rho)
    V_v = V(rho, inpts.q, inpts.m)
    w_v = w(rho)
    foo = Matrix{Complex{eltype(x)}}(undef, size(D))
    # Factors of (-2) come from changing coordinates from x to rho
    @views ThreadsX.foreach(eachindex(x)) do I
        foo[I,:] = ((-2) * pp_v[I] / w_v[I]) .* D[I,:] + ((-2)^2 * p_v[I] / w_v[I]) .* DD[I,:]
        foo[I,I] += V_v[I] / w_v[I]
    end
    return foo
end

function L2(x::Array, D::Matrix, gamma, gammap, w)
    rho = (1/2) .- (x ./ 2)
    g_v = gamma(rho)
    gp_v = gammap(rho)
    w_v = w(rho)
    foo = Matrix{Complex{eltype(x)}}(undef, size(D))
    # Factors of (-2) come from changing coordinates from x to rho
    @views ThreadsX.foreach(eachindex(x)) do I
        foo[I,:] = (2 * (-2) * g_v[I] / w_v[I]) .* D[I,:]
        foo[I,I] += gp_v[I] / w_v[I]
    end
    return foo
end


# Generalized Eigenvalue problem: gives correct values
#=
# Use SL functions to calculate operator
function L1(x::Array, D::Matrix, DD::Matrix, pp, p, V, q::Float64)
    foo = Matrix{Complex{eltype(x)}}(undef, (length(x),length(x)))
    z = (1 .- x) ./ 2
    p_v = p(z)
    pp_v = pp(z)
    V_v = V(z, q)
    # Factors of (2) come from change of derivatives to z from x
    @views ThreadsX.foreach(eachindex(x)) do i
        foo[i,:] = ((-2) * pp_v[i]) .* D[i,:] + ((-2)^2 * p_v[i] ) .* DD[i,:]
        foo[i,i] += V_v[i]
    end
    return foo
end

function L2(x::Array, D::Matrix)
    foo = Matrix{Complex{eltype(x)}}(undef, (length(x),length(x)))
    z = (1 .- x) ./ 2
    # Factors of (-2) come from change of derivatives to z from x
    @views ThreadsX.foreach(eachindex(x)) do i
        foo[i,:] = ((-2) * 2 * 1im * z[i]^2) .* D[i,:]
    end
    return foo
end
=#


######################
#= Condition number =#
######################

function condition(L::Matrix, G::Matrix, Ginv::Matrix)
    # Condition number for each eigenvalue: k_i = ||v_i|| ||w_i|| / |<v_i,w_i>|
    # Construct the adjoint from the Gram matrices
    Ladj = Ginv * adjoint(L) * G
    # Eigen is not available for BigFloat type; create temp versions if
    # current precision is higher than Float64
    if typeof(G) == Matrix{BigFloat}
        F_r = eigen(convert(Matrix{Complex{Float64}}, L)) # Right eigensystem
        F_l = eigen(convert(Matrix{Complex{Float64}}, Ladj)) # Left eigensystem
    else
        F_r = eigen(L) # Right eigensystem
        F_l = eigen(Ladj) # Left eigensystem
    end
    # Calculate the condition numbers of the eigenvalues
    k = Vector(undef, length(F_r.values))
    @inbounds @views ThreadsX.foreach(eachindex(k)) do i
        vsize = sqrt(F_r.vectors[:,i]' * G * F_r.vectors[:,i])
        wsize = sqrt(F_l.vectors[:,i]' * G * F_l.vectors[:,i])
        vw_prod = F_r.vectors[:,i]' * G * F_l.vectors[:,i]
        # Condition numbers are purely real; imaginary components are 
        # numerical error only
        k[i] = vsize * wsize / sqrt(real(vw_prod)^2 + imag(vw_prod)^2)
    end
    # Normalize
    return k ./ k[1]
end

##############################
#= Psuedospectrum functions =#
##############################

function make_Z(inputs::Inputs, x::Vector)
    Nsteps = inputs.xgrid
    xvals = Vector{eltype(x)}(undef, Nsteps+1)
    yvals = similar(xvals)
    dx = (inputs.xmax - inputs.xmin)/Nsteps
    dy = (inputs.ymax - inputs.ymin)/Nsteps
    xvals .= inputs.xmin .+ dx .* (0:Nsteps)
    yvals .= inputs.ymin .+ dy .* (0:Nsteps)
    # Meshgrid matrix
    foo = Matrix{Complex{eltype(x)}}(undef, (Nsteps+1, Nsteps+1))
    # Automatic load balancing, false sharing protection
    ThreadsX.foreach(Iterators.product(eachindex(xvals), eachindex(yvals))) do (i,j) # Index i is incremented first
        foo[i,j] = xvals[i] + yvals[j]*1im
    end
    return foo
end

#######################################
#= Incomplete Cholesky Factorization =#
#######################################


##################################################
#= Serial psuedospectrum for timing =#
##################################################

function serial_sigma(G::Matrix, Ginv::Matrix, Z::Matrix, L::Matrix)
    foo = similar(Z)
     # Include progress bar for long calculations
     p = Progress(length(Z), dt=0.1, desc="Computing pseudospectrum...", 
     barglyphs=BarGlyphs("[=> ]"), barlen=50)
     for i in 1:size(Z)[1]
        for j in 1:size(Z)[2]
            # Calculate the shifted matrix
            Lshift = L - Z[i,j] .* I
            # Calculate the adjoint
            Lshift_adj = Ginv * adjoint(Lshift) * G
            # Calculate the pseudospectrum
            foo[i,j] = minimum(GenericLinearAlgebra.svdvals(Lshift_adj * Lshift))
            next!(p)
        end
    end
    return foo
    finish!(p)
end

################################
#= Distributed pseudospectrum =#
################################

# Requires workers to have already been spawned
@everywhere function pspec(G::Matrix, Ginv::Matrix, Z::Matrix, L::Matrix)
    if nprocs() > 1
        # Calculate all the shifted matrices
        ndim = size(Z)[1]
        println("Constructing shifted matrices...")
        foo = pmap(i -> (L - Z[i] .* LinearAlgebra.I), eachindex(Z))
        # Apply svd to (Lshift)^\dagger Lshift
        println("Constructing adjoint products...")
        bar = pmap(x -> (Ginv * adjoint(x) * G) * x, foo)
        println("Calculating SVDs...")
        sig = pmap(GenericLinearAlgebra.svdvals!, bar)
        # Reshape and return sigma
        return reshape(minimum.(sig), (ndim, ndim))
    else
        println("No workers have been spawned");
        return 1
    end 
end

##########
#= Main =#
##########

if length(ARGS) != 1
    println("Usage:")
    println("julia -t M pspec.jl P")
    println("M (int): the number of tasks to launch in parallel regions")
    println("P (int): digits of precision for calculations - default is 64")
    println("NOTE: Requires the file 'Inputs.txt' to be in the current directory")
    println("")
    exit()
else
    P = parse(Int64, ARGS[1])
end

if nthreads() > 1
    println("Number of threads: ", nthreads())
end
if P > 64
    setprecision(P)
    println("Precision set to ", Base.precision(BigFloat), " bits")
end

# Read the inputs from a file
inputs = readInputs("./Inputs.txt")

# Compute the basis
x, D, DD = make_basis(inputs, P)

println("Constructing the operator...")
# Remove first row & column from each matrix
nrows, ncols = size(D)
println("N = ", inputs.N)
println("x: ", size(x))
println("D: ", size(D))
L_up = reduce(hcat, [view(zeros(eltype(x), size(D)), 1:nrows-1, 1:ncols-1), view(diagm(ones(eltype(x), length(x))), 1:nrows-1, 1:ncols-1)])
L_down = reduce(hcat, [view(L1(x,D,DD, slf.p, slf.pp, slf.V, slf.w, inputs), 1:nrows-1, 1:ncols-1), view(L2(x,D, slf.gamma, slf.gammap, slf.w), 1:nrows-1, 1:ncols-1)])
BigL = 1im .* vcat(L_up, L_down)

# Debug
if debug > 1
    println(""); print("Collocation points = ", size(x), " "); show(x); println("")
    println(""); print("D = ", size(D), " "); show(D); println("")
    println(""); print("DD = ", size(DD), " "); show(DD); println("")
    println(""); print("Lup = ", size(L_up), " "); show(L_up); println("")
    println(""); print("Llow = ", size(L_down), " "); show(L_down); println("")
    println(""); print("L = ", size(BigL), " "); show(BigL); println("")
    #println(""); print("G = ", size(G), " "); show(G); println("")
end

println("Computing eigenvalues...")
#vals = ThreadsX.sort!(GenericLinearAlgebra.eigvals!(copy(BigL)), alg=ThreadsX.StableQuickSort, by = x -> sqrt(real(x)^2 + imag(x)^2))
println("Done!")
#println(vals)
# Write eigenvalues to file
#io.writeData(vals, inputs.m, inputs.q)
#exit()
# Generalized eigenvalue problem
#=
println("Done!") 
# Construct operator matrix
function V(x::Array, l::Int64)
    theta = (pi / 4) .* (x .+ 1)
    foo = Vector{eltype(x)}(undef, length(x))
    ThreadsX.map!(i -> sec(theta[i])^2 * (2 + l * (l + 1) / (tan(theta[i]))^2), foo, eachindex(x))
    return foo
end
println("Computing eigenvalues...")
vals = ThreadsX.sort!(GenericLinearAlgebra.eigvals!(L1(x, D, DD, slf.pp, slf.p, slf.V, inputs.q), L2(x, D)), alg=ThreadsX.StableQuickSort, by = x -> sqrt(real(x)^2 + imag(x)^2))
println("Done!")
print("Eigenvalues = "); show(vals); println("")
# Write eigenvalues to file
io.writeData(vals, inputs.m, inputs.q)
=#

# Copy of basis at double spectral resolution
inputs2 = deepcopy(inputs)
inputs2.N = 2 * inputs.N
y, Dy, DDy = make_basis(inputs2, P)

# Construct the Gram matrix: compute at double resolution, then 
# interpolate down and finally remove rows and columns corresponding
# to rho = 1
println("Iterpolating integrals...")
G, F = quad.Gram(x, D, y, Dy, inputs.m, inputs.q)
println("Done!")

##################################
#= Calculate the Psuedospectrum =#
##################################

# Make the rescaled operator corresponding to the rescaled 
# energy norm

function L1_tilde(x::Array, D::Matrix, DD::Matrix, m::Float64, q::Float64)
    foo = Matrix{eltype(x)}(undef, size(D))
    rho = (1/2) .- (x ./ 2)
    # From direct calculation of the rescaled operator
    @views ThreadsX.foreach(eachindex(x)) do I
        foo[I,:] = ((-4 * 20*rho[I] - 30*rho[I]^2 + 20*rho[I]^3 - 5*rho[I]^4) * (-2) / ((rho[I] - 1) * (1 + (rho[I] - 1)^4))) .* D[I,:] + ((-2)^2 * (1 - (rho[I] - 1)^4) / (1 + (rho[I] - 1)^4)) .* DD[I,:]
        foo[I,I] -= (m^2 + q^2 * (rho[I] - 1)^2 + 4 * (1 + (rho[I] - 1)^4)) / ((rho[I]-1)^2 * (1 + (rho[I] - 1)^4))
    end
    return foo
end

function L2_tilde(x::Array, D::Matrix)
    foo = Matrix{eltype(x)}(undef, size(D))
    rho = (1/2) .- (x ./ 2)
    # From direct calculation of the rescaled operator
    @views ThreadsX.foreach(eachindex(x)) do I
        foo[I,:] = (2 * (-2) * (rho[I] - 1)^4 / (1 + (rho[I] - 1)^4)) .* D[I,:]
        foo[I,I] -= 5 * (1 - rho[I])^3 / (1 + (rho[I] - 1)^4)
    end
    return foo
end


#println(L1_tilde(x,D,DD,inputs.m, inputs.q))
#println(L2_tilde(x,D))

# Remove first row & column from each matrix
#=
nrows, ncols = size(D)
Ldown_tilde = reduce(hcat, [view(L1_tilde(x,D,DD,inputs.m, inputs.q), 1:nrows-1, 1:ncols-1), view(L2_tilde(x,D), 1:nrows-1, 1:ncols-1)])

BigL_tilde = 1im .* vcat(L_up, Ldown_tilde)
=#
#print("BigL_tilde: ", size(BigL_tilde), " "); show(BigL_tilde); println("")

# Make the meshgrid
Z = make_Z(inputs, x)

# Calculate the sigma matrix. Rough benchmarking favours multiprocessor
# methods if N > 50 and grid > 10
println("Computing the psuedospectrum...")
sig = gpusvd.pspec(Z, copy(F * BigL * inv(F)))
#sig = gpusvd.pspec(Z, copy(F * BigL_tilde * inv(F)))

# Debug
if debug > 2
    ssig = serial_sigma(G, Ginv, Z, BigL)
    print("Parallel/Serial calculation match: "); println(isapprox(ssig, sig))
end

# Write Psuedospectrum to file (x vector for data type)
io.writeData(sig, x, inputs.m, inputs.q, inputs)
println("Done!")
#print("Sigma = "); show(sig); println("")
println("Smallest sigma: ", minimum(abs, sig))


# Calculate the condition numbers of the eigenvalues
#=
println("Calculating conditition numbers...")
k = condition(L, G, Ginv)
println("Done! Condition numbers = ", k)
io.writeCondition(k)
=#


# Debug/timing
if debug > 0
    #=
    print("Ginv * G = I: "); println(isapprox(Ginv * G, I))
    # Serial Timing
    println("Timing for serial sigma:")
    @btime serial_sigma(G, Ginv, Z, BigL) 
    # Large, block matrix
    println("Timing for gpusvd.sigma:")
    @btime gpusvd.sigma(G, Ginv, Z, BigL)
    # Threaded over individual shifted matrices
    println("Timing for gpusvd.pspec:")
    @btime gpusvd.pspec(G, Ginv, Z, BigL)
    # Multiproc method
    addprocs(nthreads())
    @everywhere using GenericLinearAlgebra
    println("Timing for Distributed pspec:")
    @btime pspec(G, Ginv, Z, BigL)
    rmprocs(workers())
    =#
end



# Add a perturbation to the potential with a specified magnitude

# Example: Gaussian distribution
#     dV = [exp(-0.5 * x[i] * x[i]) / sqrt(2 * pi) for i in eachindex(x)]
# Example: Random values
#     Random.seed!(1234)
#     dV = Vector{eltype(x)}(rand(length(x)))


#=
dV = cos.((2*pi*50) .* x)
epsilon = 1e-3
pert.vpert(epsilon, dV, slf.w, x, G, Ginv, BigL)

=#