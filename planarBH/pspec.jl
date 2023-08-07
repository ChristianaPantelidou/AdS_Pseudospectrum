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
        x = Vector{BigFloat}(undef, inputs.N+1)
    else
        x = Vector{Float64}(undef, inputs.N+1)
    end

    # Reference the data type of the collocation vector 
    # for the other matrices
    D = Matrix{eltype(x)}(undef, (length(x), length(x)))
    DD = similar(D)

    println("Using the ", inputs.basis, " collocation grid.")
    # Algorithms for different collocation sets
    if inputs.basis == "GC"
        n = length(x)
        # Collocation points
        ThreadsX.map!(i -> cos(pi * (i + 0.5) / n), x, 0:n-1)
        # First derivative matrix
        ThreadsX.foreach(Iterators.product(1:n, 1:n)) do (i,j)
            if i != j
                D[i,j] = (-1)^(i+j) * sqrt((1 - x[j] * x[j]) /
                (1 - x[i]^2)) / (x[i] - x[j])
            else
                D[i,i] = x[i] / (2*(1 - x[i]^2))
            end
        end
        # Second derivative matrix
        ThreadsX.foreach(Iterators.product(1:n, 1:n)) do (i,j)
            if i != j
                DD[i,j] = D[i,j] * (x[i] / (1 - x[i] * x[i]) - 2 / (x[i] - x[j]))
            else
                DD[j,j] = x[j] * x[j] / (1 - x[j] * x[j])^2 - (n * n - 1) / (3 * (1. - x[j] * x[j]))
            end
        end
        return x, D, DD

    elseif inputs.basis::String == "GL"
        # Size of abscissa is N + 1
        n = length(x)
        N = length(x) - 1
        # Collocation points
        ThreadsX.map!(i-> cos(pi*i / N), x, 0:N)
        kappa = [i == 1 ? 2 : i == n ? 2 : 1 for i in eachindex(x)]
        # First derivative matrix
        ThreadsX.foreach(Iterators.product(eachindex(x), eachindex(x))) do (i,j)
            if i == j
                if i == 1 && j == 1
                    D[i,j] = (2 * N^2 + 1) / 6
                elseif i == n && j == n
                    D[i,j] = -(2 * N^2 + 1) / 6
                else
                    D[i,i] = -x[i] / (2*(1 - x[i]^2))
                end
            else
                D[i,j] = kappa[i] * (-1)^(i-j) / (kappa[j] * (x[i] - x[j]))
            end
        end
        # Second derivative matrix
        ThreadsX.foreach(Iterators.product(1:n, 1:n)) do (i,j)
            if i == j
                if i == 1 && j == 1
                    DD[i,i] = (N^4 - 1) / 15
                elseif i == n && j == n
                    DD[i,i] = (N^4 - 1) / 15
                else
                    DD[i,i] = -1 / (1 - x[i]^2)^2 - (N^2 - 1) / (3 * (1 - x[i]^2))
                end
            elseif i == 1 && j != 1
                DD[i,j] = 2 * (-1)^(j-1) * ((2*N^2 + 1) * (1 - x[j]) - 6) / (3 * kappa[j] * (1 - x[j])^2)
            elseif i == n && j != n
                DD[i,j] = 2 * (-1)^(N+j-1) * ((2*N^2 + 1) * (1 + x[j]) - 6) / (3 * kappa[j] * (1 + x[j])^2)
            else
                DD[i,j] = (-1)^(i-j) * (x[i]^2 + x[i]*x[j] - 2) / 
                            (kappa[j] * (1 - x[i]^2) * (x[j] - x[i])^2)
            end
        end
        return x, D, DD

    elseif inputs.basis::String == "LGR"
        # Size of abscissa is N + 1
        n = length(x) - 1
        mrange = [pi*(2*n + 1 - 2*i) / (2*n + 1) for i in 0:n]
        # Collocation points
        x = @tturbo @. cos(mrange)
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

# Use SL functions to calculate operator
function L1(x::Array, D::Matrix, DD::Matrix, pp, p, V, w, q::Float64, m::Float64)
    foo = Matrix{Complex{eltype(x)}}(undef, (length(x),length(x)))
    p_v = p(x)
    pp_v = pp(x)
    V_v = V(x, m, q)
    w_v = w(x)
    # Factors of (2) come from change of derivatives to rho from x
    @views ThreadsX.foreach(eachindex(x)) do i
        foo[i,:] = (pp_v[i] / ((2) * w_v[i])) .* D[i,:] + (p_v[i] / ((2)^2 * w_v[i])).* DD[i,:]
        foo[i,i] -= V_v[i] / w_v[i]
    end
    return foo
end

function L2(x::Array, D::Matrix, gamma, gammap, w)
    g_v = gamma(x)
    gp_v = gammap(x)
    w_v = w(x)
    foo = Matrix{Complex{eltype(x)}}(undef, (length(x),length(x)))
    # Factors of (2) come from change of derivatives to rho from x
    @views ThreadsX.foreach(eachindex(x)) do i
        foo[i,:] = (2 * g_v[i] / ((2) * w_v[i])) .* D[i,:]
        foo[i,i] += gp_v[i] / w_v[i]
    end
    return foo
end

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

function myCholesky(A::Matrix)
    out = similar(A)
    N = size(A,1)
    for i in 1:N
        for j in i:N
            mysum = A[i,j]
            if i > 1
                mysum -= ThreadsX.sum(A[i,k] * A[j,k] for k in i-1:-1:1)
            end
            if i == j
                mysum <= 0 ? println("Cholesky iteration failed.") : nothing
                out[i,i] = sqrt(mysum)
            else
                out[i,j] = mysum / out[i,i]
            end
        end
    end
    ThreadsX.foreach(eachindex(out)) do I
        if abs(out[I]) < 10E-300
            out[I] = 0
        end
    end
    return out
end


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
rho = x ./2 .+ (1/2)

# Construct operator but remove columns and rows corresponding to rho=1 boundary
Lup = reduce(hcat, [zeros(eltype(x), (length(x)-1,length(x)-1)), view(diagm([(1 - rho[i])^(-2) for i in eachindex(rho)]), 2:length(x), 2:length(x))])
Llow = reduce(hcat, [view(L1(x,D,DD,slf.pp,slf.p,slf.V,slf.w,inputs.m,inputs.q), 2:length(x), 2:length(x)), view(L2(x,D,slf.gamma,slf.gammap,slf.w), 2:length(x), 2:length(x))])
BigL = 1im .* vcat(Lup, Llow)

vals = ThreadsX.sort!(GenericLinearAlgebra.eigvals!(copy(BigL)), alg=ThreadsX.StableQuickSort, by = x -> sqrt(real(x)^2 + imag(x)^2))
print("Done! Eigenvalues = "); show(vals); println("")
# Write eigenvalues to file
#io.writeData(vals, inputs.m, inputs.q)

# Copy of basis at double spectral resolution
inputs2 = deepcopy(inputs)
inputs2.N = 2 * inputs.N
y, Dy, DDy = make_basis(inputs2, P)

# Construct the Gram matrix: compute at double resolution, then 
# interpolate down and finally remove rows and columns corresponding
# to rho = 1
G = quad.Gram(x, D, y, Dy, inputs.m, inputs.q)
Ginv = inv(G)

# Cholesky factoring
F = myCholesky(G)

print("Cholesky factorization: "); show(F); println("")
print("F*F = G? ", LinearAlgebra.isapprox(F' * F, G)); show(F' * F - G); println("")


# Debug
if debug > 0
    println(""); print("Collocation points = ", size(x), " "); show(x); println("")
    println(""); print("Rho collocations = ", size(rho), " "); show(rho); println("")
    println(""); print("D = ", size(D), " "); show(D); println("")
    println(""); print("DD = ", size(DD), " "); show(DD); println("")
    println(""); print("Lup = ", size(Lup), " "); show(Lup); println("")
    println(""); print("Llow = ", size(Llow), " "); show(Llow); println("")
    println(""); print("L = ", size(BigL), " "); show(BigL); println("")
    println(""); print("G = ", size(G), " "); show(G); println("")
end

exit()

##################################
#= Calculate the Psuedospectrum =#
##################################

# Make the meshgrid
Z = make_Z(inputs, x)

# Calculate the sigma matrix. Rough benchmarking favours multiprocessor
# methods if N > 50 and grid > 10
println("Computing the psuedospectrum...")
sig = gpusvd.pspec(Z, L)

# Debug
if debug > 0
    ssig = serial_sigma(G, Ginv, Z, BigL)
    print("Parallel/Serial calculation match: "); println(isapprox(ssig, sig))
end

# Write Psuedospectrum to file (x vector for data type)
io.writeData(sig, x, inputs.m, inputs.q)
print("Done! Sigma = "); show(sig); println("")


# Calculate the condition numbers of the eigenvalues
#=
println("Calculating conditition numbers...")
k = condition(L, G, Ginv)
println("Done! Condition numbers = ", k)
io.writeCondition(k)
=#


# Debug/timing
if debug > 0
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
