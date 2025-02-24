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


####################
#= Inputs =#
####################

@with_kw mutable struct Inputs
    N::Int64 = 4
    l::Int64 = 0
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
        elseif k == "l"
            inpts.l = parse(Int64, get(data, k, nothing))
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
            # Don't need to parse strings
            inpts.basis = get(data, k, nothing)
            # If none of the inputs are chosen, default to interior
            if !(occursin(inpts.basis, "GC GL LGR RGR")) || typeof(inpts.basis) == Nothing
                inpts.basis = "GC"
            end
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
    println("the number of grip points is", inputs.N)
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
z = x ./ 2 .+ (1/2)


println("Constructing the operator...")

##########changed############
L2=view(diagm(4. *z)* D.+5*diagm(ones(eltype(x), length(x))), 1:length(x), 1:length(x))
#LHSup = reduce(hcat, [L2, zeros(eltype(x),(length(x),length(x)))])
#LHSlow = reduce(hcat, [zeros(eltype(x), (length(x),length(x))),L2])
#LHS = vcat(LHSup, LHSlow)
LHS=1im .* L2

####changed####
L1=view(diagm(ThreadsX.map(i ->16*z[i]^3,eachindex(z)))+diagm(ThreadsX.map(i ->2*(9* z[i]^4-5),eachindex(z)))*D+diagm(ThreadsX.map(i ->4*z[i]*(z[i]^4-1),eachindex(z)))*DD, 1:length(x), 1:length(x))
#Lup = reduce(hcat, [L1,zeros(eltype(x), (length(x),length(x)))])
#Llow = reduce(hcat, [zeros(eltype(x),(length(x),length(x))), L1])
#BigL = 1im .* vcat(Lup, Llow)
BigL =  L1
#println(""); print("BigL ", size(BigL), " = "); show(BigL); println("")
println("Done!")

#println(D)
#println("L2")
#println(LHS)
#println("L1")
#println(BigL)

#exit()
println("Computing eigenvalues...")
vals = eigen(BigL, LHS)
println("Done! ", ThreadsX.sort!(vals.values, alg=ThreadsX.StableQuickSort, by = x -> sqrt(real(x)^2 + imag(x)^2)))


#print("Eigenvalues = "); show(vals); println("")
# Write eigenvalues to file
io.writeData(vals.values, inputs.N)


#println(size(BigL))
#println(size(LHS))

# Copy of basis at double spectral resolution
inputs2 = deepcopy(inputs)
inputs2.N = 2 * inputs.N
y, Dy, DDy = make_basis(inputs2, P)

#println(size(x),",",size(y))
# Construct the Gram matrix: compute at double resolution, then 
# interpolate down and finally remove rows and columns corresponding
# to z = 0
println("Iterpolating integrals...")
G, F = quad.Gram(x, D, y, Dy)
println("Done!")


##################################
#= Calculate the Pseudospectrum =#
##################################

# Make the meshgrid
Z = make_Z(inputs, x)

# Calculate the sigma matrix. Rough benchmarking favours multiprocessor
# methods if N > 50 and grid > 10
println("Computing the psuedospectrum...")

println(size(Z))

sig = gpusvd.pspec(Z, copy(F * BigL * inv(F)),copy(F * LHS * inv(F)))

# Write Psuedospectrum to file (x vector for data type)
io.writeData(sig, x, inputs)
println("Done!")
#print("Sigma = "); show(sig); println("")
println("Smallest sigma: ", minimum(abs, sig))

exit()
