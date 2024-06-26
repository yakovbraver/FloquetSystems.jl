# A driving script for non-interactive calculation of the exact spectrum. Launch in multiprocess mode as
#   $ julia --project --check-bounds=no -p 8 -e 'include("examples/driver-exact.jl")' -- f ω Umin Umax N
# where `N` is the number of Us to scan.
# Lattice size, diffeq tolerance, and whether sorting should be used are not exposed as the input arguments and should be set in the code below.

using FloquetSystems, DelimitedFiles

f, ω, Umin, Umax = parse.(Float32, ARGS[1:4])
N = parse.(Int, ARGS[5])

Us = range(Umin, Umax, N)

J = 1.0f0
U = 1
sort = true
reltol = 1e-3

# warm up
lattice = Lattice(;dims=(1, 5), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U, f, ω)
outdir = "f$(f)_w$(ω)_U$(Umin)-$(Umax)_$(lattice.dims[1])x$(lattice.dims[2])-exact"

quasienergy(bh, Us; sort, reltol);

# actual calculation
lattice = Lattice(;dims=(2, 4), isperiodic=true)
outdir = "f$(f)_w$(ω)_U$(Umin)-$(Umax)_$(lattice.dims[1])x$(lattice.dims[2])-exact"
bh = BoseHamiltonian(lattice, J, U, f, ω)

ε, sp = quasienergy(bh, Us; sort, reltol);

open(outdir*".txt", "w") do io
    writedlm(io, vcat(Us', ε))
end

sort && open(outdir*"-perm.txt", "w") do io
    writedlm(io, sp)
end