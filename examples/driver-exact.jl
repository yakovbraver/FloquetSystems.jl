# A driving script for non-interactive calculation of the exact spectrum. Launch in multiprocessed mode as
#   $ julia --project --check-bounds=no -p 8 -e 'include("examples/driver-exact.jl")' -- f ω Umin Umax N
# where `N` is the number of Us to scan.
using FloquetSystems, DelimitedFiles

f, ω, Umin, Umax = parse.(Float32, ARGS[1:4])
N = parse.(Int, ARGS[5])

Us = range(Umin, Umax, N)

J = 1.0f0
U = 1

# warm up
lattice = Lattice(;dims=(1, 5), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U, f, ω)
outdir = "f$(f)_w$(ω)_U$(Umin)-$(Umax)_$(lattice.dims[1])x$(lattice.dims[2])-exact"

quasienergy(bh, Us, sort=false, showprogress=true);

# actual calculation
lattice = Lattice(;dims=(2, 4), isperiodic=true)
outdir = "f$(f)_w$(ω)_U$(Umin)-$(Umax)_$(lattice.dims[1])x$(lattice.dims[2])-exact"
bh = BoseHamiltonian(lattice, J, U, f, ω)

nstates = length(lattice.basis_states)

ε, sp = quasienergy(bh, Us, sort=false, showprogress=true);

open(outdir*".txt", "w") do io
    writedlm(io, vcat(Us', ε))
end

open(outdir*"-perm.txt", "w") do io
    writedlm(io, sp)
end