# A driving script for autonomous (non-interactive) calculation. Launch as e.g.
#   $ julia --project --check-bounds=no -p 8 -e 'include("examples/driver.jl")' -- f ω Umin Umax N
# where `N` is the number of Us to scan
@everywhere using FloquetSystems, SharedArrays
using LinearAlgebra, DelimitedFiles

BLAS.set_num_threads(1)

f, ω, Umin, Umax = parse.(Float32, ARGS[1:4])
N = parse.(Int, ARGS[5])
np = nprocs()-1

# using ThreadPinning
# pinthreads([pid-1])

Us = range(Umin, Umax, N)

J = 1.0f0
U = 1

# warm up
lattice = Lattice(;dims=(1, 5), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U, f, ω)
outdir = "f$(f)_w$(ω)_U$(Umin)-$(Umax)_$(lattice.dims[1])x$(lattice.dims[2])-exact"

nstates = length(lattice.basis_states)
E = SharedArray{typeof(J), 2}((nstates, N))
SP = SharedArray{Int, 2}((nstates, N))

@sync @distributed for pid in 1:np
    Umask = pid:np:N
    quasienergy(bh, Us; showprogress=false, sort=true, Umask, outdir, E, SP);
end

# actual calculation
lattice = Lattice(;dims=(1, 6), isperiodic=true)
outdir = "f$(f)_w$(ω)_U$(Umin)-$(Umax)_$(lattice.dims[1])x$(lattice.dims[2])-exact"
bh = BoseHamiltonian(lattice, J, U, f, ω)

nstates = length(lattice.basis_states)
E = SharedArray{typeof(J), 2}((nstates, N)) # the first row of `ε` will contain `Us`, hence +1
SP = SharedArray{Int, 2}((nstates, N))

@sync @distributed for pid in 1:np
    Umask = pid:np:N
    quasienergy(bh, Us; showprogress=false, sort=true, Umask, outdir, E, SP);
end

open(outdir*".txt", "w") do io
    writedlm(io, vcat(Us', E))
end

open(outdir*"-perm.txt", "w") do io
    writedlm(io, SP)
end