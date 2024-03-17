# A driving script for autonomous (non-interactive) calculation. Launch as e.g.
#   $ julia --project --check-bounds=no -t 1 -e 'include("examples/driver.jl")' -- f ω Umin Umax N nprocs pid
# where `N` is the number of Us to scan, `nprocs` is the total number of processes among which work is split, and `pid` is the ID of the current process, starting from 1
using FloquetSystems, LinearAlgebra, DelimitedFiles

BLAS.set_num_threads(1)

f, ω, Umin, Umax = parse.(Float32, ARGS[1:4])
N, nprocs, pid = parse.(Int, ARGS[5:7])

# using ThreadPinning
# pinthreads([pid-1])

Umask = pid:nprocs:N
Us = range(Umin, Umax, N)

J = 1.0f0
U = 1

# warm up
lattice = Lattice(;dims=(1, 5), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U, f, ω)
outdir = "f$(f)_w$(ω)_U$(Umin)-$(Umax)_$(lattice.dims[1])x$(lattice.dims[2])-exact"
quasienergy(bh, Us; showprogress=false, sort=true, Umask, outdir);

# actual calculation
lattice = Lattice(;dims=(1, 6), isperiodic=true)
outdir = "f$(f)_w$(ω)_U$(Umin)-$(Umax)_$(lattice.dims[1])x$(lattice.dims[2])-exact"
bh = BoseHamiltonian(lattice, J, U, f, ω)
quasienergy(bh, Us; showprogress=false, sort=true, Umask, outdir);