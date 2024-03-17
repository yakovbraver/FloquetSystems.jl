# A driving script for autonomous (non-interactive) calculation. Launch as e.g.
#   $ julia --project --check-bounds=no -t 21 -e 'include("examples/driver.jl")' -- f ω Umin Umax N
using FloquetSystems, ThreadPinning, DelimitedFiles
pinthreads(:cores)

J = 1.0f0
U = 1

f, ω, Umin, Umax = parse.(Float32, ARGS[1:4])
N = parse.(Int, ARGS[5])

Us = range(Umin, Umax, N)

# warm up
lattice = Lattice(;dims=(1, 5), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U, f, ω)
outdir = "2x4/f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])-exact"
quasienergy(bh, Us; showprogress=false, gctrick=true, sort=true, outdir);
rm(outdir, recursive=true) # we don't need those file, but the `outdir` argument was used to compile the function

# actual calculation
lattice = Lattice(;dims=(2, 4), isperiodic=true)
outdir = "2x4/f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])-exact"
bh = BoseHamiltonian(lattice, J, U, f, ω)
GC.gc()
ε, sp = quasienergy(bh, Us; showprogress=false, gctrick=true, sort=true, outdir);

open(outdir*".txt", "w") do io
    writedlm(io, vcat(Us', ε))
end

open(outdir*"-perm.txt", "w") do io
    writedlm(io, sp)
end