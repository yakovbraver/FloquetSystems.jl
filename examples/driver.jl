# A driving script for autonomous (non-interactive) calculation. Launch as e.g.
#   $ julia --project -t 8 -e 'include("examples/driver.jl")' &
using FloquetSystems, DelimitedFiles

J = 1.0f0
ω = 20
U = 1
f = 2
Us = range(12, 15, 24)

# warm up
lattice = Lattice(;dims=(1, 6), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U, f, ω)
quasienergy(bh, Us; order=true);

# actual calculation
lattice = Lattice(;dims=(2, 4), isperiodic=true)
outdir = "f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])"
bh = BoseHamiltonian(lattice, J, U, f, ω)
ε, sp = quasienergy(bh, Us; order=true, outdir);

open(joinpath(outdir, outdir*"-exact.txt"), "w") do io
    writedlm(io, vcat(Us', ε))
end

open(joinpath(outdir, outdir*"-exact-perm.txt"), "w") do io
    writedlm(io, sp)
end