# launch as e.g. 
#   $ julia --project -t 8 -e 'include("src/driver.jl")' &
using FloquetSystems, DelimitedFiles

J = 1.0f0
ω = 20
U = 1
f = 2
Us = range(12, 15, 24)

# warm up
lattice = Lattice(;dims=(1, 6), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U, f, ω)
quasienergy(bh, Us);

# actual calculation
outdir = "f$(f)_w$(ω)_U$(Us[1])-$(Us[end])_$(lattice.dims[1])x$(lattice.dims[2])"
lattice = Lattice(;dims=(2, 4), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U, f, ω)
ε = quasienergy(bh, Us; outdir, nthreads=8);

open(joinpath(outdir, "final.txt"), "w") do io
    writedlm(io, vcat(Us', ε))
end
