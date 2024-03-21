# A driving script for non-interactive calculation of dpt spectrum. Launch in multithreaded mode as
#   $ julia --project --check-bounds=no -t 64 -e 'include("examples/driver-dpt.jl")' -- f ω Umin Umax N order type
# where `N` is the number of Us to scan, `order` is dpt orded (1, 2, or 3), and `type` is `dpt` or `dpt_quick`.
using FloquetSystems, DelimitedFiles
# using ThreadPinning
# pinthreads(:cores)

J = 1.0f0
U = 1

f, ω, Umin, Umax = parse.(Float32, ARGS[1:4])
N, order = parse.(Int, ARGS[5:6])
type = Symbol(ARGS[7])

Us = range(Umin, Umax, N)
ωₗ = -ω/2

# warm up
lattice = Lattice(;dims=(1, 5), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U, f, ω, ωₗ; order, type)
dpt(bh, Us; sort=true, showprogress=false)

# actual calculation
lattice = Lattice(;dims=(2, 4), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U, f, ω, ωₗ; order, type)
GC.gc()
ε, sp = dpt(bh, Us; sort=true, showprogress=false)

basename = "f$(f)_w$(ω)_U$(Umin)-$(Umax)_$(lattice.dims[1])x$(lattice.dims[2])-$(type)$(order)"
open(basename*".txt", "w") do io
    writedlm(io, vcat(Us', ε))
end

open(basename*"-perm.txt", "w") do io
    writedlm(io, sp)
end