# A driving script for non-interactive calculation of dpt/edpt spectrum. Launch in multithreaded mode as
#   $ julia --project --check-bounds=no -t 64 -e 'include("examples/driver-dpt.jl")' -- f ω Umin Umax N order n d
# where `N` is the number of Us to scan, and `order` is dpt/edpt order (1, 2, or 3).
# If two last arguments are not passed, then EDPT is used.
# If `n` and `d` are passed, then DPT is used with resonance number `n/d`.
# Lattice size and whether sorting should be used are not exposed as the input arguments and should be set in the code below.

using FloquetSystems, DelimitedFiles
# using ThreadPinning
# pinthreads(:cores)

J = 1.0f0
U = 1
sort = true

f, ω, Umin, Umax = parse.(Float32, ARGS[1:4])
N, order = parse.(Int, ARGS[5:6])
if length(ARGS) == 6
    type = :edpt
    r = 0//1
else
    type = :dpt
    r = parse(Int, ARGS[7]) // parse(Int, ARGS[8])
end

Us = range(Umin, Umax, N)

# warm up
lattice = Lattice(;dims=(1, 5), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U, f, ω; r, order, type)
if type == :edpt
    edpt(bh, Us; sort, showprogress=false)
else
    dpt(bh, Us; sort, showprogress=false)
end

# actual calculation
lattice = Lattice(;dims=(1, 6), isperiodic=true)
bh = BoseHamiltonian(lattice, J, U, f, ω; r, order, type)
GC.gc()
if type == :edpt
    ε, sp = edpt(bh, Us; sort, showprogress=false)
    us = Us
else
    E, SP = dpt(bh, Us; sort, showprogress=false)
    if findfirst(isnan, @view(E[1, :])) !== nothing
        mask = @. !isnan(@view(E[1, :]))
        ε = @view E[:, mask]
        sp = @view SP[:, mask]
        us = @view Us[mask]
    else
        ε, sp, us = E, SP, Us
    end
end

basename = "f$(f)_w$(ω)_U$(Umin)-$(Umax)_$(lattice.dims[1])x$(lattice.dims[2])-$(type)$(order)"
open(basename*".txt", "w") do io
    writedlm(io, vcat(us', ε))
end

sort && open(basename*"-perm.txt", "w") do io
    writedlm(io, sp)
end