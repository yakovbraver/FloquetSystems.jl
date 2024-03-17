# A driving script for autonomous (non-interactive) calculation. Launch as e.g.
#   $ julia --project -e 'include("examples/driver.jl")' -- f ω Umin Umax N nprocs sort
using DelimitedFiles, FloquetSystems

f, ω, Umin, Umax = parse.(Float32, ARGS[1:4])
N, sort = parse.(Int, ARGS[5:6])

lattice = Lattice(;dims=(1, 6), isperiodic=true)

nstates = length(lattice.basis_states)
ε = Matrix{Float32}(undef, nstates+1, N) # the first row of `ε` will contain `Us`
sp = Matrix{Int}(undef, nstates, N) # sorting matrix

rm("f$(f)_w$(ω)_U$(Umin)-$(Umax)_1x5-exact", recursive=true) # directory of warm-up calculation

basename = "f$(f)_w$(ω)_U$(Umin)-$(Umax)_$(lattice.dims[1])x$(lattice.dims[2])-exact"
cd(basename)

for i in 1:N
    if sort == 1
        temp = readdlm("$i.txt")
        ε[:, i] = temp[:, 1]
        sp[:, i] = temp[2:end, 2]
    else
        ε[:, i] = readdlm("$i.txt")
    end
end

open(basename*".txt", "w") do io
    writedlm(io, ε)
end

if sort == 1
    open(basename*"-perm.txt", "w") do io
        writedlm(io, sp)
    end
end