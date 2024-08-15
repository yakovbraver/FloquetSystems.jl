# Simple benchmarking script used to obtain the results in 10.1103/PhysRevA.110.023315 (arXiv:2405.01383), Appendix B
# Tested with 1 Julia thread, 8 BLAS threads

using FloquetSystems

J = 1.0f0
f = 2
ω = 20.0
Us = [2ω/3]

for N = [5; 5:10] # first iteration is for compiling
    lattice = Lattice(;dims=(1, N), isperiodic=true)
    if N < 9
        bh = BoseHamiltonian(lattice, J, 1, f, ω)
        GC.gc()
        @time "exact, N=$N" quasienergy(bh, Us, sort=false, showprogress=false)
    end

    bh = BoseHamiltonian(lattice, J, 1, f, ω; r=0//1, order=3, type=:edpt)
    GC.gc()
    @time "EDPT , N=$N" edpt(bh, Us; sort=false, showprogress=false)

    bh = BoseHamiltonian(lattice, J, 1, f, ω; r=2//3, order=3, type=:dpt)
    GC.gc()
    @time "DPT  , N=$N" dpt(bh, Us; sort=false, showprogress=false)
    println()
end