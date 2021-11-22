include("hamiltonian.jl")

using SparseArrays, KrylovKit
using Plots, LaTeXStrings
pyplot()
theme(:dark, size=(600, 600))

"""
Plot occupations of each lattice cell in a state `state`, which is a superposition of the basis states of `bh`.
A rectangular lattice is assumed.
"""
function plotstate(bh::BoseHamiltonian, state::Vector{<:Number}, ε::Float64)
    ncells = prod(bh.lattice.dims)
    final_state = zeros(ncells)
    for i in eachindex(state)
        final_state .+= abs2(state[i]) .* bh.basis_states[i]
    end
    final_state ./= bh.lattice.nbozons # normalise to unity (otherwise normalised to `nbozons`)
    x, y = 1:bh.lattice.dims[2], 1:bh.lattice.dims[1]
    state_matrix = reshape(final_state, bh.lattice.dims[1], bh.lattice.dims[2])
    fig = heatmap(x, y, state_matrix, xticks=x, yticks=y, yflip=true, color=:viridis)
    title!(L"\varepsilon = %$(round(ε, sigdigits=6))" * "; fluxes are in units of pi")
    
    defects = findall(bh.lattice.is_defect)
    defects_rows = [(cell-1) % bh.lattice.dims[1] + 1 for cell in defects]
    defects_cols = [(cell-1) ÷ bh.lattice.dims[1] + 1 for cell in defects]
    scatter!(defects_cols, defects_rows, color=:white, markersize=5, label="defect")

    for row in 1:bh.lattice.dims[1]
        for col in 1:bh.lattice.dims[2]
            flux = get_flux(bh.lattice, (row, col)) / π
            if abs(flux) > 1e-3
                f = rationalize(flux, tol=1e-5)
                n, d = numerator(f), denominator(f)
                annotate!([(col + 0.5, row + 0.5, (L"\frac{%$n}{%$d}", :white, 8))]) # default size 16
            end
        end
    end
    display(fig)

    state_matrix
end

#-------
nbozons = 1
lattice3 = Lattice(dims=(6, 6), J_default=1, periodic=true, nϕ=2, driving_type=:linear; nbozons)
bh = BoseHamiltonian(lattice3)
add_defects!(bh, [11,16,21,26,31])
bh.lattice.J[6, 1]

vals, vecs, info = eigsolve(bh.H, 1, :SR)
plotstate(bh, vecs[1], vals[1])

#-------
nbozons = 1
# lattice35 = Lattice(dims=(3, 5), J_default=1, periodic=false, Δϕ=[π/2, π/2]; nbozons)
lattice35 = Lattice(dims=(3, 3), J_default=1, periodic=false, Δϕ=[π/3, π/3]; nbozons)
bh = BoseHamiltonian(lattice35)
add_defects!(bh, [5, 8])
get_flux(bh.lattice, (1, 1))
get_flux(bh.lattice, (2, 1))
get_flux(bh.lattice, (1, 2))
get_flux(bh.lattice, (2, 2))

#-------
nbozons = 1
lattice35 = Lattice(dims=(35, 35), J_default=1, periodic=true; nbozons)
bh = BoseHamiltonian(lattice35)
add_defects!(bh, collect(range(103, length=4, step=34)))

vals, vecs, info = eigsolve(bh.H, 1, :SR)

fs = plotstate(bh, vecs[1], vals[1])
#------
include("optimise.jl")
ndefects = 6
nbozons = 1
lattice6 = Lattice(dims=(8, 8), J_default=1, periodic=true, nϕ=2; nbozons)
bh = BoseHamiltonian(lattice6)
add_defects!(bh, [13,12,19,20,27,29])
# move_defects!(bh, [15], [21])
best_defects, best_val = optimise_defects(bh, ndefects)
move_defects!(bh, findall(bh.lattice.is_defect), best_defects)
vals, vecs, info = eigsolve(bh.H, 1, :SR)

fs = plotstate(bh, vecs[1], vals[1])
savefig("$(ndefects).pdf")
#------
"Repeat optimisation `niter` times and return the best result as a tuple `(defects, value)`."
function search(lattice, ndefects, niter=20)
    best_defects = Vector{Int}(undef, ndefects)
    best_val = 10.0
    defects = Vector{Int}(undef, ndefects)
    val = 0.0
    for _ in 1:niter
        lat = deepcopy(lattice)
        bh = BoseHamiltonian(lat)
        defects, val = optimise_defects(bh, ndefects)
        if val < best_val
            best_val = val
            best_defects = copy(defects)
        end
    end
    (best_defects, best_val)
end

ndefects = 3
nbozons = 1
lattice = Lattice(dims=(20, 20), J_default=1, periodic=true, nϕ=1, driving_type=:linear; nbozons)
best_defects, best_val = search(lattice, ndefects, 10)
bh = BoseHamiltonian(lattice)
add_defects!(bh, [1,3,5,7])
add_defects!(bh, best_defects)

vals, vecs, info = eigsolve(bh.H, 1, :SR)

fs = plotstate(bh, vecs[1], vals[1])
savefig("$(ndefects)_optimal.pdf")


ndefects = 3
nbozons = 1

su = 0
# for i = 1:20
    lattice = Lattice(dims=(20, 20), J_default=1, periodic=true, nϕ=1, driving_type=:linear; nbozons)
    bh = BoseHamiltonian(lattice)
    result = optimise_defects(bh, ndefects)
    su += result[1]
# end
print(su/20)
# nf = 1:
# 0.6 0.8   -3.8992585335206407
# 0.6 0.2   -3.9051314002712254
# 0.5 0.2   -3.91085869948141 -- 100 % accuracy

# nf = 2
# 0.7 0.7    -3.9147032344709567
# 0.7 0.2    -3.918400850397906
# 0.5 0.2    -3.9200123055358653
# 0.5 0.8    -3.9167476145633726
# 0.4 0.8    -3.9204495471432046 
# 0.4 0.5    -3.9202588817726394
# 0.3 0.8    -3.922184444366672  -- 
# 0.3 0.5    -3.9245450855090156 -- 
# 0.3 0.2    -3.927662912334796  -- 
# 0.3 0.1    -3.9259474499460736 -- 
# 0.3 0.01   -3.9112211732734985
# 0.2 0.2    -3.9264654644298176 -- 
# 0.1 0.2    -3.9255605473931903 -- 
# 0.1 0.8    -3.9309900496129657 -- 2 
# 0.1 0.9    -3.931894966649596 -- 1
# 0.1 0.95   -3.9309900496129644 -- 2

println(result[1])
println(result[3])
println(result[4])

old_defects = findall(bh.lattice.is_defect)
new_defects = ceil.(Int, result[2])
dublicates = intersect(old_defects, new_defects)
move_defects!(bh, setdiff(old_defects, dublicates), setdiff(new_defects, dublicates))

vals, vecs, info = eigsolve(bh.H, 1, :SR)

fs = plotstate(bh, vecs[1], vals[1])
savefig("$(ndefects)_optimal.pdf")

lattice = Lattice(dims=(6, 6), J_default=1, periodic=true, nϕ=2, driving_type=:linear; nbozons)
bh = BoseHamiltonian(lattice)
add_defects!(bh, [11,10,16,15,8,21,20,26])