module GaugeFields

export LightField,
    𝑈

mutable struct LightField
    ϵ::Real
    ϵc::Real
    χ::Real
end

function 𝑈(lf::LightField, xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
    (;ϵ, ϵc) = lf
    kᵣ = 2π #/ √2
    U = Matrix{Float64}(undef, length(xs), length(ys))
    for (iy, y) in enumerate(ys)
        for (ix, x) in enumerate(xs)
            β₋ = sin(kᵣ*(x-y)); β₊ = sin(kᵣ*(x+y))
            U[ix, iy] = (β₊^2 + (ϵc*β₋)^2) / 𝛼(lf, x, y)^2 * 2ϵ^2*(1+ϵc^2)
        end
    end
    return U
end

function 𝛼(lf::LightField, x::Real, y::Real)
    (;ϵ, ϵc, χ) = lf
    kᵣ = 2π #/ √2
    η₋ = cos(kᵣ*(x-y)); η₊ = cos(kᵣ*(x+y))
    return ϵ^2 * (1 + ϵc^2) + η₊^2 + (ϵc*η₋)^2 - 2ϵc*η₊*η₋*cos(χ)  
end

# η₊ = cos(kᵣ*x + kᵣ*y); η₋ = cos(kᵣ*x - kᵣ*y)
# β₊ = sin(kᵣ*x + kᵣ*y); β₋ = sin(kᵣ*x - kᵣ*y)

# function 𝑈(lf::LightField, xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
#     (;kᵣ) = lf
#     U = Matrix{Float64}(undef, length(xs), length(ys))
#     for (ix, x) in enumerate(xs)
#         for (iy, y) in enumerate(ys)
#             U[ix, iy] = abs2(∇𝜉ₓ(lf, x, y)) + abs2(∇𝜉y(lf, x, y)) / kᵣ / (1+abs2(𝜉(lf, x, y)))^2
#         end
#     end
#     return U
# end

# function 𝜉(lf::LightField, x::Real, y::Real)
#     (;kᵣ, χ, Ω₊, Ω₋, Ωₚ) = lf
#     return - Ω₋/Ωₚ * cis(χ/2)cos(kᵣ*x - kᵣ*y) + Ω₊/Ωₚ * cis(-χ/2)cos(kᵣ*x + kᵣ*y)
# end

# function ∇𝜉ₓ(lf::LightField, x::Real, y::Real)
#     (;kᵣ, χ, Ω₊, Ω₋, Ωₚ) = lf
#     return kᵣ * Ω₋/Ωₚ * cis(χ/2)sin(kᵣ*x - kᵣ*y) - kᵣ * Ω₊/Ωₚ * cis(-χ/2)sin(kᵣ*x + kᵣ*y)
# end

# function ∇𝜉y(lf::LightField, x::Real, y::Real)
#     (;kᵣ, χ, Ω₊, Ω₋, Ωₚ) = lf
#     return -kᵣ * Ω₋/Ωₚ * cis(χ/2)sin(kᵣ*x - kᵣ*y) - kᵣ * Ω₊/Ωₚ * cis(-χ/2)sin(kᵣ*x + kᵣ*y) 
# end

# A = im * (ξ' * ∇ξₓ - ξ * ∇ξₓ') / 2(1+abs2(ξ))

# η₊ = cos(kᵣ*x + kᵣ*y); η₋ = cos(kᵣ*x - kᵣ*y)
# β₊ = sin(kᵣ*x + kᵣ*y); β₋ = sin(kᵣ*x - kᵣ*y)
# ϵ = Ωₚ / √(Ω₊^2 + Ω₋^2)
# ϵc = Ω₋ / Ω₊
# α = ϵ^2 * (1 + ϵc^2) + η₊^2 + (ϵc*η₋)^2 - 2ϵc*η₊*η₋*cos(χ)
# A = kᵣ * sin(2kᵣ*y) * ϵc * sin(χ) / α

end