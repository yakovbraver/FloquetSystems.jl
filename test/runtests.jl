include("../src/GaugeFields.jl")
using .GaugeFields
using Test
using SparseArrays, FFTW, LinearAlgebra

@testset "Basic tests" begin

    @testset "GaugeFields tests" begin
        include("gaugefields_tests.jl")
    end
end