include("../src/GaugeFields.jl")
using .GaugeFields
using Test
using SparseArrays, FFTW

@testset "Basic tests" begin

    @testset "GaugeFields tests" begin
        include("gaugefields_tests.jl")
    end
end