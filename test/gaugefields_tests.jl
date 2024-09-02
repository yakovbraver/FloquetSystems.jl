@testset "Test `filter_count!` and `fft_to_matrix!`" begin
    u = [collect(0:7)'; collect(10:17)'; collect(20:27)']
    n_elem = GaugeFields.filter_count!(u, factor=1e-3)
    @test 210 == n_elem
    
    H_rows = Vector{Int}(undef, n_elem)
    H_cols = Vector{Int}(undef, n_elem)
    H_vals = Vector{Float64}(undef, n_elem)
    
    GaugeFields.fft_to_matrix!(H_rows, H_cols, H_vals, u)
    H_true =
    [  0   1   2   3   4  10  11  12  13  14  20  21  22  23  24
       1   0   1   2   3  17  10  11  12  13  27  20  21  22  23
       2   1   0   1   2  16  17  10  11  12  26  27  20  21  22
       3   2   1   0   1  15  16  17  10  11  25  26  27  20  21
       4   3   2   1   0  14  15  16  17  10  24  25  26  27  20
      10  17  16  15  14   0   1   2   3   4  10  11  12  13  14
      11  10  17  16  15   1   0   1   2   3  17  10  11  12  13
      12  11  10  17  16   2   1   0   1   2  16  17  10  11  12
      13  12  11  10  17   3   2   1   0   1  15  16  17  10  11
      14  13  12  11  10   4   3   2   1   0  14  15  16  17  10
      20  27  26  25  24  10  17  16  15  14   0   1   2   3   4
      21  20  27  26  25  11  10  17  16  15   1   0   1   2   3
      22  21  20  27  26  12  11  10  17  16   2   1   0   1   2
      23  22  21  20  27  13  12  11  10  17   3   2   1   0   1
      24  23  22  21  20  14  13  12  11  10   4   3   2   1   0]
    H = sparse(H_rows, H_cols, H_vals)
    @test H_true == H
end

@testset "Test that FFT of `𝑈` is real and even" begin
    ϵ = 0.1 # testing for Float64
    ϵc = 1
    χ = 0
    gf = GaugeField(ϵ, ϵc, χ; n_harmonics=10, fft_threshold=0.05)

    L = π # periodicity of the potential
    M = 20
    dx = L / 2M
    x = range(0, L-dx, 2M)
    U = 𝑈(gf, x, x) .* (dx/L)^2
    u = rfft(U)
    @test sum(abs.(imag.(u))) < 1e-10 # test that imaginary part is zero

    @views s = u[1:M, 1:M]
    @test sum(abs.(s - transpose(s))) < 1e-10 # test that matrix is symmetric
end
