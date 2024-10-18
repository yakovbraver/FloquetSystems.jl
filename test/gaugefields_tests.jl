@testset "Test `filter_count!` and `fft_to_matrix!`" begin
    u = [collect(0:7)'; collect(10:17)'; collect(20:27)']
    n_elems = GaugeFields.filter_count!(u, fft_threshold=1e-3)
    @test n_elems == 210
    
    H_rows = Vector{Int}(undef, n_elems)
    H_cols = Vector{Int}(undef, n_elems)
    H_vals = Vector{Float64}(undef, n_elems)
    
    GaugeFields.fft_to_matrix!(H_rows, H_cols, H_vals, u, (0, 0))
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
    @test H == H_true
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
    U = 𝑈(x, x; ϵ, ϵc, χ) .* (dx/L)^2
    u = rfft(U)
    @test sum(abs.(imag.(u))) < 1e-10 # test that imaginary part is zero

    @views s = u[1:M, 1:M]
    @test sum(abs.(s - transpose(s))) < 1e-10 # test that matrix is symmetric
end

@testset "Test `fill_blockband!`" begin
    blocksize = 2
    q_rows = [1, 1, 2, 2]
    q_cols = [1, 2, 1, 2]
    q_vals = [1, 2, 3, 4]
    nelems = length(q_vals)
    nblockrows = 3
    Q_rows = Vector{Int}(undef, (nblockrows * blocksize)^2)
    Q_cols = similar(Q_rows)
    Q_vals = similar(Q_rows)
    ω = 10
    # diagonal blocks
    counter = 1
    GaugeFields.fill_blockband!(Q_rows, Q_cols, Q_vals, q_rows, q_cols, q_vals, 0, blocksize, nblockrows, counter)
    counter += nblockrows * nelems
    # off-diagonal blocks
    for m in 1:nblockrows-1
        GaugeFields.fill_blockband!(Q_rows, Q_cols, Q_vals, q_rows, q_cols, q_vals, m, blocksize, nblockrows, counter)
        counter += 2(nblockrows-m) * nelems
    end
    Q = sparse(Q_rows, Q_cols, Q_vals)
    Q_true = [
        1  2  1  3  1  3
        3  4  2  4  2  4
        1  2  1  2  1  3
        3  4  3  4  2  4
        1  2  1  2  1  2
        3  4  3  4  3  4
    ]
    @test Q == Q_true
end

@testset "Test `fillkronmatrix!`" begin
    M = 1

    n_diag = (2M+1)^2 # number of diagonal elements in 𝐻
    n_elem = 4(2M)^2 + n_diag
    
    H_rows = Vector{Int}(undef, n_elem)
    H_cols = Vector{Int}(undef, n_elem)
    H_vals = Vector{Int}(undef, n_elem)

    counter = GaugeFields.fillkronmatrix!(H_rows, H_cols, H_vals; nx=1, ny=1, c=1, M, position=(1, 3), counter=1)
    B13 = diagm(4 => [1, 1, 0, 1, 1])
    counter = GaugeFields.fillkronmatrix!(H_rows, H_cols, H_vals; nx=1, ny=-1, c=2, M, position=(2, 3), counter)
    B23 = diagm(-2 => [0, 2, 2, 0, 2, 2, 0])
    counter = GaugeFields.fillkronmatrix!(H_rows, H_cols, H_vals; nx=-1, ny=1, c=3, M, position=(3, 1), counter)
    B31 = diagm(2 => [0, 3, 3, 0, 3, 3, 0])
    counter = GaugeFields.fillkronmatrix!(H_rows, H_cols, H_vals; nx=-1, ny=-1, c=4, M, position=(3, 2), counter)
    B32 = diagm(-4 => [4, 4, 0, 4, 4])
    counter = GaugeFields.fillkronmatrix!(H_rows, H_cols, H_vals; nx=0, ny=0, c=5, M, position=(3, 3), counter)
    B33 = Diagonal(fill(5, 9))
    
    H = Matrix(sparse(H_rows, H_cols, H_vals));

    Z = zeros(Int, size(B33))
    H_true = [
        Z   Z   B13
        Z   Z   B23
        B31 B32 B33
    ]
    
    @test H == H_true
end