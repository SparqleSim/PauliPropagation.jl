@testset "Test Imaginary Pauli Rotation" begin
    nq = rand(4:256)

    rho = PauliString(nq, :I, 1)

    symbs = [:X, :Y, :Z]

    for support in [1, 2, 3, 4]
        gate_generator = rand(symbs, support)
        gate_inds = shuffle(1:nq)[1:support] # drawing without replacement
        gate_pstr = PauliString(nq, gate_generator, gate_inds)

        gate = ImaginaryPauliRotation(gate_generator, gate_inds)

        tau = rand()
        # choose a random propagation backend
        PropType = rand((PauliSum, VectorPauliSum))
        rho_sum = PropType(rho)
        rho_cache = PropagationCache(deepcopy(rho_sum))
        rho_cache_evolved = propagate!(gate, rho_cache, tau; heisenberg=false, min_abs_coeff=0)
        rho_evolved = PropType(rho_cache_evolved)
        @test length(rho_evolved) == 2
        @test getcoeff(rho_evolved, 0) ≈ 1.0
        @test sinh(tau) / cosh(tau) ≈ scalarproduct(rho_evolved, gate_pstr)

        rho_cache = PropagationCache(deepcopy(rho_sum))
        rho_cache_evolved = propagate!(gate, rho_cache, tau; heisenberg=false, normalize_coeffs=false, min_abs_coeff=0)
        rho_evolved = PropType(rho_cache_evolved)
        @test length(rho_evolved) == 2
        @test cosh(tau) ≈ getcoeff(rho_evolved, 0) != 1.0
        @test sinh(tau) ≈ scalarproduct(rho_evolved, gate_pstr)


        # default behavior is Heisenberg, which we currently don't support
        @test_throws ErrorException propagate(gate, rho, tau)
    end
end