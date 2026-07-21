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
        # e^{-τ/2 P} I e^{-τ/2 P} = cosh(τ) I - sinh(τ) P, normalized by cosh(τ)
        @test -sinh(tau) / cosh(tau) ≈ scalarproduct(rho_evolved, gate_pstr)

        rho_cache = PropagationCache(deepcopy(rho_sum))
        rho_cache_evolved = propagate!(gate, rho_cache, tau; heisenberg=false, normalize_coeffs=false, min_abs_coeff=0)
        rho_evolved = PropType(rho_cache_evolved)
        @test length(rho_evolved) == 2
        @test cosh(tau) ≈ getcoeff(rho_evolved, 0) != 1.0
        @test -sinh(tau) ≈ scalarproduct(rho_evolved, gate_pstr)


        # default behavior is Heisenberg, which we currently don't support
        @test_throws ErrorException propagate(gate, rho, tau)
    end
end

@testset "Test Imaginary Time Evolution Cools Towards the Ground State: diagonal Hamiltonian" begin
    # a small Ising-type Hamiltonian, diagonal in the computational (Z) basis
    nq = 4
    hs = [0.6, -0.4, 0.9, -0.7]
    pairs = [(1, 2), (2, 3), (3, 4)]
    Js = [0.5, -0.6, 0.4]

    H = PauliSum(nq)
    for i in 1:nq
        add!(H, :Z, i, hs[i])
    end
    for (k, (a, b)) in enumerate(pairs)
        add!(H, [:Z, :Z], [a, b], Js[k])
    end

    # H is diagonal, so its ground energy is found by brute-force enumeration
    # over all computational basis states
    function _diagonalenergy(z)
        e = sum(hs[i] * z[i] for i in 1:nq)
        e += sum(Js[k] * z[a] * z[b] for (k, (a, b)) in enumerate(pairs))
        return e
    end
    all_energies = [_diagonalenergy([b == 0 ? 1 : -1 for b in digits(s, base=2, pad=nq)]) for s in 0:(2^nq-1)]
    E_ground = minimum(all_energies)

    circuit = Gate[]
    for i in 1:nq
        push!(circuit, ImaginaryPauliRotation(:Z, i))
    end
    for (a, b) in pairs
        push!(circuit, ImaginaryPauliRotation([:Z, :Z], [a, b]))
    end
    # all generators are diagonal, hence mutually commuting, so this
    # Trotterization is exact regardless of step size, and a handful of large
    # steps is as accurate as many small ones
    step_params = vcat(1.0 .* hs, 1.0 .* Js)

    rho = PauliString(nq, :I, 1)
    energies = Float64[scalarproduct(H, rho)]
    for _ in 1:10
        rho = propagate!(circuit, rho, step_params; heisenberg=false, min_abs_coeff=0)
        push!(energies, scalarproduct(H, rho))
    end

    # the natural, un-negated Hamiltonian coefficients must monotonically cool
    # the state towards the ground state energy
    @test issorted(energies, rev=true)
    @test isapprox(energies[end], E_ground; atol=1e-2)
end

@testset "Test Imaginary Time Evolution Cools Towards the Ground State: product-state X Hamiltonian" begin
    # a non-interacting Hamiltonian of single-qubit X terms, whose ground state
    # is the exactly known product state of ±X eigenstates
    nq = 4
    h = [0.8, -0.5, 1.2, -0.3]

    H = PauliSum(nq)
    for i in 1:nq
        add!(H, :X, i, h[i])
    end
    E_ground = -sum(abs.(h))

    circuit = [ImaginaryPauliRotation(:X, i) for i in 1:nq]
    # single-qubit terms all commute, so the Trotterization is exact regardless
    # of step size, and a handful of large steps is as accurate as many small ones
    tau = 1.0
    step_params = tau .* h

    rho = PauliString(nq, :I, 1)
    energies = Float64[scalarproduct(H, rho)]
    for layer in 1:10
        rho = propagate!(circuit, rho, step_params; heisenberg=false, min_abs_coeff=0)
        beta = layer * tau
        # matches the closed-form thermal expectation value at every step
        E_exact = -sum(h[i] * tanh(beta * h[i]) for i in 1:nq)
        push!(energies, scalarproduct(H, rho))
        @test energies[end] ≈ E_exact atol = 1e-9
    end

    @test issorted(energies, rev=true)
    @test isapprox(energies[end], E_ground; atol=1e-2)

    # the final state approaches the product state polarized opposite to each
    # local field, i.e. <X_i> -> -sign(h_i)
    for i in 1:nq
        @test scalarproduct(PauliString(nq, :X, i), rho) ≈ -sign(h[i]) atol = 1e-2
    end
end