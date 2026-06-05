using Test


@testset "Test Visualization Module" begin

    @testset "PauliTreeTracker Creation" begin
        tracker = PauliTreeTracker(1.0)
        @test tracker.coeff == 1.0
        @test isa(tracker.node_id, String)
        @test !isempty(tracker.node_id)
    end

    @testset "Tree Operations" begin
        reset_tree!()

        # Test adding nodes and edges
        add_node!("test1", "XYZ", "RX(θ)")
        @test haskey(EVOLUTION_TREE, "test1")

        add_edge!("test1", "test2", "cos(θ)", "RX")
        @test length(EVOLUTION_EDGES) >= 1

        # Verify tree state
        @test length(EVOLUTION_TREE) >= 1
        @test isa(EVOLUTION_TREE["test1"], TreeNode)
    end

    @testset "Pauli String Formatting" begin
        pstr = PauliString(3, [:X, :Y, :Z], [1, 2, 3], 1.0)
        formatted = format_pauli_string(pstr)
        @test isa(formatted, String)
        @test !isempty(formatted)
        @test contains(formatted, "X") || contains(formatted, "Y") || contains(formatted, "Z")
    end

    @testset "Visualization Functions" begin
        # Test that print_tree_summary doesn't throw errors
        @test_nowarn print_tree_summary()
    end

    @testset "Export Functions" begin
        # Test GraphViz export
        test_dot_file = "test_output.dot"
        @test_nowarn export_to_graphviz(test_dot_file)
        @test isfile(test_dot_file)
        rm(test_dot_file, force=true)

        # Test JSON export
        test_json_file = "test_output.json"
        @test_nowarn export_to_json(test_json_file)
        @test isfile(test_json_file)
        rm(test_json_file, force=true)
    end

end