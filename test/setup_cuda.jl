# GPU tests will by default be skipped unles we set ENV["TEST_CUDA"]="true" in the environment.
# from then on it is our responsibility to ensure that CUDA.jl is functional. 

# this returns the boolean flag when running the file
if parse(Bool, get(ENV, "TEST_CUDA", "false"))
    @info "ENV[\"TEST_CUDA\"]=\"true\". Adding CUDA.jl to the test environment..."
    using Pkg
    Pkg.add("CUDA")
    using CUDA
    true
else
    @info "Skipping GPU tests. Set ENV[\"TEST_CUDA\"]=\"true\" to run them."
    false
end