# Generates HTML documentation from the contents of
# the docs folder. To generate, we must first setup
# a symlink from the repo README.md to src/index.md,
# in order re-use the README in Documenter.jl
# From the docs/ directory (containing this file):
#     cd src
#     ln -s ../../README.md index.md
#     cd ../
#
# This need only be done once per-machine. Then,
# generating/updating the doc is triggered via
#     julia --project make.jl
# 
# If triggered within a Github Action, the generated
# HTML files will then be committed to the 'gh-pages'
# branch, which Github Pages can be configured to
# display at SparqleSim.github.io/PauliPropagation.jl/
# 
# Note documentation generated from non-main branches
# will be uploaded to subdomain /dev/, even when not
# from the 'dev' branch, and doc generated from pull
# requests will be uploaded to /previews/PR#.
#
# If this breaks, break Tyson's legs


using Documenter, PauliPropagation


const DOCS_DIR = @__DIR__
const REPO_ROOT = dirname(DOCS_DIR)
const EXAMPLES_DIR = joinpath(REPO_ROOT, "examples")
const GENERATED_EXAMPLES_DIR = joinpath(DOCS_DIR, "src", "examples")

const EXAMPLE_NOTEBOOKS = [
    "1-basic-example.ipynb" => "Basic Example",
    "2-datatypes.ipynb" => "Data Types",
    "3-utility-example.ipynb" => "Utility Functions",
    "4-pauli-transfer-matrix.ipynb" => "Pauli Transfer Matrix",
    "5-custom-gates.ipynb" => "Custom Gates",
    "6-numerical-certificate.ipynb" => "Numerical Certificates",
    "7-custom-pathproperties.ipynb" => "Custom Path Properties",
    "8-automatic-differentiation.ipynb" => "Automatic Differentiation",
    "9-advanced-custom-gates.ipynb" => "Advanced Custom Gates",
    "introduction-example-error-mitigation.ipynb" => "Error Mitigation",
    "imaginary-time-evolution.ipynb" => "Imaginary Time Evolution",
    "PP-Surrogate.ipynb" => "Pauli Propagation Surrogate",
    "PP-from-Python.ipynb" => "Using PauliPropagation.jl from Python",
    "Symmetry-PP.ipynb" => "Symmetry",
    "visualization_example.ipynb" => "Visualization",
    "ex_ttfi_op_evolution.ipynb" => "TTFI Operator Evolution",
]

const EXAMPLE_PAGES = [
    title => joinpath("examples", replace(notebook, r"\.ipynb$" => ".md"))
    for (notebook, title) in EXAMPLE_NOTEBOOKS
]

function convert_example_notebooks()
    if !isdir(EXAMPLES_DIR)
        error("Could not find examples directory at $(EXAMPLES_DIR)")
    end

    rm(GENERATED_EXAMPLES_DIR; recursive=true, force=true)
    mkpath(GENERATED_EXAMPLES_DIR)

    jupyter = get(ENV, "JUPYTER", "jupyter")
    for (notebook, _) in EXAMPLE_NOTEBOOKS
        notebook_path = joinpath(EXAMPLES_DIR, notebook)
        if !isfile(notebook_path)
            error("Could not find example notebook at $(notebook_path)")
        end

        run(Cmd([
            jupyter,
            "nbconvert",
            "--to", "markdown",
            "--output-dir", GENERATED_EXAMPLES_DIR,
            notebook_path,
        ]))
    end
end


convert_example_notebooks()


# Generate doc HTML files, saved to build/
makedocs(
    # Add favicon.ico
    format=Documenter.HTML(
        assets=[
            "assets/favicon.ico",
        ],
    ), sitename="PauliPropagation.jl",

    # determines site layout
    pages=[

        # index.md does not exist; it is a symlink
        # to the repo's README.md file, created as
        # per the comments above, to avoid duplicating
        # the README.md contents into Documenter.jl 
        # pages. We manually override its name in the
        # left navbar to be "Introduction"
        "Home" => "index.md",

        # these other 'top-level' files DO exist, and
        # have names inferred from their section names

        "Tutorials" => "tutorials.md",

        # generated from examples/*.ipynb by convert_example_notebooks()
        "Examples" => EXAMPLE_PAGES,

        # these 'lower-level' files also exist, and will
        # be grouped under an 'API' section in the navbar
        "API" => [
            "api/PauliDataTypes.md",
            "api/PauliAlgebra.md",
            "api/Gates.md",
            "api/Circuits.md",
            "api/Propagation.md",
            "api/StateOverlap.md",
            "api/PathProperties.md",
            "api/PauliTransferMatrix.md",
            "api/Surrogate.md",
            "api/Symmetry.md",
            "api/NumericalCertificates.md",
            "api/Truncations.md"
        ]
    ]
)


# When run from a Github Action, commit those files to the 'gh-pages' branch,
# depending upon the triggering branch or whether it is a release/pull-request.
deploydocs(
    repo="github.com/SparqleSim/PauliPropagation.jl.git",

    # Enable generation of doc from PRs, under a /previews/PR## sub-domain.
    # Beware that this requires the Github Action was explicitly triggered by
    # a 'pull_request' event (not a 'push')
    push_preview=true,

    # Specify that changes to our 'dev' branch (rather than default 'main')
    # should update the doc visible at the /dev/ sub-domain. Note this means
    # pushes to the main branch never generate doc; only new releases will
    # (see below)
    devbranch="dev",
    devurl="dev",

    # Control which Github releases (here, all) trigger re-generation of the
    # main documentation, and their URLs. Below, we specify that:
    # - subdomain /stable/ presents the very latest release doc
    # - all versions (including patches) have their own hosted doc under /vX.Y.Z/
    # - changes to the dev branch should update the /dev/ sub-domain; this seems
    #   gratuitous since specified above and since irrelevant to Github releases,
    #   but it is present in the deploydocs() doc without elaboration. Grr!
    versions=["stable" => "v^", "v#.#.#", "dev" => "dev"]
)


# Once 'gh-pages' branch is updated, and Github Pages has been configured to
# publish files from that branch, the documentation is visible at either:
# - SparqleSim.github.io/PauliPropagation.jl/
# - SparqleSim.github.io/PauliPropagation.jl/dev/
# - SparqleSim.github.io/PauliPropagation.jl/previews/PR#
# where # above is replaced with the pull request number.
#
# These "doc clones" are deleted whenever a commit is pushed to the main
# branch (signifying a version release), so that development history does
# not bloat the repo. Deletion is performed by the 'tidy-doc' CI job.
