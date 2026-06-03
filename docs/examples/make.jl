using IJulia

examples_dir = @__DIR__
notebooks_dir = joinpath(examples_dir, "notebooks")
docs_examples_dir = normpath(joinpath(examples_dir, "..", "src", "examples"))
template_path = joinpath(examples_dir, "markdown_template.tpl")
kernel_name = "paulipropagation-docs"
timeout = 1000  # benchmarking in some files takes a long time
python = get(ENV, "PYTHON", "python3")

IJulia.installkernel(
    kernel_name,
    "--project=$(examples_dir)";
    specname = kernel_name,
    displayname = kernel_name,
)

rm(docs_examples_dir; recursive = true, force = true)
mkpath(docs_examples_dir)

notebooks = sort(filter(endswith(".ipynb"), readdir(notebooks_dir; join = true)))

for notebook in notebooks
    run(`$python -m nbconvert --to markdown --execute
        --ExecutePreprocessor.kernel_name=$kernel_name
        --ExecutePreprocessor.timeout=$timeout
        --output-dir $docs_examples_dir \
        --template-file $examples_dir/markdown_template.tpl \
        --NbConvertBase.display_data_priority "['image/svg+xml', 'image/png', \
            'image/jpeg', 'text/markdown', 'text/plain']" \
        $notebook`)
end
