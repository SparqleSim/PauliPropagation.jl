using IJulia

examples_dir = @__DIR__
docs_examples_dir = normpath(joinpath(examples_dir, "..", "docs", "src", "examples"))
template_path = joinpath(examples_dir, "markdown_template.tpl")
kernel_name = "paulipropagation-docs"
timeout = 600  # benchmarking in some files takes a long time
python = get(ENV, "PYTHON", "python3")

# Use the custom Julia kernel for Julia notebooks to ensure they use the correct project environment.
IJulia.installkernel(
    kernel_name,
    "--project=$(examples_dir)";
    specname = kernel_name,
    displayname = kernel_name,
)

rm(docs_examples_dir; recursive = true, force = true)
mkpath(docs_examples_dir)

notebooks = sort(filter(endswith(".ipynb"), readdir(examples_dir; join = true)))

sem = Base.Semaphore(5)

@sync for notebook in notebooks
    @async begin
        Base.acquire(sem)
        try
            # Python notebooks should use their default kernel.
            is_python = occursin("\"language\": \"python\"", read(notebook, String))
            kernel_arg = is_python ? `` : `--ExecutePreprocessor.kernel_name=$kernel_name`

            run(`$python -m nbconvert --to markdown \
                --execute \
                $kernel_arg \
                --ExecutePreprocessor.timeout=$timeout \
                --output-dir $docs_examples_dir \
                --template-file $examples_dir/markdown_template.tpl \
                --NbConvertBase.display_data_priority "['image/svg+xml', 'image/png', \
                    'image/jpeg', 'text/markdown', 'text/plain']" \
                $notebook`)
        catch e
            @error "$notebook failed to run with the error: \n $e"
        finally
            Base.release(sem)
        end
    end
end
