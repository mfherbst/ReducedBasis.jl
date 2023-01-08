push!(LOAD_PATH, "../src/")
using ReducedBasis
using Documenter

DocMeta.setdocmeta!(ReducedBasis, :DocTestSetup, :(using ReducedBasis); recursive=true)

makedocs(;
    modules=[ReducedBasis],
    authors="Michael F. Herbst <info@michael-herbst.com> and Paul Brehmer",
    repo="https://github.com/mfherbst/ReducedBasis.jl/blob/{commit}{path}#{line}",
    sitename="ReducedBasis.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mfherbst.github.io/ReducedBasis.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Introduction" => "index.md",
        "Examples" => [
            "examples/xxz_ed.md",
            "examples/xxz_dmrg.md",
        ],
        "API reference" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/mfherbst/ReducedBasis.jl",
    devbranch="master",
)
