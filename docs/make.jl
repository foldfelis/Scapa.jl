using Scapa
using Documenter

DocMeta.setdocmeta!(Scapa, :DocTestSetup, :(using Scapa); recursive=true)

makedocs(;
    modules=[Scapa],
    authors="JingYu Ning <foldfelis@gmail.com> and contributors",
    repo="https://github.com/foldfelis/Scapa.jl/blob/{commit}{path}#{line}",
    sitename="Scapa.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://foldfelis.github.io/Scapa.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/foldfelis/Scapa.jl",
    devbranch="main",
)
