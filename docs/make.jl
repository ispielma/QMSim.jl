using Documenter
using QMSim

makedocs(
    sitename = "QMSim",
    format = Documenter.HTML(),
    # modules = [QMSim, MatrixBuilders]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
