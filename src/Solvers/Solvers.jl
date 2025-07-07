"""
    Solvers.jl

A module containing all of the solvers that I have defined thus far
"""
module Solvers

    include("ERD_SOC.jl"); using .ERD_SOC
    export ERD_SOC_Solver

end