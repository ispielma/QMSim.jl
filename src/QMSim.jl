"""
    QMSim.jl

A toolkit for solving quantum mechanics problems, with a focus on those described by compact mathematical expressions that can be encoded as a set of rules.
"""
module QMSim

include("Helpers.jl"); using .Helpers
export DimensionWithSpace

# Todo: move DimensionWithSpace to a new helpers.jl

include("AbstractTypes.jl"); using .AbstractTypes
export AbstractRule, AbstractRuleBuilder, AbstractMatrixWithRules # types
export get_array, build_rules!, set_default_kwargs!, dim # methods

include("MatrixBuilders.jl"); using .MatrixBuilders
export RelativeRule, ElementRule, AbsoluteRule, ExplicitRule, MatrixWithRules, MatricesWithRules
export add_rule!, build!, generate_builders!, generate_builders, add_matrix!

include("MatrixSolvers.jl"); using .MatrixSolvers
export QMSolver, eigensystem!

end
