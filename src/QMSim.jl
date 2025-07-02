"""
    QMSim.jl

A toolkit for solving matrix based quantum mechanics problems, with a focus on those described by
compact mathematical expressions that can be encoded as a set of rules.
"""
module QMSim

include("Helpers.jl"); using .Helpers
export DimensionWithSpace

include("Rules.jl"); using .Rules
export AbstractRule, AbstractRuleBuilder, RelativeRule, ElementRule, AbsoluteRule, ExplicitRule
export matrix_builder

include("AbstractMatrixTypes.jl"); using .AbstractMatrixTypes
export AbstractMatrixWithRules, isleaf
export get_matrix, set_matrix!, get_dimensions, set_dimensions!, get_default_kwargs, set_default_kwargs!
export get_rules, set_rules!, add_rule!, get_leafs, add_leaf!
export dim, build_rules!, error_check, generate_builders!, generate_builders, build!, build

include("MatrixBuilders.jl"); using .MatrixBuilders
export MatrixWithRules, MatricesWithRules

# include("MatrixSolvers.jl"); using .MatrixSolvers
# export QMSolver, eigensystem!

end
