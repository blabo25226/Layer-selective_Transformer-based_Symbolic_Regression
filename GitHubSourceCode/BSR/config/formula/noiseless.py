import boolean

algebra = boolean.BooleanAlgebra()


##################
## HYPER PARAMS ##
##################

# List of the number of operators and dimensionality used
DIMENSION_MAX = 10
DIMENSION_MIN = 1
BINARY_OP_MAX = 500
BINARY_OP_MIN = 100
UNARY_OP_MAX = 250
UNARY_OP_MIN = 0
EXPR_SIZE_MAX = 200  #SOS and EOS included
ACTIVE_VAR: int = 10
NOISY: bool = False

####################
## SPECIAL TOKENS ##
####################

# Special tokens
INPUT_SPECIAL_TOKENS = ["<PAD>"]  # pad for the evaluation generation
OUTPUT_SPECIAL_TOKENS = ["<SOS_0>", "<SOS_1>", "<EOS>", "<PAD>"]  # Start with less_frqt_rslt; pad for the formula generation; end


###################
#### OPERATORS ####
###################

# Unary operators
UNARY_OPERATORS = {
    '~': lambda op: algebra.NOT(op)
}

# Binary operators
BINARY_OPERATORS = {
    '&': lambda a, b: algebra.AND(a, b),
    '|': lambda a, b: algebra.OR(a, b)
}

###################
#### VARIABLES ####
###################

# Variables (e.g., a, b, c, d)
VARIABLES = {f"x{i + 1}": algebra.Symbol(f"x{i + 1}") for i in range(DIMENSION_MAX)}

###################
## PROBABILITIES ##
###################

# There is only one non trivial unary operator: the not. => its probability = 100%

# List of binary operators (assigning probabilities to binary operators)
BINARY_OP_PROB = {
    '&': 1,  # Probability of using "and"
    '|': 1,    # Probability of using "or"
}
