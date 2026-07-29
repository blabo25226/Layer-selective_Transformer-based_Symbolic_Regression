from typing import List
import boolean
from boolean import Symbol, AND, NOT, OR

import torch
import random

from .ConfigClasses import ConfigFormula



def pts_generator(c_formula : ConfigFormula, nb_candidates: int) -> torch.Tensor:
    """
    Generates all possible binary combinations for a given dimension using PyTorch.
    
    Args:
        dimension (int): The dimensionality of each point (i.e., the number of variables).
        
    Returns:
        torch.Tensor: A tensor containing all possible binary combinations, with each element being either 0 or 1.
    """

    # If the data are noisy
    if c_formula.NOISY:
        return rdm_walk_pts_generator(c_formula, nb_candidates)
    else:
        dimension = nb_candidates # c_formula.DIMENSION_MAX

        # Number of combinations is 2**dimension
        num_combinations = 2 ** dimension
        
        # Create a range of integers from 0 to 2^dimension - 1
        indices = torch.arange(num_combinations, dtype=torch.int64)
        
        # Convert each integer to its binary representation
        binary_combinations = ((indices.unsqueeze(1) >> torch.arange(dimension - 1, -1, -1)) & 1).float()
        return binary_combinations


def rdm_walk_pts_generator(c_formula: ConfigFormula, nb_candidates: int) -> torch.Tensor:
    """
    Generates a subset of points in the binary hypercube using a random walk.
    
    Args:
        dimension (int): The dimensionality of the binary hypercube.
        prob_rd_walk (tuple): The range of probabilities for flipping coordinates (γexpl).
                              E.g., (0.05, 0.25).
        nb_pts (int): The number of points to generate.
        
    Returns:
        torch.Tensor: A tensor containing the generated points, with each element being either 0 or 1.
    """
    prob_rd_walk = c_formula.PROB_RD_WALK
    # dimension = c_formula.DIMENSION_MAX
    nb_pts_range = c_formula.NB_INPUTS

    # Ensure the probability range is valid
    if not (0 <= prob_rd_walk[0] <= prob_rd_walk[1] <= 1):
        raise ValueError("Invalid range for prob_rd_walk. Must be within [0, 1].")

    # Sample the flipping probability (γexpl) uniformly within the given range
    gamma_expl = random.uniform(*prob_rd_walk)
    nb_pts = random.randint(*nb_pts_range)

    # Initialize the list of points
    points = []

    # Sample an initial binary point uniformly
    current_point = torch.randint(0, 2, (nb_candidates,), dtype=torch.float32)
    points.append(current_point)

    # Perform the random walk to generate the remaining points
    for _ in range(nb_pts - 1):
        # Create a mask for flipping each coordinate with probability γexpl
        flip_mask = torch.bernoulli(torch.full((nb_candidates,), gamma_expl))
        # Generate the new point by flipping selected bits in the current point
        current_point = (current_point + flip_mask) % 2  # Mod 2 to ensure binary values
        points.append(current_point)

    # Stack the points into a single tensor
    return torch.stack(points)


def noisy_rdm_pts_generator(c_formula: ConfigFormula, nb_candidates: int, nb_pts: int = 1000) -> torch.Tensor:
    """
    Generates a fully random subset of points in the binary hypercube.
    
    Args:
        c_formula (ConfigFormula): Configuration object containing parameters.
        nb_candidates (int): The dimensionality of the binary hypercube (number of candidates).
        
    Returns:
        torch.Tensor: A tensor containing the generated points, with each element being either 0 or 1.
    """
    # Generate random binary points
    points = torch.randint(0, 2, (nb_pts, nb_candidates), dtype=torch.float32)

    return points


def add_noise(eval: torch.Tensor, c_formula : ConfigFormula):
    if not c_formula.NOISY:
        return eval
    
    prob_flip_range = c_formula.PROB_FLIP_INTERVAL

    prob_flip = random.uniform(*prob_flip_range)


    # Create a mask for flipping each coordinate with probability γexpl
    flip_mask = torch.bernoulli(torch.full((eval.shape), prob_flip))
    # Generate the new point by flipping selected bits in the current point
    noisy_eval = (eval + flip_mask) % 2  # Mod 2 to ensure binary values

    return noisy_eval, prob_flip


def expr_to_polish(expr, config):
    """
    Transforms a boolean expression into a list of tokens representing the expression in Polish notation.
    Args:
        expr (boolean.Expression): The boolean expression to convert.
        config: Configuration object with operators.
    Returns:
        List[str]: The expression converted to a list of tokens in Polish notation.
    Raises:
        NotImplementedError: If the expression contains unsupported types.
    """

    # Check if the expression is a literal (variable or constant)
    if isinstance(expr, Symbol):
        # It's a symbol
        return [str(expr)]
    
    # Check for unary operators (e.g., NOT)
    elif isinstance(expr, NOT):
        op_name = '~'
        operand = expr_to_polish(expr.args[0], config)
        return [op_name] + operand
    
    # Check for binary operators (AND, OR)
    elif isinstance(expr, (AND, OR)):
        if isinstance(expr, AND):
            op_name = '&'
        else:
            op_name = '|'

        operands = [expr_to_polish(arg, config) for arg in expr.args]
        
        if len(operands) > 2:
            # Nest the operators to handle more than two operands
            nested = operands[0]
            for operand in operands[1:]:
                nested = [op_name] + nested + operand
            return nested
        else:
            # For two operands, no nesting is needed
            flat_list = [op_name] + [item for sublist in operands for item in sublist]
            return flat_list
    
    else:
        raise NotImplementedError(f"Conversion for operator '{expr}' is not implemented.")

def polish_to_expr(tokens: List[str], config: ConfigFormula) -> boolean.Expression:
    """
    Transforms a list of tokens representing a Boolean expression in Polish notation into a boolean.Expression.

    Args:
        tokens (List[str]): The list of tokens in Polish notation.
        config: Configuration object with operators.

    Returns:
        boolean.Expression or str: The Boolean expression converted from the list of tokens,
                                   or "Invalid Polish" if the input is invalid.
    """
    index = 0

    invalid_polish = "Invalid Polish"
    
    def parse():
        nonlocal index
        if index >= len(tokens):
            return invalid_polish  # End parsing immediately if out of tokens

        token = tokens[index]
        index += 1

        if token in config.VARIABLES or token.isidentifier():
            return Symbol(token)

        elif token in config.UNARY_OPERATORS:
            operand = parse()
            if operand == invalid_polish:
                return invalid_polish  # Quit parsing
            return config.UNARY_OPERATORS[token](operand)

        elif token in config.BINARY_OPERATORS:
            left = parse()
            if left == invalid_polish:
                return invalid_polish  # Quit parsing
            right = parse()
            if right == invalid_polish:
                return invalid_polish  # Quit parsing
            return config.BINARY_OPERATORS[token](left, right)
        else:
            return invalid_polish  # Unknown token, quit parsing

    # Start parsing
    expr = parse()

    # Validate parsing result
    if expr == invalid_polish or index != len(tokens):
        return invalid_polish  # Return invalid if parsing failed or extra tokens exist

    return expr