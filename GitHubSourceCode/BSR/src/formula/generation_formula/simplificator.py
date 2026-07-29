import random
import boolean
from boolean import BooleanAlgebra, Symbol, NOT, AND, OR
from typing import Optional, Union

from src.formula.generation_formula.generator import Node
from src.ConfigClasses.ConfigFormula import ConfigFormula

# Initialize the boolean algebra object
algebra = BooleanAlgebra()

def tree_to_expr(tree: Node, config: ConfigFormula) -> boolean.Expression:
    """Convert a tree representation to a boolean expression.
    
    Args:
        tree: Root node of the tree to convert
        config: Configuration containing operator definitions
        
    Returns:
        boolean.Expression: The resulting boolean expression
        
    Raises:
        ValueError: If tree is invalid or contains unknown operators
    """
    if tree is None:
        return None
    
    if tree.is_leaf():
        return algebra.Symbol(tree.value)

    # Handle unary operators
    if tree.is_unary():
        if tree.value not in config.UNARY_OPERATORS:
            raise ValueError(f"Unknown unary operator: {tree.value}")
        right_expr = tree_to_expr(tree.right, config)
        return config.UNARY_OPERATORS[tree.value](right_expr)

    # Handle binary operators
    if tree.value not in config.BINARY_OPERATORS:
        raise ValueError(f"Unknown binary operator: {tree.value}")
    left_expr = tree_to_expr(tree.left, config)
    right_expr = tree_to_expr(tree.right, config)
    return config.BINARY_OPERATORS[tree.value](left_expr, right_expr)

def expr_to_tree(expr: boolean.Expression) -> Optional[Node]:
    """Convert a boolean expression to a tree representation.
    
    Args:
        expr: Boolean expression to convert
        
    Returns:
        Node: Root node of the resulting tree, or None if conversion fails
        
    Raises:
        ValueError: If expression contains unknown operators
    """
    if isinstance(expr, Symbol):
        return Node(str(expr))

    # Handle NOT operator
    if isinstance(expr, NOT):
        right_tree = expr_to_tree(expr.args[0])
        return Node("not", None, right_tree)

    # Handle binary operators
    if isinstance(expr, (AND, OR)):
        op_str = "and" if isinstance(expr, AND) else "or"
        left_tree = expr_to_tree(expr.args[0])
        right_tree = expr_to_tree(expr.args[1])
        return Node(op_str, left_tree, right_tree)

    return None

def get_alternative_tree(tree: Node) -> Node:
    """Generate an alternative tree structure with equivalent logic.
    
    This function creates a new tree that represents the same logical expression
    but with a potentially different structure (e.g., by swapping operands).
    
    Args:
        tree: Root node of the original tree
        
    Returns:
        Node: Root node of the alternative tree
    """
    if tree is None:
        return None

    # 1) Base case: leaf nodes are copied directly
    if tree.is_leaf():
        return Node(tree.value)

    # 2) Handle unary operators
    if tree.is_unary():
        new_right_child = get_alternative_tree(tree.right)
        return Node(tree.value, None, new_right_child)

    # 3) Handle binary operators
    new_left_child = get_alternative_tree(tree.left)
    new_right_child = get_alternative_tree(tree.right)

    # Randomly swap children for commutative operators
    if random.random() < 0.5:
        new_left_child, new_right_child = new_right_child, new_left_child

    return Node(tree.value, new_left_child, new_right_child)

def rename_variables(expr: boolean.Expression) -> boolean.Expression:
    """
    Renames variables in a boolean.py expression to follow a sequential naming convention (x1, x2, ...).

    Args:
        expr: The original boolean.py expression.

    Returns:
        A new boolean.py expression with variables renamed sequentially.
    """
    # Extract unique symbols from the expression, sorted in lexicographical order by their name
    symbols = sorted(set(str(s) for s in expr.get_symbols()), key=lambda s: int(s[1:]))

    # Create a mapping from old variable names to sequential names (x1, x2, ...)
    rename_mapping = {boolean.Symbol(old_name): boolean.Symbol(f'x{i + 1}') for i, old_name in enumerate(symbols)}

    # Substitute the old variable names with the new sequential names in the expression
    renamed_expr = expr.subs(rename_mapping)

    return renamed_expr



def eliminate_double_negation(expr: boolean.Expression) -> boolean.Expression:
    """
    Eliminates double negations from a boolean expression.

    Args:
        expr: The original boolean expression.

    Returns:
        The boolean expression with double negations removed.
    """
    # Check for double negation: NOT(NOT(A)) -> A
    if isinstance(expr, boolean.NOT) and isinstance(expr.args[0], boolean.NOT):
        return eliminate_double_negation(expr.args[0].args[0])  # Remove double negation

    # Recursively apply to sub-expressions if expr is a compound expression (AND, OR)
    if isinstance(expr, (boolean.AND, boolean.OR)):
        return expr.__class__(*(eliminate_double_negation(arg) for arg in expr.args))
    else:
        return expr



def apply_de_morgans_laws(expr: boolean.Expression) -> boolean.Expression:
    """
    Applies De Morgan's laws to simplify expressions with NOT over AND/OR.

    Args:
        expr: The original boolean expression.

    Returns:
        The boolean expression after applying De Morgan's transformations.
    """
    # Apply De Morgan's: NOT(A AND B) -> NOT(A) OR NOT(B)
    if isinstance(expr, boolean.NOT) and isinstance(expr.args[0], boolean.AND):
        return algebra.OR(*(apply_de_morgans_laws(~arg) for arg in expr.args[0].args))
    
    # Apply De Morgan's: NOT(A OR B) -> NOT(A) AND NOT(B)
    if isinstance(expr, boolean.NOT) and isinstance(expr.args[0], boolean.OR):
        return algebra.AND(*(apply_de_morgans_laws(~arg) for arg in expr.args[0].args))

    # Recursively apply to sub-expressions if expr is a compound expression (AND, OR)
    if isinstance(expr, (boolean.AND, boolean.OR)):
        return expr.__class__(*(apply_de_morgans_laws(arg) for arg in expr.args))
    else:
        return expr