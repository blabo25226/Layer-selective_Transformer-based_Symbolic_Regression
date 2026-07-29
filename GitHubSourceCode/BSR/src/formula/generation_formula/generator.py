from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
import random

from src.ConfigClasses.ConfigFormula import ConfigFormula

@dataclass
class Node:
    """A node in a binary tree representing a boolean formula.
    
    Attributes:
        value: The operator or variable/constant value
        left: Left child node (None for unary operators)
        right: Right child node
    """
    value: str
    left: Optional['Node'] = None
    right: Optional['Node'] = None

    def is_leaf(self) -> bool:
        """Check if this node is a leaf node (no children)."""
        return not self.left and not self.right
    
    def is_unary(self) -> bool:
        """Check if this node represents a unary operator (only right child)."""
        return self.left is None and self.right is not None

    def __str__(self) -> str:
        """Generate a string representation of the subtree rooted at this node."""
        if self.is_leaf():
            return str(self.value)
        left_str = str(self.left) if self.left else ""
        right_str = str(self.right) if self.right else ""
        return f"({left_str} {self.value} {right_str})"

def get_all_nodes(node: Node) -> list:
    """
    Recursively collects all nodes from the binary tree.

    Args:
        node (Node): The root node from which to start collecting nodes.

    Returns:
        List[Node]: A list containing all the nodes in the tree, including internal nodes and leaves.
    """
    if not node:
        return []
    
    nodes = [node]
    
    if node.left:
        nodes.extend(get_all_nodes(node.left))
    if node.right:
        nodes.extend(get_all_nodes(node.right))
    
    return nodes

def normalize_probabilities(probabilities: dict) -> dict:
    """
    Normalizes the probabilities in a given dictionary to ensure they sum to 1.

    Args:
        probabilities (dict): A dictionary where keys represent possible events and values represent raw probabilities.

    Returns:
        dict: A new dictionary where the values are normalized to sum to 1.
    """
    total = sum(probabilities.values())
    normalized_dict = {k: v / total for k, v in probabilities.items()}
    return normalized_dict



def get_operator(probabilities: dict):
    """
    Selects a random operator based on the probabilities dictionary, 
    then retrieves the corresponding operator function from the operators dictionary.
    
    Args:
        operators (dict): A dictionary of operator functions.
        probabilities (dict): A dictionary of probabilities associated with each operator.
    
    Returns:
        str: A randomly chosen operator, with the selection weighted by the given probabilities.
    """
    # Select a random operator name based on the probabilities
    selected_op = random.choices(list(probabilities.keys()), weights=list(probabilities.values()))[0]
    
    # Return the corresponding function from the operators dictionary
    return selected_op


def get_variables(dim: int, all_variables: List[str]) -> List[str]:
    """Select a random subset of variables.
    
    Args:
        dim: Number of variables to select
        all_variables: List of all available variable names
        
    Returns:
        List[str]: Selected variable names
        
    Raises:
        ValueError: If dim is larger than available variables
    """
    if dim > len(all_variables):
        raise ValueError(f"Requested {dim} variables but only {len(all_variables)} available")
    return random.sample(all_variables, dim)



def insert_unary_operator(node: Node, operator: str) -> Node:
    """
    Wraps a given node in a unary operator node.

    Args:
        node (Node): The node to be wrapped with a unary operator.
        operator (str): The unary operator to apply.

    Returns:
        Node: A new node with the specified unary operator as the parent of the original node.
    """
    # Wrap the node with the unary operator
    return Node(operator, right=node)


def insert_n_unary_operators(root: Node,
                           unary_op: Dict[str, Any],
                           n: int) -> Node:
    """Insert n unary operators randomly in the tree.
    
    Args:
        root: Root node of the tree
        unary_op: Dictionary of available unary operators
        n: Number of unary operators to insert
        
    Returns:
        Node: Root of the modified tree
    """
    if not root or not unary_op or n <= 0:
        return root

    all_nodes = get_all_nodes(root)
    for _ in range(n):
        if not all_nodes:
            break
            
        # Select random node and operator
        target = random.choice(all_nodes)
        op = random.choice(list(unary_op.keys()))
        
        # Create new unary node
        new_node = Node(value=op, right=target)
        
        # Replace target with new node
        if target is root:
            root = new_node
        else:
            root = replace_node_in_parent(root, target, new_node)
            
        # Update available nodes
        all_nodes = get_all_nodes(root)
    
    return root


def replace_node_in_parent(root: Node, target: Node, new_node: Node) -> Node:
    """
    Replaces a target node in the binary tree with a new node, preserving the structure of the tree.

    Args:
        root (Node): The root of the binary tree.
        target (Node): The node to be replaced.
        new_node (Node): The node to replace the target.

    Returns:
        Node: The modified tree with the target node replaced.
    """
    if not root:
        return None
    if root is target:
        return new_node
    root.left = replace_node_in_parent(root.left, target, new_node)
    root.right = replace_node_in_parent(root.right, target, new_node)
    return root



def generate_random_function(config: ConfigFormula) -> Tuple[Node, int]:
    """Generate a random boolean function tree based on configuration parameters.
    
    Args:
        config: Configuration object containing generation parameters
        
    Returns:
        Tuple containing:
            - Root node of the generated formula tree
            - Number of candidate variables used
            
    Raises:
        ValueError: If configuration parameters are invalid
    """
    # Variables import
    dim_max = config.DIMENSION_MAX
    dim_min = config.DIMENSION_MIN
    b_max = config.BINARY_OP_MAX
    b_min = config.BINARY_OP_MIN
    u_max = config.UNARY_OP_MAX
    u_min = config.UNARY_OP_MIN
    binary_op = config.BINARY_OPERATORS
    binary_op_prob = config.BINARY_OP_PROB
    unary_op = config.UNARY_OPERATORS
    all_variables = config.VARIABLES

    nb_active_var = config.ACTIVE_VAR
    nb_candidates = random.randint(nb_active_var, dim_max)

    # Definition of the parameters of the structure
    dim = random.randint(dim_min, nb_active_var)
    nb_bin_op = random.randint(b_min + max(0, dim - 1), dim_max + b_max)  # Ensure b is not negative and within max limits
    nb_unary_op = random.randint(u_min, u_max)
    variables = get_variables(dim, list(all_variables.keys())[:nb_candidates])

    # Normalization of the probabilities
    normalized_bin_op_prob = normalize_probabilities(binary_op_prob)

    # Tree generation
    tree = generate_binary_tree(nb_bin_op, binary_op, 
                                normalized_bin_op_prob, variables)
    tree = insert_n_unary_operators(tree, unary_op, nb_unary_op)

    return tree, nb_candidates


def print_tree(node: Node, level: int = 0, prefix: str = "Root: ") -> None:
    """
    Prints the structure of the binary tree in a hierarchical format, with indentation reflecting depth.

    Args:
        node (Node): The current node being printed.
        level (int): The current depth of the node in the tree (used for indentation).
        prefix (str): A label to prepend to the value of each node (e.g., "Root", "L---", "R---").
    """
    if node is not None:
        # Print the current node with indentation based on its level in the tree
        print(" " * (level * 4) + prefix + str(node.value))
        if node.left:
            print_tree(node.left, level + 1, prefix="L--- " )
        if node.right:
            print_tree(node.right, level + 1, prefix="R--- ")

def generate_binary_tree(nb_bin_op: int, 
                        binary_op: Dict[str, Any],
                        binary_op_prob: Dict[str, float],
                        variables: List[str]) -> Node:
    """Generate a random binary tree with the specified operators.
    
    Args:
        nb_bin_op: Number of binary operators to include
        binary_op: Dictionary of available binary operators
        binary_op_prob: Probability distribution for operator selection
        variables: List of available variables
        
    Returns:
        Node: Root of the generated binary tree
        
    Raises:
        ValueError: If probability distribution is invalid
    """
    if not binary_op_prob or abs(sum(binary_op_prob.values()) - 1.0) > 1e-6:
        raise ValueError("Invalid probability distribution for binary operators")

    # Create initial leaf nodes
    leaves = [Node(value=var) for var in variables]
    
    # Add binary operators
    for _ in range(nb_bin_op):
        if len(leaves) < 2:
            break
            
        # Select operator based on probability distribution
        op = random.choices(list(binary_op_prob.keys()), 
                          weights=list(binary_op_prob.values()))[0]
        
        # Select and remove two random leaves
        left, right = random.sample(leaves, 2)
        leaves.remove(left)
        leaves.remove(right)
        
        # Create new node and add it to leaves
        new_node = Node(value=op, left=left, right=right)
        leaves.append(new_node)
    
    return leaves[0] if leaves else None