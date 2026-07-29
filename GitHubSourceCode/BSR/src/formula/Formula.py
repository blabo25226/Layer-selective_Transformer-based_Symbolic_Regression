from typing import List, Any

import torch
import boolean
import logging

from sklearn.metrics import f1_score

from ..ConfigClasses import ConfigFormula
from .generation_formula import tree_to_expr, get_alternative_tree, expr_to_tree
from ..general_functions import expr_to_polish, polish_to_expr

class Formula:
    """A class representing a boolean formula with evaluation capabilities.
    
    Attributes:
        config (ConfigFormula): Configuration for formula generation and evaluation
        is_valid (bool): Whether the formula is valid
        math_expr: The mathematical expression representing the formula
        polish_expr: The formula in Polish notation
    """

    # class attributes
    config = None

    def __init__(self, config: ConfigFormula, tree: Any = None, math_expr: Any = None, simplify: bool = True):
        """Initialize a Formula instance.
        
        Args:
            config: Configuration object for the formula
            tree: Tree representation of the formula
            math_expr: Mathematical expression of the formula
            simplify: Whether to simplify the formula after creation
        
        Raises:
            ValueError: If neither tree nor math_expr is provided
        """
        self.is_valid = True

        # Define the class attribute if undefined
        if self.config is None:
            self.config = config

        # Create the mathematical expression
        if math_expr is not None:
            self.math_expr = math_expr 
        elif tree is not None:
            self.math_expr = tree_to_expr(tree, self.config)
        else:
            raise ValueError("Math_expr is not the right type or tree does not exist")
        
        # Create the Polish expression
        self.polish_expr = expr_to_polish(self.math_expr, self.config)
        if simplify:
            self.loop_polish_simplify()

        # Check the validity
        if self.dim == 0 or len(self.polish_expr) < 1 or len(self.polish_expr) > config.EXPR_SIZE_MAX - 2:  # The SOS and EOS are included
            # Formula is invalid
            self.is_valid = False  


    def __len__(self):
        if self.is_valid is False:
            return 0
        return len(self.polish_expr)
    
    @property
    def dim(self):
        return len(set(self.math_expr.get_symbols()))
    
    def alternative(self):
        tree = expr_to_tree(self.math_expr)
        alternative_tree = get_alternative_tree(tree)

        alternative_expr = tree_to_expr(alternative_tree, self.config)

        return Formula(self.config, math_expr=alternative_expr, simplify=False)
   

    def score(self, tgt_evaluations: torch.Tensor) -> float:
        """Calculate the score of the formula against target evaluations.
                
        Args:
            tgt_evaluations: Target evaluations tensor
            
        Returns:
            float: score between 0 and 1
        """
        try:
            pts = tgt_evaluations[:, :-1]
            tgt_results = tgt_evaluations[:, -1]

            pred_evaluations = self.evaluate_pts(pts)
            pred_results = pred_evaluations[:, -1]

            pred_results = pred_results.cpu()
            tgt_results = tgt_results.cpu()

            score = torch.sum(pred_results == tgt_results)/tgt_results.size(0)

            return score.item()
        except Exception as e:
            logging.error("Error in formula score: %s", e)
            return 0

    def f1_score(self, tgt_evaluations: torch.Tensor) -> float:
        """Calculate the F1 score between formula predictions and target evaluations.
        
        Args:
            tgt_evaluations: Target evaluations tensor
            
        Returns:
            float: F1 score between 0 and 1
        """
        try:
            pts = tgt_evaluations[:, :-1]
            tgt_results = tgt_evaluations[:, -1]

            pred_evaluations = self.evaluate_pts(pts)
            pred_results = pred_evaluations[:, -1]

            score = f1_score(pred_results.cpu().numpy(), tgt_results.cpu().numpy())
            return score
        except Exception as e:
            logging.error("Error in formula f1_score: %s", e)
            return 0

    def loop_polish_simplify(self):
        """
        No clue of why it behaves like that while simplify 2 times does not change anything
        but it seems that going through the polish notation and then call simplify, simplifies the formula again.
        """
        not_converged = True
        counter = 0

        while not_converged:
            current_math_expr = self.math_expr
            new_math_expr = polish_to_expr(self.polish_expr, self.config)
            new_math_expr = new_math_expr.simplify()

            if new_math_expr != current_math_expr:
                counter += 1
                if counter > 100:
                    self.is_valid = False
                    raise ValueError("Infinite loop in polish simplify!!")
                
                self.math_expr = new_math_expr
                self.polish_expr = expr_to_polish(new_math_expr, self.config)
            else:
                not_converged = False


    def evaluate_pts(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the boolean formula at specified points (nb_pts, dim_max) composed of 0/1 values.
        Each point is used to evaluate the boolean expression. The logic is unchanged,
        but the variable-to-coordinate assignment is adapted to handle arbitrary xN variables.
        """

        def evaluate_expr(expr, values):
            """
            Manually evaluates a boolean expression using the provided values for the variables.
            """
            # WARNING: The operators are hard coded (but hard to do it another way)
            # Check if the expression is a symbol (e.g., x1, x2)
            if isinstance(expr, boolean.Symbol):
                return values[str(expr)]
            
            # If it's an AND operation
            if expr.operator == '&':
                return all(evaluate_expr(arg, values) for arg in expr.args)
            
            # If it's an OR operation
            if expr.operator == '|':
                return any(evaluate_expr(arg, values) for arg in expr.args)
            
            # If it's a NOT operation
            if expr.operator == '~':
                return not evaluate_expr(expr.args[0], values)

            # Raise an error if the expression can't be evaluated
            raise ValueError(f"Unsupported operator or symbol: {expr}")


        # Ensure the points are binary (0 or 1)
        assert torch.all((points == 0) | (points == 1)), "Input points must be binary (0 or 1)."

        # Gather the unique symbolic variables used in the boolean expression
        symbols = sorted(set(self.math_expr.get_symbols()), key=lambda x: str(x))

        nb_pts = points.shape[0]
        evaluated_values = []
        sum_results = 0

        for i in range(nb_pts):
            point_values = {}

            # For each symbolic variable, find its numeric index and pick the corresponding bit from points[i]
            for symbol in symbols:
                symbol_str = str(symbol)  # e.g. 'x5', 'x2', or maybe just 'x1'
                if symbol_str.startswith('x'):
                    # Parse out the integer after 'x'
                    try:
                        var_idx = int(symbol_str[1:]) - 1  # e.g. 'x5' -> index=4
                    except ValueError  as e:
                        # If the symbol name isn't 'xN' with N an integer, skip or handle differently
                        print(f"Warning: symbol '{symbol_str}' not recognized as xN form.")
                        continue
                    # Now retrieve the value from the points tensor
                    point_values[symbol_str] = bool(points[i, var_idx].item())
                else:
                    # If the expression includes something like Symbol('2'), decide how to handle it:
                    # For now, let's skip or treat as constant "True/False" if needed.
                    # We'll set them to False by default or raise an error
                    # If your use case is different, handle accordingly.
                    print(f"Warning: symbol '{symbol_str}' is not in form 'xN'. Setting to False.")
                    point_values[symbol_str] = False

            try:
                # Evaluate the expression on this point
                result = evaluate_expr(self.math_expr, point_values)
                sum_results += int(result)

                # Append the original point coordinates (all dim_max bits) + the result
                # If you only want the used bits, you'd slice. But the request says
                # "return the concatenated points and their value," so we keep the entire row.
                row_data = list(points[i].cpu().numpy()) + [int(result)]
                evaluated_values.append(row_data)

            except Exception as e:
                self.is_valid = False
                raise ValueError(f"Error evaluating point {points[i]}: {e}")

        # If the formula is trivially always 0 or always 1, mark invalid
        if sum_results == 0 or sum_results == nb_pts:
            self.is_valid = False

        # Convert results to a tensor: shape (nb_pts, dim_max + 1)
        evaluated_values = torch.tensor(evaluated_values, dtype=torch.float32)

        return evaluated_values