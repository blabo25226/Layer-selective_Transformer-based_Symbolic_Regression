import sympy as sp
from .BaseConfig import BaseConfig

class ConfigFormula(BaseConfig):
    def __init__(self, hparams=None, py_config_path=None, args=None):
        """Initialize configuration specific to formula generation."""
        
        # Define all attributes in config_attrs instead of instance variables
        self.config_attrs = {
            'DIMENSION_MAX': None,
            'DIMENSION_MIN': None,
            'BINARY_OP_MAX': None,
            'BINARY_OP_MIN': None,
            'UNARY_OP_MAX': None,
            'UNARY_OP_MIN': None,
            'EXPR_SIZE_MAX': None,
            'PROB_VAR_VS_CST': None,
            'NOISY': None,
            'ACTIVE_VAR': None,
            'PROB_FLIP_INTERVAL': None,
            'NB_INPUTS': None,
            'PROB_RD_WALK': None,
            'INPUT_SPECIAL_TOKENS': None,
            'OUTPUT_SPECIAL_TOKENS': None,
            'UNARY_OPERATORS': None,
            'BINARY_OPERATORS': None,
            'VARIABLES': None,
            'BINARY_OP_PROB': None,
        }

        # List of required attributes for validation
        required_attributes = [
            'DIMENSION_MAX', 'DIMENSION_MIN', 'BINARY_OP_MAX', 'BINARY_OP_MIN',
            'UNARY_OP_MAX', 'UNARY_OP_MIN', 'EXPR_SIZE_MAX', 'INPUT_SPECIAL_TOKENS', 
            'OUTPUT_SPECIAL_TOKENS', 'UNARY_OPERATORS', 'BINARY_OPERATORS', 'VARIABLES', 
        ]

        # Call the base class initializer with the provided Python config path and arguments
        super().__init__(hparams=hparams, py_config_path=py_config_path, args=args, required_attributes=required_attributes, expected_attrs=self.config_attrs)
