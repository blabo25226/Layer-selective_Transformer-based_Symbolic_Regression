from .BaseConfig import BaseConfig

class ConfigTransformer(BaseConfig):
    def __init__(self, hparams=None, py_config_path=None, args=None):
        """Initialize configuration specific to Transformer model."""

        # Attributes for transformer configuration
        self.config_attrs = {
            # Model architecture
            'D_EMBED': None,
            'D_MODEL': None,
            'NUM_ENCODER_LAYERS': None,
            'NUM_DECODER_LAYERS': None,
            'NUM_HEADS': None,
            'DIM_FEEDFORWARD': None,
            'DROPOUT': None,
            'ATTENTION_SIZE': None,  # EXPR_SIZE_MAX - 1 (should be set externally or computed)
            'INPUT_POINT_DIM_MAX': None,  # DIMENSION_MAX (should be set externally or computed)
            'NB_INPUT_TOKENS': None,
            
            # Learning rate and scheduling
            'LEARNING_RATE': None,
            'WARMUP_STEPS': None,
            'STATIONARY_STEPS': None,
            'DECAY_STEPS': None,
            'WARMUP_START_FACTOR': None,
            'DECAY_END_FACTOR': None,
        }

        # List of required attributes for validation (except size attributes, which are properties)
        required_attributes = [
            'D_EMBED', 'D_MODEL', 'NUM_ENCODER_LAYERS', 'NUM_DECODER_LAYERS', 'NUM_HEADS', 'DIM_FEEDFORWARD', 
            'DROPOUT', 'ATTENTION_SIZE', 'INPUT_POINT_DIM_MAX', 'NB_INPUT_TOKENS',
            'LEARNING_RATE', 'WARMUP_STEPS', 'STATIONARY_STEPS', 'DECAY_STEPS',
            'WARMUP_START_FACTOR', 'DECAY_END_FACTOR'
        ]

        # Call the base class initializer
        super().__init__(hparams=hparams, py_config_path=py_config_path, args=args, required_attributes=required_attributes, expected_attrs=self.config_attrs)
