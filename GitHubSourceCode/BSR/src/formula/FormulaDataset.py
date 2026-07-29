from typing import List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import timeout_decorator
import logging

from .Vocabulary import Vocabulary
from .Formula import Formula
from .generation_formula import generate_random_function
from ..general_functions import pts_generator, add_noise
from ..ConfigClasses import ConfigFormula

class FormulaDataset(Dataset):
    """Dataset class for generating and managing boolean formulas.
    
    This dataset generates formulas on-the-fly rather than storing them,
    making it effectively infinite in size.
    
    Attributes:
        config (ConfigFormula): Configuration for formula generation
        input_vocab (Vocabulary): Vocabulary for input tokenization
        output_vocab (Vocabulary): Vocabulary for output tokenization
    """

    def __init__(self, 
                 configFormula: ConfigFormula, 
                 input_vocab: Optional[Vocabulary] = None,
                 output_vocab: Optional[Vocabulary] = None):
        """Initialize the dataset.
        
        Args:
            configFormula: Configuration for formula generation
            input_vocab: Optional vocabulary for input tokenization
            output_vocab: Optional vocabulary for output tokenization
            
        Raises:
            ValueError: If vocabularies are not properly initialized
        """
        self.config = configFormula
        self._input_vocab = None
        self._output_vocab = None

        if input_vocab is not None:
            self.input_vocab = input_vocab
        if output_vocab is not None:
            self.output_vocab = output_vocab

    @property
    def input_vocab(self) -> Vocabulary:
        """Get the input vocabulary."""
        if self._input_vocab is None:
            raise ValueError("Input vocabulary not initialized")
        return self._input_vocab

    @input_vocab.setter
    def input_vocab(self, vocab: Vocabulary) -> None:
        """Set the input vocabulary.
        
        Args:
            vocab: The vocabulary to use for input tokenization
            
        Raises:
            ValueError: If vocab is not a Vocabulary instance
        """
        if not isinstance(vocab, Vocabulary):
            raise ValueError("input_vocab must be an instance of the Vocabulary class")
        self._input_vocab = vocab

    @property
    def output_vocab(self) -> Vocabulary:
        """Get the output vocabulary.
        
        Returns:
            Vocabulary: The vocabulary for output tokenization
            
        Raises:
            ValueError: If output vocabulary is not initialized
        """
        if self._output_vocab is None:
            raise ValueError("Output vocabulary not initialized")
        return self._output_vocab

    @output_vocab.setter
    def output_vocab(self, vocab: Vocabulary) -> None:
        """Set the output vocabulary.
        
        Args:
            vocab: The vocabulary to use for output tokenization
            
        Raises:
            ValueError: If vocab is not a Vocabulary instance
        """
        if not isinstance(vocab, Vocabulary):
            raise ValueError("output_vocab must be an instance of the Vocabulary class")
        self._output_vocab = vocab

    def __len__(self) -> int:
        """Return effectively infinite size since formulas are generated on-the-fly."""
        return int(1e12)

    @torch.no_grad()
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a random formula and its evaluations.
        
        Args:
            idx: Index (unused since generation is random)
            
        Returns:
            Tuple containing:
                - Tokenized evaluations tensor
                - Tokenized expression tensor
                
        Raises:
            ValueError: If vocabularies are not initialized
        """
        if not isinstance(self.input_vocab, Vocabulary) or not isinstance(self.output_vocab, Vocabulary):
            raise ValueError("input_vocab and output_vocab must be defined")
        
        while True:
            try:
                # Generate formula with timeout protection
                tree, nb_candidates = generate_random_function(self.config)
                
                @timeout_decorator.timeout(1, timeout_exception=StopIteration)
                def create_formula() -> Formula:
                    return Formula(config=self.config, tree=tree, simplify=True)

                formula = create_formula()

                if not formula.is_valid:
                    continue

                # Generate and evaluate points
                pts = pts_generator(self.config, nb_candidates)
                evaluations = formula.evaluate_pts(points=pts)
                
                if self.config.NOISY:
                    evaluations, prob_flip = add_noise(evaluations, c_formula=self.config)

                # Tokenize if formula remains valid
                if formula.is_valid:
                    tokenized_evaluations, less_freq_rslt = self.input_vocab.tokenize_eval(
                        evaluations, configFormula=self.config
                    )
                    tokenized_expression = self.output_vocab.tokenize_expr(
                        formula.polish_expr, configFormula=self.config, less_freq_rslt=less_freq_rslt
                    )

                if tokenized_evaluations is not None and tokenized_expression is not None:
                    return tokenized_evaluations, tokenized_expression

            except Exception as e:
                logging.debug(f"Formula generation failed: {str(e)}")
                continue