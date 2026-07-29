import torch
from typing import List, Dict, Optional, Tuple, Union
import logging
from pathlib import Path

from ..ConfigClasses import ConfigFormula


class Vocabulary:
    """Manages tokenization vocabularies for formula processing.
    
    Handles conversion between tokens and IDs for both input evaluations
    and output expressions.
    
    Attributes:
        token_to_id: Maps tokens to unique IDs
        id_to_token: Maps IDs back to tokens
        special_tokens: List of special tokens (PAD, SOS, etc.)
    """

    def __init__(self, 
                 configFormula: Optional[ConfigFormula] = None,
                 input_auto_fill: bool = False,
                 output_auto_fill: bool = False):
        """Initialize vocabulary.
        
        Args:
            configFormula: Configuration for vocabulary setup
            input_auto_fill: Whether to auto-fill input vocabulary
            output_auto_fill: Whether to auto-fill output vocabulary
            
        Raises:
            ValueError: If auto-fill requested without config
        """
        self.token_to_id: Dict[Union[str, torch.Tensor], int] = {}    
        self.id_to_token: Dict[int, str] = {}
        self.special_tokens: List[str] = []

        if input_auto_fill:
            if configFormula is None:
                raise ValueError("Configuration required for auto-fill")
            self.input_auto_fill_vocabulary(configFormula=configFormula)
            
        if output_auto_fill:
            if configFormula is None:
                raise ValueError("Configuration required for auto-fill")
            self.output_auto_fill_vocabulary(configFormula=configFormula)

    @property
    def PAD_id(self) -> int:
        """Get the ID of the padding token."""
        return self.token_to_id.get("<PAD>", 0)

    def __len__(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.token_to_id)

    def SOS_id(self, less_freq_rslt):
        return self.token_to_id[f"<SOS_{less_freq_rslt}>"]


    def input_auto_fill_vocabulary(self, configFormula: ConfigFormula):
        """Populates the vocabulary with tokens from configuration settings."""
        # Appropriate tokens definitions
        self.special_tokens = list(configFormula.INPUT_SPECIAL_TOKENS)
        input_tokens_list = self.special_tokens

        input_tokens_list += ['0', '1']

        for token in input_tokens_list:
            self.add_token(token)

    def output_auto_fill_vocabulary(self, configFormula: ConfigFormula):
        """Populates the vocabulary with tokens from configuration settings."""
        self.special_tokens = list(configFormula.OUTPUT_SPECIAL_TOKENS)

        # Appropriate tokens definitions
        output_tokens_list = self.special_tokens + list(configFormula.UNARY_OPERATORS) + \
            list(configFormula.BINARY_OPERATORS) + list(configFormula.VARIABLES)

        # Add the tokens
        for token in output_tokens_list:
            self.add_token(token)

    def add_token(self, token: str):
        """Adds a new token to the vocabulary if it doesn't already exist."""
        if token not in self.token_to_id:
            num_tokens = len(self.token_to_id)
            self.token_to_id[token] = num_tokens
            self.id_to_token[num_tokens] = token


    def print_vocab(self, nb_to_show=None):
        """
        Prints the vocabulary tokens and their corresponding IDs, optionally limiting the number shown.

        Args:
            nb_to_show (int, optional): Number of vocabulary entries to display. Defaults to None, which means all
            entries are displayed.
        """
        sorted_vocab = sorted(self.id_to_token.items(), key=lambda item: item[0])
        # If nb_to_show is None or exceeds the number of tokens, display all entries; otherwise, display up to
        # nb_to_show entries.
        end_index = nb_to_show if nb_to_show is not None and nb_to_show < len(sorted_vocab) else len(sorted_vocab)
        for id, token in sorted_vocab[:end_index]:
            print(f"{id}: {token}")

    def tokenize_expr(self, expression: List[str], configFormula: ConfigFormula, less_freq_rslt: int = None) -> torch.Tensor:
        """
        Tokenizes an expression by converting each token into its corresponding ID.
        Assumes all tokens in the expression are already in the vocabulary.
        """
        # Prepare tensor for <SOS> token
        if less_freq_rslt == None:
           raise ValueError("less_freq_rslt is undefined in tokenize_expr")
        start_token = torch.tensor([self.SOS_id(less_freq_rslt)], dtype=torch.long)

        # Tokenize the expression
        token_ids = [self.token_to_id[token] for token in expression if isinstance(token, str)]
        tokenized = torch.tensor(token_ids, dtype=torch.long)

        # Prepare tensor for <PAD> tokens
        nb_pad_tokens = max(0, configFormula.EXPR_SIZE_MAX - len(tokenized) - 2)  # -2 bc of the SOS and the EOS
        padding_tokens = torch.full((nb_pad_tokens,), self.PAD_id, dtype=torch.long)
        end_token = torch.tensor([self.token_to_id["<EOS>"]], dtype=torch.long)

        # Combine all parts into one tensor
        result = torch.cat([start_token, tokenized, end_token, padding_tokens], dim=0)

        return result

    def detokenize_expr(self, numerical_tokens: List[int]) -> List[str]:
        """
        Converts numerical token IDs back into their original string tokens.

        Args:
            numerical_tokens (List[int]): List of numerical token IDs to detokenize.

        Returns:
            List[str]: List of string tokens corresponding to the numerical IDs.
        """
        list_tokens = numerical_tokens
        if isinstance(list_tokens, torch.Tensor):
            list_tokens = list_tokens.tolist()

        # return [self.id_to_token.get(id, "<UNK>") for id in numerical_tokens]
        detokenized = []
        for token_id in list_tokens:
            # Retrieve the token string from id_to_token, using '<UNK>' if the token ID is not recognized
            token = self.id_to_token.get(token_id, '<UNK>')
            if token not in self.token_to_id.keys():
                print("Token_id not recognized : ", token_id)
            if token not in self.special_tokens:
                detokenized.append(token)
        return detokenized

    def tokenize_eval(self, 
                     evaluations: torch.Tensor,
                     configFormula: ConfigFormula) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize evaluation results.
        
        Args:
            evaluations: Tensor of evaluation results
            configFormula: Configuration for tokenization
            
        Returns:
            Tuple containing:
                - Tokenized evaluations tensor
                - Less frequent result tensor
                
        Raises:
            ValueError: If tokenization fails
        """
        # Tokenization for boolean points and values
        def pre_tokenize(value: float) -> str:
            return str(int(value))
        
        def less_frequent(results):
            """
            Determine the less frequent result (either 0 or 1)

            Args:
                results (torch.Tensor[nb_results]): Tensor of all the results of each point: 0 or 1

            Returns:
                torch.Tensor[1]: the less frequent result (0 or 1). 1 if it is symmetric.
            """

            nb_results = results.shape[0]
            sum_results = results.sum()

            if sum_results > nb_results/2:  # in case of ==, less reuslts is 1
                # Only one result present; less frequent is the only one available
                less_frequent_result = torch.tensor(0)
            else:
                # Find the result with the minimum count
                less_frequent_result = torch.tensor(1)

            return less_frequent_result      
        
        num_dimensions = evaluations.shape[1] - 1  # Last column is the evaluated value
        dim_max = configFormula.DIMENSION_MAX

        # Split points and values
        points = evaluations[:, :-1]
        results = evaluations[:, -1]


        # Change the output if it is noisy
        if configFormula.NOISY:
            final_len = configFormula.NB_INPUTS[1]
            pad_dimension_len = dim_max - num_dimensions
            pad_dimension_tensor = torch.full((pad_dimension_len, ), self.PAD_id, dtype=torch.long)


            eval_tokens = []
            for i in range(points.shape[0]):
                point = points[i]
                result = results[i]

                point_tokens = torch.tensor([self.token_to_id[pre_tokenize(el.item())] for el in point], dtype=torch.long)
                result_token = torch.tensor([self.token_to_id[pre_tokenize(result)]], dtype=torch.long)

                line_tokens = torch.cat([point_tokens, pad_dimension_tensor, result_token], dim=0)
                eval_tokens.append(line_tokens.clone().detach().unsqueeze(0))

            eval_tokens_tensor = torch.cat(eval_tokens, dim=0)
            less_frequent_result = torch.tensor(0)  # Convention: returns 0 for the noisy model

        else:
            final_len = 2 ** (dim_max - 1)

            less_frequent_result = less_frequent(results)
                    
            # Filter evaluations to include only points with the less frequent result
            mask = (results == less_frequent_result)
            filtered_points = points[mask]

            # Proceed with tokenization using the filtered data
            variable_tokens = []
            for i in range(num_dimensions):
                variable = filtered_points[:, i]
                dimension_tokens = [self.token_to_id[pre_tokenize(value.item())] for value in variable]
                variable_tokens.append(torch.tensor(dimension_tokens, dtype=torch.long))

            # Stack tokens for each variable
            eval_tokens_tensor = torch.stack(variable_tokens, dim=1)

        nb_tok_eval, nb_tok_dim_eval = eval_tokens_tensor.shape

        # Handle padding of the number of formulas
        pad_formulas_len = final_len - nb_tok_eval # At most 2 ** (dim_max - 1) points to consider as we only take the less frqt rslt
        pad_formulas_tensor = torch.full((pad_formulas_len, nb_tok_dim_eval), self.PAD_id, dtype=torch.long)

        # Combine formulas tokens and padding tokens for the tokenized formulas tensor
        full_tokenized_tensor = torch.cat([eval_tokens_tensor, pad_formulas_tensor], dim=0)

        return full_tokenized_tensor, less_frequent_result



def create_both_vocabs(config: ConfigFormula, verbose: bool = False) -> Tuple[Vocabulary, Vocabulary]:
    """Create input and output vocabularies for formula processing.
    
    Args:
        config: Configuration object containing vocabulary settings
        verbose: Whether to print creation status
        
    Returns:
        Tuple containing:
            - Input vocabulary
            - Output vocabulary
    """
    input_vocab = Vocabulary(configFormula=config, input_auto_fill=True)
    output_vocab = Vocabulary(configFormula=config, output_auto_fill=True)

    if verbose:
        logging.info("Vocabularies have been successfully created!")

    return input_vocab, output_vocab
