from typing import List, Optional, Tuple, Any
import torch, random
from torch.optim import Adam
from torch.optim.lr_scheduler import SequentialLR, LinearLR
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
import logging

from .TransformerModel import TransformerModel
from .metrics import accuracy
from ..ConfigClasses import ConfigTransformer, ConfigFormula
from ..formula import Vocabulary, Formula
from ..general_functions import polish_to_expr

class LtnTransformer(LightningModule):
    """Lightning module for training the transformer model.
    
    Handles training, validation, and prediction logic for the boolean formula
    transformer model.
    
    Attributes:
        c_transformer: Transformer configuration
        model: The underlying transformer model
        input_vocab: Vocabulary for input tokenization
        output_vocab: Vocabulary for output tokenization
    """

    def __init__(
            self,
            c_transformer: ConfigTransformer = None,
            input_vocab: Vocabulary = None,
            output_vocab: Vocabulary = None,
            learning_rate: float = 0.0002,
            warmup_steps: int = 10000,
            start_factor: float = 0.001,
            decay_steps: int = 10000,
            final_factor: float = 0.001):
        """Initialize the transformer model.
        
        Args:
            c_transformer: Configuration for transformer architecture
            input_vocab: Vocabulary for input tokenization
            output_vocab: Vocabulary for output tokenization
            learning_rate: Initial learning rate
            warmup_steps: Number of warmup steps
            start_factor: Starting factor for learning rate
            decay_steps: Number of decay steps
            final_factor: Final learning rate factor
        """
        super().__init__()

        # Store configuration and vocabularies
        if c_transformer.config_attrs:  # Weird but mandatory to allow the restart from last checkpoint
            self.c_transformer = ConfigTransformer(hparams=c_transformer.config_attrs) 
                   
        self.model = TransformerModel(config=c_transformer, 
                                    input_vocab=input_vocab,
                                    output_vocab=output_vocab)

        # Set up vocabularies and special tokens
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self._setup_special_tokens()

        # Training parameters
        self.learning_rate = learning_rate
        self.full_criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.out_PAD_id)

        self.warmup_steps = warmup_steps
        self.start_factor = start_factor
        self.decay_steps = decay_steps
        self.final_factor = final_factor

        # Calculate total steps for training
        self.total_steps = (c_transformer.WARMUP_STEPS + 
                           c_transformer.STATIONARY_STEPS + 
                           c_transformer.DECAY_STEPS)

        self.save_hyperparameters()


    def _setup_special_tokens(self) -> None:
        """Set up special tokens and their IDs."""
        self.in_spe_tokens = self.input_vocab.special_tokens
        self.out_spe_tokens = self.output_vocab.special_tokens
        self.in_spe_ids = [self.input_vocab.token_to_id[token] 
                          for token in self.input_vocab.special_tokens]
        self.out_spe_ids = [self.output_vocab.token_to_id[token] 
                           for token in self.output_vocab.special_tokens]
        self.out_PAD_id = self.output_vocab.PAD_id

    def on_fit_start(self):
        # Update the WandB config
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.config.update({
                'input_vocab_id_to_token': self.input_vocab.id_to_token,
                'output_vocab_id_to_token': self.output_vocab.id_to_token
            })

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.c_transformer.LEARNING_RATE)
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.c_transformer.WARMUP_START_FACTOR,
            end_factor=1.0,
            total_iters=self.c_transformer.WARMUP_STEPS
        )
        
        # Constant LR scheduler (for stationary phase)
        constant_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=self.c_transformer.STATIONARY_STEPS
        )
        
        # Decay scheduler
        decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.c_transformer.DECAY_END_FACTOR,
            total_iters=self.c_transformer.DECAY_STEPS
        )
        
        # Combine schedulers
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, constant_scheduler, decay_scheduler],
            milestones=[self.c_transformer.WARMUP_STEPS, 
                       self.c_transformer.WARMUP_STEPS + self.c_transformer.STATIONARY_STEPS]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output tensor
        """
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], 
                     batch_idx: int) -> torch.Tensor:
        """Perform a training step.
        
        Args:
            batch: Tuple of (input, target) tensors
            batch_idx: Index of the current batch
            
        Returns:
            Loss tensor
        """
        src, tgt = batch

        tgt_input = tgt[:, :-1]  # Excluding the last token for input
        tgt_output = tgt[:, 1:]  # Target excludes the first token

        output = self.model(src, tgt_input)

        # Log the learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, prog_bar=True, on_step=True, on_epoch=False, logger=True)

        # logs and accuracy metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        loss = self.criterion(output.permute(0, 2, 1), tgt_output)  # Adjusting output for CrossEntropyLoss
        full_loss = self.full_criterion(output.permute(0, 2, 1), tgt_output)  # Taking the PAD in count!
        
        acc = accuracy(output, tgt_output, self.out_spe_ids)
        self.log("train_full_loss", full_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_acc", acc.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return full_loss
    
    def get_formula_scores(self, 
                         c_formula: ConfigFormula,
                         pred_ids: torch.Tensor,
                         evaluated_pts: torch.Tensor) -> Tuple[List[Tuple[Formula, float]], int]:
        """Calculate scores for predicted formulas.
        
        Args:
            c_formula: Formula configuration
            pred_ids: Tensor of predicted token IDs
            evaluated_pts: Tensor of evaluation points
            
        Returns:
            Tuple containing:
                - List of (formula, score) tuples
                - Number of invalid formulas
        """
        # Move tensors to CPU for processing
        pred_ids = pred_ids.cpu()
        evaluated_pts = evaluated_pts.cpu()

        pts = evaluated_pts[:, :-1]
        tgt_results = evaluated_pts[:, -1]

        formula_scores = []
        nb_invalid = 0

        for pred_id in pred_ids:
            pol_pred = self.output_vocab.detokenize_expr(pred_id)
            pred_expr = polish_to_expr(pol_pred, config=c_formula)

            if pred_expr == "Invalid Polish":
                logging.debug("Invalid Polish expression detected")
                nb_invalid += 1
                continue

            try:
                pred_formula = Formula(c_formula, math_expr=pred_expr, simplify=True)
                pred_evaluations = pred_formula.evaluate_pts(pts)

                if pred_evaluations is not None:
                    pred_results = pred_evaluations[:, -1]
                    score = torch.sum(pred_results == tgt_results) / tgt_results.size(0)
                    formula_scores.append((pred_formula, score.item()))
            except Exception as e:
                logging.debug(f"Formula evaluation failed: {str(e)}")
                nb_invalid += 1
                continue

        # Sort by score (descending) and formula length (ascending)
        formula_scores.sort(key=lambda x: (-x[1], len(x[0])))
        return formula_scores, nb_invalid

    def predict(self, 
                c_formula: ConfigFormula,
                list_evaluated_pts: List[torch.Tensor],
                beam: int,
                temperature: float = 0.7,
                use_beam: bool = False) -> List[Tuple[List[Formula], List[float], int, int]]:
        """Generate predictions for a batch of evaluation points.
        
        Args:
            c_formula: Formula configuration
            list_evaluated_pts: List of evaluation point tensors
            beam: Number of predictions to generate
            temperature: Sampling temperature
            use_beam: Whether to use beam search
            
        Returns:
            List of tuples containing:
                - List of predicted formulas
                - List of corresponding scores
                - Number of predictions made
                - Number of invalid predictions
        """
        batch_size = len(list_evaluated_pts)
        batch_evaluated_pts = []
        sos_ids_token = []

        # Prepare input tensors
        for evaluated_pts in list_evaluated_pts:
            evaluated_pts_token, less_freq_rslt = self.input_vocab.tokenize_eval(
                evaluated_pts, c_formula
            )
            batch_evaluated_pts.append(evaluated_pts_token)
            sos_ids_token.append(self.output_vocab.SOS_id(less_freq_rslt))

        # Stack tensors for batch processing
        batch_evaluated_pts = torch.stack(batch_evaluated_pts).to(self.device)
        sos_ids_token = torch.tensor(sos_ids_token, device=self.device).unsqueeze(1).unsqueeze(2)

        # Generate predictions
        if use_beam:
            batch_pred_ids, _ = self.model.beam_search(
                batch_evaluated_pts, sos_ids_token, beam=beam, temperature=temperature
            )
        else:
            batch_pred_ids, _ = self.model.generate(
                batch_evaluated_pts, sos_ids_token, nb_to_gen=beam, temperature=temperature
            )
        
        # Process results
        results = []
        for b in range(batch_size):
            pred_ids = batch_pred_ids[b]
            pts = list_evaluated_pts[b]
            formula_scores, nb_invalid = self.get_formula_scores(c_formula, pred_ids, pts)

            formulas = [item[0] for item in formula_scores]
            scores = [item[1] for item in formula_scores]

            results.append((formulas, scores, len(pred_ids), nb_invalid))

        return results

    def _predict_genetic(self, c_formula, evaluated_pts, cut_seq_ids, temperature=1):
        evaluated_pts_token, _ = self.model.input_vocab.tokenize_eval(evaluated_pts, c_formula)
        evaluated_pts_token = evaluated_pts_token.to(self.device)

        pred_ids, _ = self.model.continue_generation(evaluated_pts_token, cut_seq_ids, nb_to_gen=1, temperature=temperature)

        formula_scores, nb_invalid = self.get_formula_scores(c_formula, pred_ids, evaluated_pts)

        return formula_scores, len(pred_ids), nb_invalid

    def genetic(self, c_formula, evaluated_pts, seed, beam, temperature=0.7):
        _, lfr = self.model.input_vocab.tokenize_eval(evaluated_pts, c_formula)

        init_formulas, init_scores, nb_init_preds, nb_init_invalids = self.predict(c_formula, evaluated_pts.unsqueeze(0), seed * beam, temperature=1)[0]
        init_formulas = init_formulas[:seed]
        init_scores = init_scores[:seed]

        formulas_scores = [(init_formulas[i], init_scores[i]) for i in range(len(init_formulas))]
        best_formula = init_formulas[0]
        best_score = init_scores[0]

        counter = 0
        nb_preds = nb_init_preds
        nb_invalids = nb_init_invalids
        while counter < 3 and best_score < 0.9999999:
            counter += 1
            
            mixed_formulas = []
            for f, _ in formulas_scores[:seed]:
                # We keep the original
                mixed_formulas.append(f)
                for _ in range(beam-1):
                    # We add the alternatives
                    alt_f = f.alternative()
                    mixed_formulas.append(alt_f)

            shortest_length = min([len(mf) for mf in mixed_formulas])
            
            cut_size  = random.randint(1, shortest_length + 1)
            polish_formulas = [mf.polish_expr for mf in mixed_formulas]
            list_polishs_tokens = [self.output_vocab.tokenize_expr(p_f, c_formula, lfr) for p_f in polish_formulas]                    
            list_cut_polishs_tokens = [p_t[0:cut_size] for p_t in list_polishs_tokens]

            cut_polish_tokens = torch.stack(list_cut_polishs_tokens)

            new_formulas_scores, nb_new_preds, nb_new_invalids = self._predict_genetic(c_formula, evaluated_pts, cut_polish_tokens, temperature=temperature)

            nb_preds += nb_new_preds
            nb_invalids += nb_new_invalids

            if new_formulas_scores[0][1] > best_score:
                best_formula, best_score = new_formulas_scores[0]
                counter = 0

            formulas_scores.extend(new_formulas_scores)

            formulas_scores.sort(key=lambda x: (-x[1], len(x[0])))
            formulas_scores = formulas_scores[:seed]

        result = best_formula, best_score, nb_preds, nb_invalids

        return result
