import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ConfigClasses import ConfigTransformer
from ..formula import Vocabulary


def create_mask(size):
    """Create a triangular (upper) mask to prevent attention to future positions."""
    mask = torch.triu(torch.ones(size, size, dtype=torch.float32) * float('-inf'), diagonal=1)
    return mask


import torch.nn as nn

class TransformerModel(nn.Transformer):
    def __init__(self, config: ConfigTransformer, input_vocab: Vocabulary, output_vocab: Vocabulary):
        """Initialize the Transformer model using the configuration provided in ConfigTransformer."""

        # Call the parent nn.Transformer initializer using values from the config
        super().__init__(d_model=config.D_MODEL,
                         nhead=config.NUM_HEADS,
                         num_encoder_layers=config.NUM_ENCODER_LAYERS,
                         num_decoder_layers=config.NUM_DECODER_LAYERS,
                         dim_feedforward=config.DIM_FEEDFORWARD,
                         dropout=config.DROPOUT,
                         norm_first=True, 
                         batch_first=True)

        # Access attributes from the ConfigTransformer instance
        self.input_vocab : Vocabulary = input_vocab
        self.output_vocab : Vocabulary = output_vocab
        self.out_spe_ids = [self.output_vocab.token_to_id[token] for token in output_vocab.special_tokens]
        self.eos_id = self.output_vocab.token_to_id["<EOS>"]
        self.input_vocab_size : int = len(input_vocab)
        self.output_vocab_size : int = len(output_vocab)
        
        self.d_embed : int = config.D_EMBED
        self.d_model : int = config.D_MODEL
        self.attention_size : int = config.ATTENTION_SIZE

        # Embedding layer using vocab sizes and dimensions from the config
        self.src_input_emb = nn.Embedding(self.input_vocab_size, self.d_embed)
        self.tgt_input_emb = nn.Embedding(self.output_vocab_size, self.d_model)
        self.tgt_position_emb = nn.Embedding(self.attention_size, self.d_model)

        # # Full mask definition
        self.register_buffer('mask', torch.triu(torch.ones(self.attention_size, self.attention_size, dtype=torch.float32) * float('-inf'), diagonal=1))

        # Output layer
        self.output_layer = nn.Linear(self.d_model, self.output_vocab_size)

        # Calculate eval_size based on input_point_dim_max and d_embed from the config
        self.eval_size = config.INPUT_POINT_DIM_MAX * self.d_embed

        self.dim_reduce = nn.Sequential(
            nn.Linear(self.eval_size, self.d_model),
            nn.SiLU()
        )

        self.tgt_mask = None

    def get_mask(self, size):
        return self.mask[:size][:size]

    def forward(self, src, tgt):
        """Forward pass of the transformer model."""
        B, T = tgt.shape

        src = self.src_input_emb(src)
        pos_embed = torch.arange(0, T, device=tgt.device, dtype=torch.long)[None, :]

        tgt = self.tgt_input_emb(tgt) + self.tgt_position_emb(pos_embed)
        tgt = tgt.to(torch.float32)

        src = src.reshape(src.size(0), src.size(1), -1)
        src = self.dim_reduce(src)

        if self.tgt_mask is None or self.tgt_mask.size(0) != tgt.size(1):
            self.tgt_mask = self.get_mask(tgt.size(1))

        memory = self.encoder(src)  # Encode source input
        output = self.decoder(tgt, memory, tgt_mask=self.tgt_mask, tgt_is_causal=True)  # Decode to target
        final_output = self.output_layer(output)  # Final output layer produces (batch_size, tgt_length, vocab_size)
        return final_output    


    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        sos_id: int | torch.Tensor,
        nb_to_gen: int,
        max_len: int = None, 
        temperature: float = 1.
    ):
        if isinstance(sos_id, torch.Tensor) is False:
            sos_id  = torch.tensor([[sos_id]])

        return self.continue_generation(src, sos_id, nb_to_gen, max_len, temperature)

    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,
        sos_id: int | torch.Tensor,
        beam: int,
        max_len: int = None, 
        temperature: float = 1.
    ):
        if isinstance(sos_id, torch.Tensor) is False:
            sos_id  = torch.tensor([[sos_id]])

        return self.continue_beam_search(src, sos_id, beam, max_len, temperature)



    @torch.no_grad()
    def continue_generation(
        self,
        src: torch.Tensor,
        cut_seq_ids: torch.Tensor,
        nb_to_gen: int = 5,
        max_len: int = None, 
        temperature: float = 1.
    ):
        """
        Continue generation from a given sequence, handling two cases:
        
        Case 1:
            - src: (nb_pts, pts_dim)
            - cut_seq_ids: (number_of_initial_sequences, seq_len_so_far)
            - Returns: (number_of_initial_sequences * nb_to_gen, max_len)

        Case 2:
            - src: (number_of_initial_conditions, nb_pts, pts_dim)
            - cut_seq_ids: (number_of_initial_conditions, number_of_initial_sequences, seq_len_so_far)
            - Returns: (number_of_initial_conditions, number_of_initial_sequences * nb_to_gen, max_len)
        """
        device = src.device
        if max_len is None:
            max_len = self.attention_size + 1

        # Determine input case
        if src.dim() == 2 and cut_seq_ids.dim() == 2:
            # Case 1: Single source, multiple sequences
            number_of_initial_sequences, seq_len_so_far = cut_seq_ids.shape
            cut_seq_ids = cut_seq_ids.repeat_interleave(nb_to_gen, dim=0)  # (number_of_initial_sequences * nb_to_gen, seq_len_so_far)

            # Encode source (same for all sequences)
            src_emb = self.src_input_emb(src).reshape(1, src.size(0), -1)  # (1, nb_pts, pts_dim * d_embed)
            src_emb = self.dim_reduce(src_emb)  # (1, nb_pts, d_model)
            memory = self.encoder(src_emb).expand(cut_seq_ids.size(0), -1, -1)  # (number_of_initial_sequences * nb_to_gen, nb_pts, d_model)

        elif src.dim() == 3 and cut_seq_ids.dim() == 3:
            # Case 2: Multiple sources, multiple sequences
            number_of_initial_conditions, number_of_initial_sequences, seq_len_so_far = cut_seq_ids.shape
            cut_seq_ids = cut_seq_ids.view(-1, seq_len_so_far).repeat_interleave(nb_to_gen, dim=0)  # (number_of_initial_conditions * number_of_initial_sequences * nb_to_gen, seq_len_so_far)

            # Encode each source separately
            src_emb = self.src_input_emb(src)  # (number_of_initial_conditions, nb_pts, pts_dim, d_embed)
            src_emb = src_emb.reshape(src.shape[0], src.shape[1], -1)  # (number_of_initial_conditions, nb_pts, pts_dim * d_embed)
            src_emb = self.dim_reduce(src_emb)  # (number_of_initial_conditions, nb_pts, d_model)
            memory = self.encoder(src_emb)  # (number_of_initial_conditions, nb_pts, d_model)

            # Expand memory for each initial sequence and nb_to_gen
            memory = memory.unsqueeze(1).expand(-1, number_of_initial_sequences, -1, -1)  # (number_of_initial_conditions, number_of_initial_sequences, nb_pts, d_model)
            memory = memory.reshape(-1, memory.shape[2], memory.shape[3])  # (number_of_initial_conditions * number_of_initial_sequences, nb_pts, d_model)
            memory = memory.repeat_interleave(nb_to_gen, dim=0)  # (number_of_initial_conditions * number_of_initial_sequences * nb_to_gen, nb_pts, d_model)

        else:
            raise ValueError(f"Invalid src/cut_seq_ids dimensions: src {src.shape}, cut_seq_ids {cut_seq_ids.shape}")

        # Initialize outputs and probability sums
        outputs = torch.zeros(cut_seq_ids.size(0), max_len, dtype=torch.long, device=device)
        outputs[:, :seq_len_so_far] = cut_seq_ids
        log_prob_sums = torch.zeros(cut_seq_ids.size(0), device=device)

        # Track active sequences
        active_indices = torch.arange(cut_seq_ids.size(0), device=device)
        lengths = torch.ones(cut_seq_ids.size(0), dtype=torch.long, device=device) * seq_len_so_far

        for step in range(seq_len_so_far, max_len):
            if len(active_indices) == 0:
                break  # Stop if all sequences are complete

            # Get active sequences
            current_seqs = outputs[active_indices, :step]

            # Compute embeddings
            tgt = self.tgt_input_emb(current_seqs)
            pos_embed = torch.arange(0, step, device=device).unsqueeze(0).repeat(len(active_indices), 1)
            tgt = tgt + self.tgt_position_emb(pos_embed)
            tgt = tgt.float()

            # Generate causal mask if needed
            if self.tgt_mask is None or self.tgt_mask.size(0) != step:
                self.tgt_mask = self.get_mask(step).to(device)

            # Decode
            dec_out = self.decoder(
                tgt,
                memory[active_indices],
                tgt_mask=self.tgt_mask,
                tgt_is_causal=True
            )

            # Get next token logits
            next_token_logits = self.output_layer(dec_out[:, -1, :]) / temperature
            log_probs = F.log_softmax(next_token_logits, dim=-1)

            # Sample next tokens
            sampled_tokens = torch.multinomial(torch.exp(log_probs), 1).squeeze(1)

            # Update sequences
            outputs[active_indices, step] = sampled_tokens
            log_prob_sums[active_indices] += log_probs[torch.arange(len(active_indices), device=device), sampled_tokens]

            # Find which sequences finished
            eos_mask = sampled_tokens == self.eos_id
            active_indices = active_indices[~eos_mask]  # Remove finished sequences
            lengths[active_indices] = step + 1  # Update lengths

        # Convert log probabilities to normal probabilities
        probabilities = torch.exp(log_prob_sums).tolist()

        # Reshape outputs and probabilities based on the case
        if src.dim() == 3:
            outputs = outputs.view(number_of_initial_conditions, number_of_initial_sequences * nb_to_gen, max_len)
            probabilities = torch.tensor(probabilities).view(number_of_initial_conditions, number_of_initial_sequences * nb_to_gen).tolist()

        return outputs, probabilities


    @torch.no_grad()
    def continue_beam_search(
        self,
        src: torch.Tensor,
        cut_seq_ids: torch.Tensor,
        nb_to_gen: int = 5,
        max_len: int = None, 
        temperature: float = 1.0,
    ):
        beam = nb_to_gen
        device = src.device
        if max_len is None:
            max_len = self.attention_size + 1

        number_of_initial_conditions, number_of_initial_sequences, seq_len_so_far = cut_seq_ids.shape
        batch_size = number_of_initial_conditions

        all_outputs = []
        all_probabilities = []

        for b in range(batch_size):
            src_b = src[b:b+1]  # (1, nb_pts, pts_dim)
            cut_seq_ids_b = cut_seq_ids[b]  # (number_of_initial_sequences, seq_len_so_far)

            num_seqs = number_of_initial_sequences
            # Initially, we expand each initial sequence with beam candidates,
            # giving num_seqs * beam beams.
            log_prob_sums = torch.zeros(num_seqs * beam, device=device)

            # Prepare encoder memory for the current batch element.
            src_emb = self.src_input_emb(src_b)
            src_emb = src_emb.reshape(1, src_b.shape[1], -1)
            src_emb = self.dim_reduce(src_emb)
            memory = self.encoder(src_emb).expand(num_seqs, -1, -1)

            # Prepare an outputs tensor for storing sequences.
            outputs = torch.full((num_seqs * beam, max_len), 0, dtype=torch.long, device=device)
            lengths = torch.full((num_seqs * beam,), seq_len_so_far, dtype=torch.long, device=device)
            # beam_parent tracks which encoder memory row each beam uses.
            beam_parent = torch.arange(num_seqs, device=device).unsqueeze(1).repeat(1, beam).view(num_seqs * beam)

            # === INITIAL STEP ===
            # Embed the initial target sequences.
            tgt = self.tgt_input_emb(cut_seq_ids_b)
            pos_embed = torch.arange(0, seq_len_so_far, device=device).unsqueeze(0).repeat(num_seqs, 1)
            tgt = tgt + self.tgt_position_emb(pos_embed)
            tgt = tgt.float()
            dec_out = self.decoder(tgt, memory, tgt_is_causal=True)
            next_token_logits = self.output_layer(dec_out[:, -1, :]) / temperature
            log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
            top_log_probs, top_tokens = log_probs.topk(beam, dim=-1)  # shape: (num_seqs, beam)

            expanded_seqs = cut_seq_ids_b.unsqueeze(1).repeat(1, beam, 1).view(num_seqs * beam, seq_len_so_far)
            expanded_seqs = torch.cat([expanded_seqs, top_tokens.view(-1, 1)], dim=-1)
            outputs[:, :seq_len_so_far + 1] = expanded_seqs
            log_prob_sums = top_log_probs.view(-1)
            lengths[:] = seq_len_so_far + 1

            # In this revised version we maintain separate tensors for active beams and finished beams.
            active_outputs = outputs.clone()          # shape: (num_beams, max_len)
            active_log_probs = log_prob_sums.clone()    # shape: (num_beams,)
            active_beam_parent = beam_parent.clone()    # shape: (num_beams,)
            active_lengths = lengths.clone()            # shape: (num_beams,)

            finished_outputs = []       # List to store finished (EOS-generated) beams.
            finished_log_probs = []     # Their corresponding log probabilities.

            # === LOOP OVER TIME STEPS ===
            for step in range(seq_len_so_far + 1, max_len):
                # print("\n\n")
                if active_outputs.size(0) == 0:
                    break  # No active beams left to expand.

                # Get current active sequences (only up to the current step).
                current_seqs = active_outputs[:, :step]
                tgt = self.tgt_input_emb(current_seqs)
                pos_embed = torch.arange(0, step, device=device).unsqueeze(0).repeat(current_seqs.size(0), 1)
                tgt = tgt + self.tgt_position_emb(pos_embed)
                tgt = tgt.float()

                # Select the appropriate encoder memory row for each beam.
                mem_for_active = memory[active_beam_parent]
                if self.tgt_mask is None or self.tgt_mask.size(0) != step:
                    self.tgt_mask = self.get_mask(step).to(device)
                dec_out = self.decoder(
                    tgt,
                    mem_for_active,
                    tgt_mask=self.tgt_mask,
                    tgt_is_causal=True
                )
                next_token_logits = self.output_layer(dec_out[:, -1, :]) / temperature
                log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                top_log_probs, top_tokens = log_probs.topk(beam, dim=-1)  # shape: (k, beam) where k = number of active beams

                # Compute new cumulative log probabilities for each candidate.
                # new_log_probs has shape (k, beam).
                new_log_probs = active_log_probs.unsqueeze(1) + top_log_probs
                # Flatten candidate dimensions: (k*beam,)
                new_log_probs_flat = new_log_probs.view(-1)
                new_tokens = top_tokens.view(-1)

                # print(f"{'active_beams indices (implicit)'}: {active_outputs.size(0)} beams")
                # Expand active sequences: each active beam is repeated beam times.
                expanded_active = active_outputs.repeat_interleave(beam, dim=0)
                expanded_active[:, step] = new_tokens

                # Similarly expand beam_parent.
                expanded_parent = active_beam_parent.repeat_interleave(beam)

                # Among the k*beam candidates, keep only the top k candidates (where k = number of active beams).
                k = active_outputs.size(0)
                best_scores, best_indices = torch.topk(new_log_probs_flat, k, dim=-1)
                new_active = expanded_active[best_indices]
                new_active_log_probs = best_scores
                new_active_parent = expanded_parent[best_indices]

                # print(f"OLD step={step}, active outputs (first two rows) =\n{active_outputs[:2, :step+2]}")
                # print(f"NEW step={step}, expanded candidates (first two rows) =\n{expanded_active[:2, :step+2]}")
                # print(f"{new_active_log_probs=}")
                # print(f"{best_indices=}")
                # print(f"BEST IDS: step={step}, new active outputs (first two rows) =\n{new_active[:2, :step+2]}")

                # Identify which of the new candidates are finished (i.e. generated <EOS>).
                finished_mask = new_active[:, step] == self.eos_id
                if finished_mask.any():
                    # Append finished beams.
                    finished_outputs.append(new_active[finished_mask])
                    finished_log_probs.append(new_active_log_probs[finished_mask])
                    # Remove finished beams from the active set.
                    keep_mask = ~finished_mask
                    new_active = new_active[keep_mask]
                    new_active_log_probs = new_active_log_probs[keep_mask]
                    new_active_parent = new_active_parent[keep_mask]

                # Update active beams for next iteration.
                active_outputs = new_active
                active_log_probs = new_active_log_probs
                active_beam_parent = new_active_parent
                # Update the beam lengths for the active beams.
                if active_outputs.size(0) > 0:
                    active_lengths[:active_outputs.size(0)] = step + 1

            if finished_outputs:
                finished_outputs = torch.cat(finished_outputs, dim=0)
                finished_log_probs = torch.cat(finished_log_probs, dim=0)
                if active_outputs.size(0) > 0:
                    # Combine finished and active beams.
                    final_outputs = torch.cat([finished_outputs, active_outputs], dim=0)
                    final_log_probs = torch.cat([finished_log_probs, active_log_probs], dim=0)
                else:
                    final_outputs = finished_outputs
                    final_log_probs = finished_log_probs
            else:
                final_outputs = active_outputs
                final_log_probs = active_log_probs

            # Optionally, sort and keep exactly num_seqs * beam beams.
            num_beams = num_seqs * beam
            final_scores, sort_indices = torch.topk(final_log_probs, num_beams, dim=-1)
            final_outputs = final_outputs[sort_indices]
            final_probs = torch.exp(final_scores)


            # Reshape to (1, num_beams, max_len) per batch element.
            all_outputs.append(final_outputs.unsqueeze(0))
            all_probabilities.append(final_probs.unsqueeze(0))
        
        final_outputs = torch.cat(all_outputs, dim=0)
        final_probabilities = torch.cat(all_probabilities, dim=0)
        return final_outputs, final_probabilities.tolist()


