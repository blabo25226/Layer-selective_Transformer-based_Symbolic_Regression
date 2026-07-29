import torch


@torch.no_grad()
def accuracy(output, tgt_output, ignore_ids: list):
    """
    Computes the accuracy of token predictions, ignoring specified token IDs.

    Args:
        output (torch.Tensor): The model output, typically logits with shape (batch_size, seq_len, vocab_size) or predictions (batch_size, seq_len).
        tgt_output (torch.Tensor): The target output with token IDs.
        ignore_ids (list): A list of token IDs to ignore during accuracy calculation (e.g., PAD, SOS tokens).

    Returns:
        float: The accuracy as a ratio of correctly predicted tokens to total valid tokens.
    """
    if output.dim() == 3:
        # Get the most probable token indices along the vocabulary dimension
        selected = torch.argmax(output, dim=-1)
    else:
        selected = output

    # Create a mask for valid target tokens (not in ignore_ids)
    mask = torch.ones_like(tgt_output, dtype=torch.bool)
    for ignore_id in ignore_ids:
        mask &= (tgt_output != ignore_id)

    # Count correctly predicted tokens
    num_same_elements = torch.sum((selected == tgt_output) * mask)
    # Count total valid target tokens
    num_total_elements = torch.sum(mask)

    # Avoid division by zero
    if num_total_elements == 0:
        print("Accuracy - num_total_elements is zero!")
        return torch.tensor(0.0, device=output.device)

    return num_same_elements / num_total_elements
