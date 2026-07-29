import torch
from src.ConfigClasses import ConfigFormula

def find_specific_id(tensor: torch.Tensor, target_id: int) -> int | None:
    """
    Finds the position of a specific ID in a tensor. Returns None if the ID is not found.

    Args:
        tensor (torch.Tensor): The tensor to search.
        target_id (int): The ID to find in the tensor.

    Returns:
        int | None: The index of the target ID in the tensor, or None if not found.
    """
    for idx, value in enumerate(tensor):
        if value.item() == target_id:
            return idx
    return None

def get_number_of_candidates(config: ConfigFormula, tensor: torch.Tensor, pad_id: int) -> int:
    """
    Computes the number of candidates in a tensor, taking into account padding and noise configuration.

    Args:
        config (ConfigFormula): Configuration object containing relevant settings.
        tensor (torch.Tensor): The tensor containing candidates.
        pad_id (int): The ID representing padding in the tensor.

    Returns:
        int: The number of candidates.
    """
    result_size_adjustment = 1 if config.NOISY else 0

    # Find the first occurrence of the pad ID
    first_pad_index = find_specific_id(tensor, pad_id)

    if first_pad_index is None:
        return len(tensor) - result_size_adjustment

    return first_pad_index 