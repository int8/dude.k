import torch
from torch import nn


class RandomFrameSwap(nn.Module):
    def __init__(self, p: float = 0.05):
        """
        Initializes the RandomFrameSwap module with a given swap probability.

        Args:
            p (float): Probability of swapping each pair of consecutive frames.
        """
        super(RandomFrameSwap, self).__init__()
        self.p = p

    def forward(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        """Performs random frame swaps on the input clip tensor.

        Args:
            clip_tensor (torch.Tensor): Input frames tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output frames tensor with frames randomly swapped.
        """
        N = clip_tensor.shape[0]
        device = clip_tensor.device

        # Initialize an index mapping and a swapped mask
        index_mapping = torch.arange(N, device=device)
        swapped = torch.zeros(N, dtype=torch.bool, device=device)

        # Generate random swap decisions for each possible swap position
        swap_mask = torch.rand(N - 1, device=device) < self.p

        # Find indices where swaps can be performed (no overlapping)
        eligible_swaps = swap_mask & (~swapped[:-1]) & (~swapped[1:])
        indices_to_swap = torch.where(eligible_swaps)[0]

        # Mark indices as swapped
        swapped[indices_to_swap] = True
        swapped[indices_to_swap + 1] = True

        # Perform the swaps using advanced indexing
        index_mapping_copy = index_mapping.clone()
        index_mapping[indices_to_swap] = index_mapping_copy[indices_to_swap + 1]
        index_mapping[indices_to_swap + 1] = index_mapping_copy[indices_to_swap]
        # Reorder the clip_tensor according to the index mapping
        clip_tensor_swapped = clip_tensor[index_mapping]
        return clip_tensor_swapped
