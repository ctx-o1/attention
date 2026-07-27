import torch


def generate_causal_mask(sequence_length: int, device=None) -> torch.Tensor:
    """Return a boolean mask where True marks future positions."""
    return torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )
