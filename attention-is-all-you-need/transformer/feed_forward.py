import torch
import torch.nn as nn

# ── Position-wise Feed-Forward Network ──────────────────────────────
class PositionWiseFeedForward(nn.Module):
  """
    Position-wise feed-forward network used in the Transformer.

    Applies two linear projections with a non-linear activation
    independently to each token position.
  """

  def __init__(self, d_model: int, expansion_dim: int) -> None:
    super().__init__()

    self.fc1_proj = nn.Linear(d_model, expansion_dim)
    self.fc2_proj = nn.Linear(expansion_dim, d_model)
    self.relu = nn.ReLU()

  def forward(self, x:torch.Tensor)-> torch.Tensor:
    return self.fc2_proj(
      self.relu(
        self.fc1_proj(x)
      )
    )