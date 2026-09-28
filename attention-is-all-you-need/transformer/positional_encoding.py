import math
import torch
import torch.nn as nn

# ── Positional Encoding ────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
  """
    Generate sinusoidal positional encodings for Transformer inputs.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
  """

  def __init__(self, max_seq_len: int, d_model: int) -> None:
    super(PositionalEncoding, self).__init__()

    if d_model % 2 != 0:
      raise ValueError(
        "d_model must be even to pair sine and cosine dimensions "
        "as in the original Transformer paper."
      )

    PE = torch.zeros(max_seq_len, d_model)

    pos = torch.arange(max_seq_len)[:, None]

    i = torch.arange(0, d_model, 2)

    div_term = torch.exp(-( i / d_model) * math.log(10_000))

    angle = pos * div_term

    PE[:, 0::2] = torch.sin(angle)
    PE[:, 1::2] = torch.cos(angle)

    self.register_buffer("PE", PE)

  def forward(self, x: torch.Tensor)-> torch.Tensor:
    seq_len = x.shape[1]
    return x + self.PE[:seq_len]