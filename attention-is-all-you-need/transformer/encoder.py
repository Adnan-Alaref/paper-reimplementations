from transformer.feed_forward import PositionWiseFeedForward
from transformer.attention import MultiHeadAttention
import torch.nn as nn
import torch


# ── Encoder Layer ────────────────────────────────────────────────────────
class EncoderLayer(nn.Module):
  """
    Transformer encoder layer using Post-LayerNorm.

    Steps:
       1. Self-attention:
           x = LayerNorm(x + Dropout(Self-Attention(x)))

       2. Position-wise feed-forward network:
           x = LayerNorm(x + Dropout(FFN(x)))
  """

  def __init__(self,
               d_model: int,
               num_heads: int,
               expansion_dim: int,
               dropout_prop: float) -> None:

    super().__init__()

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)

    self.dropout = nn.Dropout(dropout_prop)

    self.self_attn = MultiHeadAttention(d_model, num_heads)
    self.feed_forward = PositionWiseFeedForward(d_model, expansion_dim)

  def forward(
      self,
      x: torch.Tensor,
      mask: torch.Tensor | None = None
    )-> torch.Tensor:

    # ── Self-Attention ──
    attn_output = self.self_attn(x, x, x, mask)
    x = self.norm1(x + self.dropout(attn_output))

    # ── Feed-Forward Network ──
    ffn_output = self.feed_forward(x)
    x = self.norm2(x + self.dropout(ffn_output))

    return x