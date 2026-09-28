from transformer.feed_forward import PositionWiseFeedForward
from transformer.attention import MultiHeadAttention
import torch.nn as nn
import torch

# ── Decoder Layer ──────────────────────────────────────
class DecoderLayer(nn.Module):
  """
    Transformer decoder layer using Post-LayerNorm.

    Steps:
        1. Masked self-attention:
           x = LayerNorm(x + Dropout(Self-Attention(x, x, x, tgt_mask)))

        2. Cross-attention:
           x = LayerNorm(x + Dropout(Cross-Attention(x, enc_output, enc_output, src_mask)))

        3. Position-wise feed-forward network:
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
    self.norm3 = nn.LayerNorm(d_model)

    self.dropout = nn.Dropout(dropout_prop)

    self.self_attn = MultiHeadAttention(d_model, num_heads)
    self.cross_attn = MultiHeadAttention(d_model, num_heads)
    self.feed_forward = PositionWiseFeedForward(d_model, expansion_dim)


  def forward(
      self,
      x: torch.Tensor,
      enc_output: torch.Tensor,
      src_mask: torch.Tensor,
      tgt_mask: torch.Tensor
    )-> torch.Tensor:

    # ── Masked Self-Attention ──
    attn_output = self.self_attn(x, x, x, tgt_mask)
    x = self.norm1(x + self.dropout(attn_output))

    # ── Cross-Attention ──
    attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
    x = self.norm2(x + self.dropout(attn_output))

    # ── Feed-Forward Network ──
    ffn_output = self.feed_forward(x)
    x = self.norm3(x + self.dropout(ffn_output))

    return x