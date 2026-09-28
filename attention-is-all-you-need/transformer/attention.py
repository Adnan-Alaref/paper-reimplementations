import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================
# Multi-Head Attention
# ==================================
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, num_heads: int) -> None:
    super(MultiHeadAttention, self).__init__()

    # ── Initialize dimensions ─────────────────────────
    self.d_model = d_model # Model's dimension
    self.num_heads = num_heads # Number of attention heads
    self.head_dim = d_model // num_heads # Dimension of each head's key, query, and value

    # ── Scaling factor ────────────────────────────
    self.scale = self.head_dim ** 0.5

    # Ensure that the model dimension (d_model) is divisible by the number of heads
    if self.head_dim * num_heads != d_model:
      raise ValueError(
        f"Embedding_dim ({d_model}) must be divisible by num heads ({num_heads})."
      )

    # ── Linear projections ──────────────────────────────────
    self.W_q = nn.Linear(d_model, d_model)
    self.W_k = nn.Linear(d_model, d_model)
    self.W_v = nn.Linear(d_model, d_model)

    # ── Output projection ──────────────────────────────────
    self.fc_proj = nn.Linear(d_model, d_model)


  def split_heads(self, M: torch.Tensor)-> torch.Tensor:
    """
      Split the embedding dimension into multiple attention heads.
      Input:
          M: (B, seq_len, d_model)
      Output:
          M: (B, num_heads, seq_len, head_dim)
    """

    B, seq_len, _ = M.shape
    return M.view(
      B,
      seq_len,
      self.num_heads,
      self.head_dim).transpose(1,2)


  def combine_heads(self, M: torch.Tensor)-> torch.Tensor:
    """
      Combine multiple attention heads.
      Input:
          M: (B, num_heads, seq_len, head_dim)
      Output:
        M: (B, seq_len, d_model)
    """

    B, _, seq_len, _ = M.shape
    return M.transpose(1,2).reshape(
      B,
      seq_len,
      self.d_model
    )


  def scaled_dot_product_attention(self,
                                   query: torch.Tensor,
                                   key: torch.Tensor,
                                   value: torch.Tensor,
                                   mask: torch.Tensor | None = None,
                                   )-> torch.Tensor:

    """Compute scaled dot-product attention."""

    # ── QKᵀ / √dₖ ──────────────────────
    attn_scores = torch.matmul(
      query,
      key.transpose(-2,-1)
    ) / self.scale

    # ── Apply mask ─────────────────────
    if mask is not None:
      attn_scores = attn_scores.masked_fill(
        ~mask,
        float("-inf")
      )

    # ── Convert scores → probabilities ─────────────
    attn_weights = F.softmax(
      attn_scores,
      dim=-1
    )

    # ── Weighted sum of values ─────────
    output = torch.matmul(
      attn_weights,
      value
    )

    return output


  def forward(self,
              Q: torch.Tensor,
              K: torch.Tensor,
              V: torch.Tensor,
              mask: torch.Tensor | None = None
              )->torch.Tensor:
    """
      Multi-head attention.
      Architecture:
            Projection → Split → Attention → Combine → Projection
    """

    # ── Linear projections ──────────
    query = self.W_q(Q)
    key = self.W_k(K)
    value = self.W_v(V)

    # ── Split into attention heads ────────────────
    query = self.split_heads(query)
    key = self.split_heads(key)
    value = self.split_heads(value)

    # ── Scaled dot-product attention ─────────────────────
    attn_output = self.scaled_dot_product_attention(
      query,
      key,
      value,
      mask
    )

    # ── Combine heads ─────────────────────────
    output = self.combine_heads(
      attn_output
    )

    # ── Output projection ────────────────────────
    return self.fc_proj(
      output
    )