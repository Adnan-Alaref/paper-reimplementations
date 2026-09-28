from transformer.positional_encoding import PositionalEncoding
from transformer.encoder import EncoderLayer
from transformer.decoder import DecoderLayer
import torch.nn as nn
import torch

# ── Transformer ────────────────────────────────────────────────────────
class Transformer(nn.Module):
  def __init__(
      self,
      src_vocab_size: int,
      tgt_vocab_size: int,
      max_seq_len: int,
      d_model: int,
      expansion_dim: int,
      num_heads: int,
      num_layers: int,
      dropout: float,
      pad_idx: int,
    ) -> None:

    super().__init__()

    self.pad_idx = pad_idx
    self.encoder_embedding = nn.Embedding(
      src_vocab_size,
      d_model,
      padding_idx=pad_idx
    )

    self.decoder_embedding = nn.Embedding(
      tgt_vocab_size,
      d_model,
      padding_idx=pad_idx
    )

    # ── Positional Encoding ───────────────────────────────────────────
    # The same max_seq_len is used for both source and target sequences,
    # so a single positional encoding module can be shared by both.
    self.positional_encoding = PositionalEncoding(
      max_seq_len,
      d_model
    )

    self.encoder_layers = nn.ModuleList([
      EncoderLayer(
        d_model,
        num_heads,
        expansion_dim,
        dropout
      )
      for _ in range(num_layers)
    ])

    self.decoder_layers = nn.ModuleList([
      DecoderLayer(
        d_model,
        num_heads,
        expansion_dim,
        dropout
      )
      for _ in range(num_layers)
    ])

    self.fc_proj = nn.Linear(
      d_model,
      tgt_vocab_size
    )

    self.dropout = nn.Dropout(
      dropout
    )

  def create_masks(self,
                   src: torch.Tensor,
                   tgt: torch.Tensor,
                  )->tuple[torch.Tensor, torch.Tensor]:
    """
      Create attention masks for an encoder-decoder Transformer.

      The source padding mask prevents attention to <PAD> tokens in:
          1. Encoder self-attention.
          2. Decoder cross-attention.

      The target mask combines:
          1. Target padding mask: prevents attention to <PAD> tokens.
          2. Target causal mask: prevents attention to future tokens.

      Args:
          src: Source token IDs of shape (batch_size, src_len).
          tgt: Target token IDs of shape (batch_size, tgt_len).

      Returns:
          A tuple containing:
              src_padding_mask:
                  Boolean mask of shape (batch_size, 1, src_len).
                    1- True  → valid token
                    2- False → <PAD>

              tgt_mask:
                  Boolean mask of shape (batch_size, tgt_len, tgt_len).
                    1- True  → current/past token is visible
                    2- False → future token is hidden or a <PAD> token.
    """

    # ── Padding masks ───────────────────────────────────────
    src_padding_mask = (src != self.pad_idx)[:, None, None, :]
    tgt_padding_mask = (tgt != self.pad_idx)[:, None, None, :]

    # ── Causal mask ─────────────────────────────────────────
    tgt_len = tgt.shape[1]

    tgt_causal_mask = torch.tril(
      torch.ones(
        1,
        1,
        tgt_len,
        tgt_len,
        dtype=torch.bool,
        device= tgt.device
      )
    )

    # ── Combined target mask ────────────────────────────────
    tgt_mask = tgt_causal_mask & tgt_padding_mask

    return src_padding_mask, tgt_mask

  def forward(self,
              src:torch.Tensor,
              tgt: torch.Tensor
             ) -> torch.Tensor:
    """
      Args:
          src: Source token IDs, shape (B, S).
          tgt: Target token IDs, shape (B, T).

      Returns:
        Output logits of shape (B, T, tgt_vocab_size).
    """

    src_mask, tgt_mask = self.create_masks(src, tgt)

    # ── Embedding + Positional Encoding ──────────────────
    src_embedded = self.dropout(
      self.positional_encoding(
        self.encoder_embedding(src)
      )
    )

    tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

    # ── Encoder ──────────────────────────────────────────
    encoder_output = src_embedded

    for enc_layer in self.encoder_layers:
      encoder_output = enc_layer(
        encoder_output,

        # Source padding mask.
        # Used by self-attention to prevent the encoder
        # from attending to <PAD> tokens in the source.
        src_mask
      )

    # ── Decoder ──────────────────────────────────────────
    decoder_output = tgt_embedded

    for dec_layer in self.decoder_layers:
      decoder_output = dec_layer(
        decoder_output,
        # same final encoder representation
        encoder_output,

        # Source padding mask.
        # Used by cross-attention to prevent the decoder
        # from attending to <PAD> tokens in the source.
        src_mask,

        # Target mask = padding mask + causal mask.
        # Used by decoder self-attention to prevent:
        # 1. attending to <PAD> tokens
        # 2. attending to future target tokens
        tgt_mask
      )

    # ── Output Projection ────────────────────────────────
    output = self.fc_proj(decoder_output)

    return output