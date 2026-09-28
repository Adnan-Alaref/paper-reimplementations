from __future__ import annotations

from src.collect_data import preprocess_languages
from src.vocabulary import Vocabulary
from typing import TYPE_CHECKING
from src.imports import *


# ── Build translation samples ────────────────────────────────────────────────────────
def build_samples(source_path: str,
                  target_path: str
                  )->Tuple[List[List[int]], List[List[int]], Vocabulary, Vocabulary]:
  """
    Build encoded source and target samples and their vocabularies.
  """

  source_sentences, target_sentences = preprocess_languages(
        source_path,
        target_path
  )

  source_vocab = Vocabulary(source_sentences)
  target_vocab = Vocabulary(
    target_sentences,
    is_target=True
  )

  source_samples = source_vocab.encode(source_sentences)
  target_samples = target_vocab.encode(target_sentences)

  return (
    source_samples,
    target_samples,
    source_vocab,
    target_vocab
  )


# ── Build Custom Dataset ────────────────────────────────────────────────────────
class TranslationDataset(Dataset):
  def __init__(self, source_path: str, target_path: str) -> None:
    super().__init__()

    (
      self.source_samples,
      self.target_samples,
      self.source_vocab,
      self.target_vocab
    ) = build_samples(
      source_path,
      target_path
    )


    if len(self.source_samples) != len(self.target_samples):
      raise ValueError(
        "Source and target must contain the same number "
        "of samples."
      )

  def __len__(self)->int:
    """Return the number of translation pairs."""
    return len(self.source_samples)

  @property
  def source_lengths(self) -> List[int]:
    """
      Return a list containing the length of each source sequence.

      The source sequences form a jagged/ragged list because each
      sequence may have a different number of tokens.
    """
    return [
      len(sentence)
      for sentence in self.source_samples
    ]

  @property
  def target_lengths(self) -> List[int]:
    """
      Return a list containing the length of each target sequence.

      The target sequences form a jagged/ragged list because each
      sequence may have a different number of tokens.
    """
    return [
      len(sentence)
      for sentence in self.target_samples
    ]

  @property
  def is_empty(self)-> bool:
    """Return whether the dataset contains no samples."""
    return len(self) == 0

  def __getitem__(self, index)->Tuple[List[int], List[int]]:
    """Return one source-target translation pair."""
    return (
      self.source_samples[index],
      self.target_samples[index]
    )

  def summary(self)-> pd.DataFrame:
    """Return basic dataset statistics as a table."""

    return pd.DataFrame({
        "Dataset": ["Source", "Target"],
        "Samples": [
            len(self.source_samples),
            len(self.target_samples)
        ],
        "Min Length": [
            min(self.source_lengths),
            min(self.target_lengths)
        ],
        "Max Length": [
            max(self.source_lengths),
            max(self.target_lengths)
        ]
    })