from src.imports import *

class Vocabulary:
  def __init__(self, sentences: List[str], is_target: bool = False)->None:
    """
      Build a vocabulary from a list of sentences.

      Args:
        sentences: Sentences used to build the vocabulary.
        is_target: Whether this vocabulary belongs to the target language.
                   Target vocabularies use BOS and EOS tokens.
    """

    self.is_target = is_target

    # ── Special tokens used by the source vocabulary.
    self.PAD_TOKEN = "<PAD>"
    self.UNK_TOKEN = "<UNK>"

    special_tokens = [
      self.PAD_TOKEN,
      self.UNK_TOKEN,
    ]

    if is_target:
      # ── Special tokens used by the target vocabulary.
      self.BOS_TOKEN = "<BOS>"
      self.EOS_TOKEN = "<EOS>"

      special_tokens.extend([
        self.BOS_TOKEN,
        self.EOS_TOKEN
      ])

    # ── Extract all unique words while preserving their order.
    unique_words = dict.fromkeys(
      " ".join(sentences).split()
    )

    # ── Build vocabulary
    self.words = [
      *special_tokens,
      *[
        word
        for word in unique_words
        if word not in special_tokens
      ]
    ]

    # ── Map each word to its integer index.
    self.word_to_idx = {
      word : idx
      for idx, word in enumerate(self.words)
    }

    # ── Map each integer index back to its corresponding word.
    self.idx_to_word = {
      idx : word
      for word, idx in self.word_to_idx.items()
    }

    # ── Store special-token indices for convenient access.
    self.PAD_IDX = self.word_to_idx[self.PAD_TOKEN]
    self.UNK_IDX = self.word_to_idx[self.UNK_TOKEN]

    if is_target:
      self.BOS_IDX = self.word_to_idx[self.BOS_TOKEN]
      self.EOS_IDX = self.word_to_idx[self.EOS_TOKEN]


  def __len__(self)-> int:
    """Return the total number of tokens in the vocabulary."""
    return len(self.words)


  def __contains__(self, word: str) -> bool:
    """Return True if the word exists in the vocabulary."""
    return word in self.word_to_idx


  def sentence_to_indices(self, sentence: str)->List[int]:
    """
      Convert a sentence into token indices.

      Target sentences automatically receive BOS and EOS tokens.
      Unknown words are mapped to UNK_IDX.
    """
    tokens = sentence.split()

    if self.is_target:
      tokens = [
        self.BOS_TOKEN,
        *tokens,
        self.EOS_TOKEN
      ]

    encoded = [
      self.word_to_idx.get(
        token,
        self.UNK_IDX
      )
      for token in tokens
    ]

    return encoded


  def indices_to_sentence(
    self,
    indices: List[int],
    ignore_invalid: bool = True,
    remove_special_tokens: bool = True
    ) -> str:
    """
      Convert token indices back into a sentence.

      For target sequences, decoding stops at EOS.
      PAD and BOS tokens are optionally removed.
      Invalid indices have two choices
        1- Ignored
        2- raise `ValueError` for debug perpose
          because they should not normally appear in a valid encoded sequence.
    """
    words = []

    for idx in indices:

      # ── Stop when the target sequence reaches EOS.
      if self.is_target and idx == self.EOS_IDX:
        break

      # Skip padding and BOS during text reconstruction.
      if remove_special_tokens:
        if idx == self.PAD_IDX:
          continue

        if self.is_target and idx == self.BOS_IDX:
          continue

      # Ignore invalid indices.
      if idx not in self.idx_to_word:
        if ignore_invalid:
          continue

        else:
          raise ValueError(f"Invalid token index: {idx}")

      words.append(self.idx_to_word[idx])

    return " ".join(words)


  def encode(self, sentences: List[str])->List[List[int]]:
    """
      Convert a list of sentences into sequences of token IDs.
      Each sentence is tokenized into words, and each word is
      mapped to its corresponding ID in the vocabulary.
    """
    all_encoded = []

    for sent in sentences:
      all_encoded.append(self.sentence_to_indices(sent))

    return all_encoded


  def decode(self, encoded_sentences: List[List[int]])->List[List[str]]:
    """
      Convert a list of token-ID sequences back into words.
      Each sequence of token IDs is mapped to its corresponding
        words using the reverse vocabulary mapping.
    """
    all_decoded = []

    for sent in encoded_sentences:
      all_decoded.append(self.indices_to_sentence(sent))

    return all_decoded

  def save(self, path:str)->None:
    """Save the vocabulary state to disk."""
    data = {
      "words": self.words,
      "word_to_idx": self.word_to_idx,
      "is_target": self.is_target,
      "PAD_TOKEN": self.PAD_TOKEN,
      "UNK_TOKEN": self.UNK_TOKEN,
    }

    if self.is_target:
      data.update({
        "BOS_TOKEN":self.BOS_TOKEN,
        "EOS_TOKEN": self.EOS_TOKEN
      })

    # Create dir
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saved Vocabulary: {path}")
    torch.save(data,path)


  @classmethod
  def load(cls, path:str)-> "Vocabulary":
    """Load a previously saved vocabulary from disk."""

    data = torch.load(path)

    # Create an instance without calling __init__.
    vocab = cls.__new__(cls)

    # ── Restore basic state ──────────────
    vocab.is_target = data["is_target"]

    vocab.words = data["words"]
    vocab.word_to_idx = data["word_to_idx"]

    # Rebuild the reverse mapping.
    vocab.idx_to_word = {
        idx: word
        for word, idx in vocab.word_to_idx.items()
    }

    # ── Restore special tokens ───────────
    vocab.PAD_TOKEN = data["PAD_TOKEN"]
    vocab.UNK_TOKEN = data["UNK_TOKEN"]

    vocab.PAD_IDX = vocab.word_to_idx[vocab.PAD_TOKEN]
    vocab.UNK_IDX = vocab.word_to_idx[vocab.UNK_TOKEN]

    if vocab.is_target:
      vocab.BOS_TOKEN = data["BOS_TOKEN"]
      vocab.EOS_TOKEN = data["EOS_TOKEN"]

      vocab.BOS_IDX = vocab.word_to_idx[vocab.BOS_TOKEN]
      vocab.EOS_IDX = vocab.word_to_idx[vocab.EOS_TOKEN]

    return vocab