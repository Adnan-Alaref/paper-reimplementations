from configs.config import ModelConfig, TrainingConfig
from src.dataset import TranslationDataset
from src.imports import *


# ── Create Training Configuration ─────────────────────────────
train_cfg = TrainingConfig()

# ── Build Custom collate_fn Function For DataLoader ────────────────────────────────────────────────────────
def custom_collate_fn(batch):
  """
    Collate variable-length translation samples into padded batch tensors.

    Each sample contains a source and target sequence represented as
    lists of token IDs. Source and target sequences are padded
    independently to the maximum length found in the current batch.

    Args:
        batch: A list of `(source_sequence, target_sequence)` pairs.

    Returns:
        A tuple containing:
            source_tensor: Padded source token IDs with shape
                `(batch_size, max_source_length)`.
            target_tensor: Padded target token IDs with shape
                `(batch_size, max_target_length)`.
  """

  # ── Batch size ──
  B = len(batch)

  # ── Get padding index ──
  pad_idx = config.pad_idx

  # ── Separate source and target sequences ──
  source_sequences, target_sequences = zip(*batch)

  # ── Find maximum sequence lengths ──
  max_src = max(map(len, source_sequences))
  max_tgt = max(map(len, target_sequences))

  # ── Create PAD-filled tensors ──
  source_tensor = torch.full(
    size=(B, max_src),
    dtype=torch.long,
    fill_value=pad_idx
  )

  target_tensor = torch.full(
    size=(B, max_tgt),
    dtype=torch.long,
    fill_value=pad_idx
  )

  # ── Copy real token IDs ──
  for i, (src_sequence, tgt_sequence) in enumerate(batch):
    source_tensor[i, :len(src_sequence)] = torch.tensor(
      src_sequence,
      dtype=torch.long
    )

    target_tensor[i, :len(tgt_sequence)] = torch.tensor(
      tgt_sequence,
      dtype=torch.long
    )

  # ── Return padded tensors ──
  return source_tensor, target_tensor



# ── Create DataLoader ──────────────────────────────

# Number of worker processes used for data loading
# > 0 enables parallel data loading
num_workers = min(4, os.cpu_count() or 1)

# Pin memory is only useful when training on CUDA
# It speeds up CPU → GPU memory transfers
pin_memory = train_cfg.device == "cuda"

loader_kwargs = dict(
  num_workers = num_workers,
  pin_memory = pin_memory,
  persistent_workers = num_workers > 0,
  # persistent_workers Keeps worker processes alive between epochs
  # Must be False when num_workers == 0
)

# prefetch_factor controls how many batches each worker preloads
# It is only valid when num_workers > 0
if num_workers > 0:
    loader_kwargs["prefetch_factor"] = 2



# ── Load translation dataset and vocabularies ──
translation_dataset = TranslationDataset(train_cfg.src_path, train_cfg.tgt_path)

# ── Get vocabularies from the dataset ──
src_vocab = translation_dataset.source_vocab
tgt_vocab = translation_dataset.target_vocab

# ── Define Vocabulary Paths ───────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

VOCAB_DIR = PROJECT_ROOT / "artifacts" / "vocab"

VOCAB_DIR.mkdir(parents=True, exist_ok=True)

src_vocab_path = VOCAB_DIR / "src_vocab.pt"
tgt_vocab_path = VOCAB_DIR / "tgt_vocab.pt"

# ── Save source and target vocabularies ──
src_vocab.save(src_vocab_path)
tgt_vocab.save(tgt_vocab_path)

# ── Create Model Configuration ────────────────────────────────────────────────
config = ModelConfig(
    pad_idx=tgt_vocab.PAD_IDX,
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
)

# ── Define DataLoader ──────────────────────────────
train_dataloader = DataLoader(dataset = translation_dataset,
                        batch_size= train_cfg.batch_size,
                        collate_fn=custom_collate_fn,
                        shuffle = True,
                        **loader_kwargs)
