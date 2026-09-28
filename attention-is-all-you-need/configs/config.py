# Instead of hardcoded values, use config
from dataclasses import dataclass
from src.imports import *

@dataclass
class ModelConfig:
  pad_idx: int
  src_vocab_size: int 
  tgt_vocab_size: int
  max_seq_len: int = 32
  d_model: int = 128
  expansion_dim: int = 512
  num_heads: int = 4
  num_layers: int = 2
  dropout: float = 0.1

@dataclass
class TrainingConfig:
  device: torch.device | None = None
  batch_size: int = 2
  eps: float = 1e-9
  epochs: int = 100
  print_every: int = 10
  betas: tuple =(0.9, 0.98)
  warmup_steps:int = 100 #  B = 15 / 2 = 8 -> increase → peak(W_up/B) → decrease 
  learning_rate: float = 1.0 # actual LR=initial LR×LambdaLR multiplier
  save_every: int = 10  
  src_path = r"/home/adnan/Desktop/Translation-English-Arabic/data/nature_english.txt"
  tgt_path = r"/home/adnan/Desktop/Translation-English-Arabic/data/nature_arabic.txt"

  def __post_init__(self):
    if self.device is None:
      self.device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
      )
      