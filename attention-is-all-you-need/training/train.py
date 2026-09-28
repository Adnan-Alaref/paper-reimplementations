"""
Run the complete training pipeline, including training, checkpointing, scheduler updates, and metric logging.
"""

from torch.amp import  GradScaler # PERFORMANCE OPTIMIZATIONS
from src.dataloader import config, train_cfg,train_dataloader
from training.setup import create_training_components
from training.train_step import train_epoch
from pathlib import Path

from src.utils import (
  accuracy_fn,
  save_checkpoints,
  save_finishing_training,
  save_training_log,
  set_all_seeds,
  model_summary
)

import torch


# ── Reproducibility ───────────────
set_all_seeds(42)

scaler = GradScaler()  # helps scale gradients safely

model, criterion, optimizer, scheduler = create_training_components()

# ── Save Model Summary ────────────────────────────────────────────────
model_summary(model)

# ── Initialize Training History, Best Metrics, and Device ────────────
best_epoch: int = 0
best_loss: float = float("inf")

train_acc, epochs = [], []
train_losses, lr_history = [], []


model.to(train_cfg.device)
for epoch in range(1, train_cfg.epochs + 1):
  # Train → Logging → Save if best → Print

  # ── Train ───────────────────────────────────────────────────────────────────
  train_history = train_epoch(model= model,
                             dataloader= train_dataloader,
                             accuracy_fn= accuracy_fn,
                             criterion= criterion,
                             optimizer= optimizer,
                             scaler= scaler,
                             scheduler = scheduler,
                             device= train_cfg.device,
                             pad_idx = config.pad_idx
                             )

  # ── Logging ─────────────────────────────────────────────────────────────────
  train_losses.append(train_history['train_loss'])
  train_acc.append(train_history['train_acc'])
  lr_history.append(train_history['lr'])
  epochs.append(epoch)

  # ── Checkpoint ─────────────────────────────────────────
  if train_history["train_loss"] < best_loss:
    best_epoch = epoch
    best_loss = train_history["train_loss"]

    save_checkpoints(
        model=model,
        scaler=scaler,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loss=train_history["train_loss"],
        train_acc=train_history["train_acc"],
        epoch=epoch,
        path="best_checkpoint.pt"
    )

  # ── Periodic Training Log ──────────────────────
  if(epoch) % train_cfg.print_every == 0:
    save_training_log(
      epoch,
      train_history
    )

# ── Save Training Results ──────────────────────────────────────────────────────
training_results = {
    "epochs": epochs,
    "train_acc": train_acc,
    "lr_history": lr_history,
    "train_losses": train_losses,
}

# ── Checkpoints Directory ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

torch.save(
    training_results,
    CHECKPOINTS_DIR / "training_results.pt"
)


#  ── Print the best checkpoint once after training ──
save_finishing_training(best_epoch, best_loss)