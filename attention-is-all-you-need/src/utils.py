from src.dataloader import config
from src.imports import *
from pathlib import Path

# ── Project Directories ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TEXT_DIR = PROJECT_ROOT / "reports" / "text"

# ── Create Output Directories ─────────────────────────
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)


# ── Save ckpt ────────────────────────────────────────────────────────
def save_checkpoints(model: nn.Module, scaler: torch.amp.GradScaler|None,
                     scheduler: torch.optim.lr_scheduler.LRScheduler,
                     optimizer: torch.optim.Optimizer, train_loss: float, train_acc: float, epoch: int, path:str)->None:
  
  # Define Chekpoint
  checkpoint = {
    "epoch": epoch,
    "train_loss": float(train_loss),
    "train_acc": float(train_acc),
    "model_state_dict": model.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
  }

  # add scaler ONLY if it exists
  if scaler is not None:
    checkpoint['scaler_state_dict'] = scaler.state_dict()

  # Create ckpt path
  checkpoint_path = CHECKPOINTS_DIR / path
  torch.save(checkpoint, checkpoint_path)
  # print(f"Saved checkpoint: {checkpoint_path}")


# ── Load ckpt ────────────────────────────────────────────────────────
def load_checkpoint(path:str, device: str|None=None)->dict:
  if not os.path.isfile(path):
    raise FileNotFoundError(f"Checkpoint file does not found.\nPath: {path}")

  # Load Data
  checkpoint_data = torch.load(path, map_location=device)
  return checkpoint_data

# ── Summary ────────────────────────────────────────────────────────
def model_summary(model):

  src_dummy_data = torch.randint(
    0, 
    config.src_vocab_size, 
    (32, 20), 
    dtype=torch.long
  )

  tgt_dummy_data = torch.randint(
    0, 
    config.tgt_vocab_size, 
    (32, 20), 
    dtype=torch.long
  )

  summary = torchinfo.summary(
      model= model,
      input_data=(src_dummy_data, tgt_dummy_data[:, :-1]),
      device = "cpu",
      depth=4
  )

  # ── Save model summary ──
  summary_path = TEXT_DIR / "model_summary.txt"

  summary_path.write_text(
    str(summary),
    encoding="utf-8",
  )

  print(f"Model summary saved to: {summary_path}")


# ── Set Seed ────────────────────────────────────────────────────────
def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

# ── Accuracy ────────────────────────────────────────────────────────
def accuracy_fn(logits: torch.Tensor, target_output: torch.Tensor, pad_idx: int)->Tuple[int, int]:
  """
    Calculate the number of correct predictions and valid tokens.

    Args:
        logits: Raw model outputs, shape [B, V, T].
        target_output: Ground-truth token IDs, shape [B, T].
        pad_idx: Index of the padding token.

    Returns:
        A tuple containing:
            - num_correct: Number of correctly predicted non-PAD tokens.
            - num_tokens: Number of non-PAD target tokens.
  """

  predictions = logits.argmax(dim=1)        # [B, T-1]
  non_pad_mask = target_output != pad_idx   # [B, T-1]

  num_correct = (
    (predictions == target_output) & non_pad_mask
  ).sum().item()

  num_tokens = non_pad_mask.sum().item()

  return num_correct, num_tokens


# ── PRINT & SAVE TRAINING LOG ────────────────────────────────────────────────
def save_training_log(
  epoch:int,
  train_history: Dict
):

  log_message = (
    f"Epoch [{epoch:>3}] - "
    f"Train loss: {train_history['train_loss']:.3f} | "
    f"Train acc: {train_history['train_acc']:.3f}% | "
    f"LR: {train_history['lr']:.6f}"
  )

  print(log_message)

  with open(TEXT_DIR / "training_log.txt", "a", encoding="utf-8") as file:
        file.write(log_message + "\n")


# ── SAVE FINISHING TRAINING SUMMARY ──────────────────────────────────────────
def save_finishing_training(
  best_epoch: int,
  best_loss: float
):


  log_message = (
    f"Best checkpoint saved — "
    f"Epoch: {best_epoch} | "
    f"Loss: {best_loss:.4f}"
  )

  print(log_message)

  with open(TEXT_DIR / "training_log.txt", "a", encoding="utf-8") as file:
          file.write(log_message + "\n")



# ── Plot Training Loss and Learning Rate ─────────────────────────────────────
def plot_training_history(
    epochs: list[int],
    train_losses: list[float],
    train_acc: list[float],
    display: bool = False
) -> None:
  """
  Plot training loss and learning rate across epochs.

  Args:
      epochs: Epoch numbers.
      train_losses: Training loss for each epoch.
      lr_history: Learning rate for each epoch.
  """

  fig = plt.figure(figsize=(15,5), dpi=80)

  # Train Loss
  plt.subplot(1,2,1)
  plt.plot(epochs, train_losses, color='royalblue', linewidth=2, label="Train Loss")
  plt.ylabel("Loss", color = 'royalblue', fontsize = 12)
  plt.grid(True, linestyle='--', alpha=0.6)
  plt.xlabel("Epochs", fontsize=12)
  plt.legend(loc='best')

  # Learning Rate
  plt.subplot(1,2,2)
  plt.plot(epochs, train_acc, color='g', linewidth=2, label="Train Accuracy")
  plt.ylabel("Accuracy", color='orange', fontsize=12)
  plt.xlabel("Epochs", fontsize=12)
  plt.legend(loc='best')

  fig.suptitle("Training Loss and Accuracy", fontsize=15, fontweight='bold')
  fig.tight_layout()

  # ── Save Figure ────────────────────────────────
  plot_path = FIGURES_DIR / "training_history.png"
  
  fig.savefig(
    plot_path,
    dpi=150,
    bbox_inches="tight",
  )

  if display:
    plt.show()

  plt.close(fig)
  print(f"Training curves saved to: {plot_path.relative_to(PROJECT_ROOT)}")



# ── Learning Rate Curve ────────────────────────────────────────────────────────
def display_lr_curve(
    epoch_values: List[int],
    lr_values: List[float],
    display: bool = False
) -> None:
    """
    Plot and save the learning rate over training epochs.

    Args:
        epoch_values (List[int]): Epoch numbers.
        lr_values (List[float]): Learning rate for each epoch.

    Returns:
        None: Saves the learning rate curve as a PNG file.
    """

    # ── Create Figure ─────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5), dpi=100)

    # ── Learning Rate Curve ────────────────────────────────────────────────────
    plt.plot(
        epoch_values,
        lr_values,
        color="orange",
        linewidth=2,
        label="Learning Rate",
    )

    plt.title("Learning Rate over Epochs", fontsize=12, weight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Learning Rate", color='orange',fontsize=12)
    plt.legend(prop={"size": 12}, loc="best")
    plt.grid(True)

    plt.tight_layout()

    # ── Save Figure ────────────────────────────────────────────────────────────
    save_path = FIGURES_DIR / "learning_rate_curve.png"

    plt.savefig(
        save_path,
        dpi=100,
        bbox_inches="tight",
    )

    if display:
      plt.show()

    plt.close()
    print(f"Learning rate curve saved to: {save_path.relative_to(PROJECT_ROOT)}")