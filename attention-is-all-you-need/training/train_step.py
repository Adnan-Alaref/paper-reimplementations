from typing import Callable, Any, Tuple
import torch

def train_epoch(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               criterion : torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scaler: torch.amp.GradScaler,
               scheduler: torch.optim.lr_scheduler.LRScheduler,
               accuracy_fn: Callable[[torch.Tensor, torch.Tensor, int], Tuple[int, int]],
               device: torch.device,
               pad_idx: int,
              )-> dict[str, Any]:
  """
    Train the model for one epoch.

    Args:
        model: Transformer model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer used to update model parameters.
        scaler: Gradient scaler for mixed-precision training.
        device: Device used for training.
        pad_idx: Padding token index.

    Returns:
        Dictionary containing training metrics for the epoch.
  """
  total_train_loss, total_correct = 0.0, 0
  total_tokens = 0

  model.train()
  for source, target in dataloader:

    # ── Move batch to device ──
    source = source.to(device, non_blocking = True)
    target = target.to(device, non_blocking =True)

    # ── Prepare decoder input and expected target (shift right)──
    target_input = target[:, :-1] # [B, T-1]
    target_output = target[:, 1:] # [B, T-1]

    # ── Clear previous gradients ──
    optimizer.zero_grad(set_to_none=True) # faster & safer

    # Forward Pass
    with torch.autocast(device_type=device.type,
                        enabled= device.type == "cuda"):
      # [B, T-1, vocab_szie]
      logits = model(source, target_input)
      '''
         logits = logits.permute(0,2,1)
         logits = logits.reshape(-1, logits.size(-1))
         and target_output = target_output.reshape(-1)
      '''
      logits = logits.transpose(-2,-1)
      loss = criterion(logits, target_output)

     # Backward Pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    # Accumulate loss
    total_train_loss += loss.item()

    # ── Token Accuracy ──
    num_correct, num_tokens = accuracy_fn(
      logits,
      target_output,
      pad_idx
    )
    total_correct += num_correct
    total_tokens += num_tokens


  # ── Get Current Learning Rate ──
  current_lr = optimizer.param_groups[0]["lr"]

  # ── Compute Mean Training Loss Across Batches ────
  avg_loss = total_train_loss / len(dataloader)

  # ── Compute Token Accuracy (%) Across the Epoch ─────────────────
  avg_acc = (total_correct / total_tokens) * 100

  # ── Return Epoch Metrics ──
  return{
    "Model_Name": model.__class__.__name__,
    "train_loss": avg_loss,
    "train_acc": avg_acc,
    "lr": current_lr
  }