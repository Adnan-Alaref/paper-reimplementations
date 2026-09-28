from transformer.transformer import Transformer
from src.dataloader import config, train_cfg
import torch.nn as nn
import torch

# ── Define Transformer Learning Rate ────────────────────────────────
def transformer_lr(step: int)->float:
  """
    Compute the learning rate from the original Transformer paper.
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Args:
        step: Current optimization step, starting from 1, 
            * LambdaLR automatically passes the current scheduler step
            * to transformer_lr(step); no manual step argument is needed.
        d_model: Transformer model dimension.
        warmup_steps: Number of warmup steps.

    Returns:
        Learning rate for the current optimization step.
  """
  step = max(step, 1)
  return (
    config.d_model ** -0.5
    * min(
      step ** -0.5,
      step * train_cfg.warmup_steps ** -1.5
    )
  )


# ── Create Training Components ───────────────────────────────────────
def create_training_components():
   
    # ── Define Model ───────────────────────
    translation_model = Transformer(**vars(config))


    # ── Define Loss ─────────────────────────────
    criterion = nn.CrossEntropyLoss(
    ignore_index=config.pad_idx,
    label_smoothing=0.1
    )

    # ── Define Optimizer ──────────────
    optimizer = torch.optim.Adam(
        params=translation_model.parameters(),
        lr= train_cfg.learning_rate,
        betas = train_cfg.betas,
        eps= train_cfg.eps
    )


    # ── Define Scheduler ─────────────────────────
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=transformer_lr
    )

    # ── Return Training Components ──────────────────────────
    return translation_model, criterion, optimizer, scheduler