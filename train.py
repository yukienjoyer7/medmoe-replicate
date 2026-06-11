"""
Training loop for MedMoE.

Total loss: L_LM + 0.1 * L_router + 0.01 * L_aux
Logs per-step and per-epoch: lm_loss, router_loss, aux_loss, router_acc.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from model.medmoe import MedMoE
from data.dataset import MedMoEDataset

# ── hyperparameters ──────────────────────────────────────────────────────────
EPOCHS      = 3
BATCH_SIZE  = 4
LR          = 3e-4
W_ROUTER    = 0.1
W_AUX       = 0.01
MAX_LENGTH  = 64
DATA_JSON   = "data/dataset.json"
SAVE_PATH   = "checkpoints/medmoe.pt"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────


def router_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def train():
    print(f"Device: {DEVICE}")

    # --- data ---
    dataset    = MedMoEDataset(DATA_JSON, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # --- model ---
    model = MedMoE().to(DEVICE)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

    # --- optimiser + scheduler ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    total_steps = EPOCHS * len(dataloader)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # --- training loop ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_lm   = 0.0
        epoch_rtr  = 0.0
        epoch_aux  = 0.0
        epoch_acc  = 0.0

        for step, (images, input_ids, labels, domain_labels) in enumerate(dataloader, 1):
            images       = images.to(DEVICE)
            input_ids    = input_ids.to(DEVICE)
            labels       = labels.to(DEVICE)
            domain_labels = domain_labels.to(DEVICE)

            lm_loss, router_logits, aux_loss = model(images, input_ids, labels)
            router_loss = F.cross_entropy(router_logits, domain_labels)

            loss = lm_loss + W_ROUTER * router_loss + W_AUX * aux_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            acc = router_accuracy(router_logits, domain_labels)

            epoch_lm  += lm_loss.item()
            epoch_rtr += router_loss.item()
            epoch_aux += aux_loss.item()
            epoch_acc += acc

            if step % 10 == 0 or step == len(dataloader):
                print(
                    f"  epoch {epoch} step {step:>3}/{len(dataloader)} | "
                    f"lm {lm_loss.item():.3f}  "
                    f"rtr {router_loss.item():.3f}  "
                    f"aux {aux_loss.item():.3f}  "
                    f"acc {acc:.2f}"
                )

        n = len(dataloader)
        print(
            f"Epoch {epoch} summary | "
            f"lm {epoch_lm/n:.3f}  "
            f"rtr {epoch_rtr/n:.3f}  "
            f"aux {epoch_aux/n:.3f}  "
            f"router_acc {epoch_acc/n:.3f}\n"
        )

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved to {SAVE_PATH}")


if __name__ == "__main__":
    train()
