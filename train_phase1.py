"""
Phase 1 — Multimodal Medical Alignment.

Trainable : projector only
Frozen    : CLIP, LLM, router
Loss      : L_align (LM loss on answer tokens)
Goal      : teach the projector to map image tokens into the LLM's token space
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from model.medmoe import MedMoE
from data.dataset import MedMoEDataset

EPOCHS     = 3
BATCH_SIZE = 2
ACCUM      = 4
LR         = 1e-3
MAX_LEN    = 64
DATA_JSON  = "data/dataset.json"
SAVE_PATH  = "checkpoints/phase1.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def freeze_for_phase1(model: MedMoE):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.proj.parameters():
        p.requires_grad = True


def train():
    print("=" * 50)
    print("PHASE 1 — Multimodal Medical Alignment")
    print("=" * 50)

    dataset    = MedMoEDataset(DATA_JSON, max_length=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True)

    model = MedMoE().to(DEVICE)
    freeze_for_phase1(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable/1e6:.2f}M (projector only)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    total_steps = EPOCHS * (len(dataloader) // ACCUM)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )
    scaler = GradScaler("cuda")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_lm = 0.0
        optimizer.zero_grad()

        for step, (images, input_ids, labels, _) in enumerate(dataloader, 1):
            images, input_ids, labels = (
                images.to(DEVICE), input_ids.to(DEVICE), labels.to(DEVICE)
            )

            with autocast("cuda"):
                lm_loss, _, _ = model(images, input_ids, labels)
                loss = lm_loss / ACCUM

            scaler.scale(loss).backward()
            epoch_lm += lm_loss.item()

            if step % ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if step % 10 == 0 or step == len(dataloader):
                print(f"  epoch {epoch} step {step:>3}/{len(dataloader)} | "
                      f"lm {lm_loss.item():.3f}")

        print(f"Epoch {epoch} summary | lm {epoch_lm/len(dataloader):.3f}\n")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved → {SAVE_PATH}")


if __name__ == "__main__":
    train()
