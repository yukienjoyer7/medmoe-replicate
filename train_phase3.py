"""
Phase 3 — Domain-Specific MoE Tuning.

Setup     : load Phase 2 checkpoint, call activate_moe() to replace FFNs
            with MoEFFN (weights copied from trained LlamaMLP, truncated to
            intermediate_dim=576)
Trainable : MoE domain experts + meta-expert only
Frozen    : CLIP, projector, router, LLM attention + embeddings + norms
Loss      : L_MoE (eq 5) — LM cross-entropy only (paper exact)
Goal      : experts specialise per domain while meta-expert captures global info
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from model.medmoe import MedMoE, MoEWrapper
from data.dataset import MedMoEDataset

EPOCHS     = 9
BATCH_SIZE = 2
ACCUM      = 4
LR         = 1e-4
MAX_LEN    = 64
DATA_JSON  = "data/dataset.json"
LOAD_PATH  = "checkpoints/phase2.pt"
SAVE_PATH  = "checkpoints/phase3.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def freeze_for_phase3(model: MedMoE):
    """Freeze everything, then unfreeze MoEWrapper subtrees only."""
    for p in model.parameters():
        p.requires_grad = False
    for module in model.lm.modules():
        if isinstance(module, MoEWrapper):
            for p in module.parameters():
                p.requires_grad = True


def train():
    print("=" * 50)
    print("PHASE 3 — Domain-Specific MoE Tuning")
    print("=" * 50)

    dataset    = MedMoEDataset(DATA_JSON, max_length=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True)

    model = MedMoE().to(DEVICE)
    model.load_state_dict(torch.load(LOAD_PATH, map_location=DEVICE))
    print(f"Loaded Phase 2 checkpoint from {LOAD_PATH}")

    model.activate_moe(num_experts=4, top_k=2)
    freeze_for_phase3(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable/1e6:.2f}M (MoE experts + meta-expert)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0
    )
    total_steps = EPOCHS * (len(dataloader) // ACCUM)
    scheduler   = get_cosine_schedule_with_warmup(
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

        n = len(dataloader)
        print(f"Epoch {epoch} summary | lm {epoch_lm/n:.3f}\n")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved → {SAVE_PATH}")


if __name__ == "__main__":
    train()
