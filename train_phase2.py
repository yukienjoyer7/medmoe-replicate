"""
Phase 2 — Instruction Tuning and Routing.

Trainable : full LLM + router + projector
Frozen    : CLIP only
Loss      : L_instr (LM) + 0.1 * L_router (domain classification)
Goal      : instruction-following ability + router learns domain signal
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from model.medmoe import MedMoE
from data.dataset import MedMoEDataset

EPOCHS     = 3
BATCH_SIZE = 2
ACCUM      = 4
LR         = 1e-4
MAX_LEN    = 64
DATA_JSON  = "data/dataset.json"
LOAD_PATH  = "checkpoints/phase1.pt"
SAVE_PATH  = "checkpoints/phase2.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def freeze_for_phase2(model: MedMoE):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.lm.parameters():
        p.requires_grad = True
    for p in model.router.parameters():
        p.requires_grad = True
    for p in model.proj.parameters():   # vision encoder frozen, projector trainable
        p.requires_grad = True


def router_accuracy(logits, labels):
    return (logits.argmax(-1) == labels).float().mean().item()


def train():
    print("=" * 50)
    print("PHASE 2 — Instruction Tuning and Routing")
    print("=" * 50)

    dataset    = MedMoEDataset(DATA_JSON, max_length=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True)

    model = MedMoE().to(DEVICE)
    model.load_state_dict(torch.load(LOAD_PATH, map_location=DEVICE))
    print(f"Loaded Phase 1 checkpoint from {LOAD_PATH}")

    freeze_for_phase2(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable/1e6:.2f}M (LLM + router)")

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
        epoch_lm, epoch_rtr, epoch_acc = 0.0, 0.0, 0.0
        optimizer.zero_grad()

        for step, (images, input_ids, labels, domain_labels) in enumerate(dataloader, 1):
            images        = images.to(DEVICE)
            input_ids     = input_ids.to(DEVICE)
            labels        = labels.to(DEVICE)
            domain_labels = domain_labels.to(DEVICE)

            with autocast("cuda"):
                lm_loss, router_logits, _ = model(images, input_ids, labels)
                router_loss = F.cross_entropy(router_logits, domain_labels)
                loss = (lm_loss + router_loss) / ACCUM

            scaler.scale(loss).backward()

            acc = router_accuracy(router_logits.detach(), domain_labels)
            epoch_lm  += lm_loss.item()
            epoch_rtr += router_loss.item()
            epoch_acc += acc

            if step % ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if step % 10 == 0 or step == len(dataloader):
                print(f"  epoch {epoch} step {step:>3}/{len(dataloader)} | "
                      f"lm {lm_loss.item():.3f}  "
                      f"rtr {router_loss.item():.3f}  "
                      f"acc {acc:.2f}")

        n = len(dataloader)
        print(f"Epoch {epoch} summary | "
              f"lm {epoch_lm/n:.3f}  "
              f"rtr {epoch_rtr/n:.3f}  "
              f"router_acc {epoch_acc/n:.3f}\n")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved → {SAVE_PATH}")


if __name__ == "__main__":
    train()
