# Med-MoE Minimal Reimplementation

A from-scratch reimplementation of **Med-MoE** at small scale — small model, small data, small epochs — for the purpose of understanding how Mixture-of-Experts routing works in medical VLMs.

> This is not meant to reproduce paper results. The goal is to build every component by hand and observe routing behavior end-to-end.

**Reference:** [Med-MoE: Mixture of Domain-Specific Experts for Lightweight Medical Vision-Language Models](https://arxiv.org/abs/2404.10237) — Jiang et al., 2024

---

## What is Med-MoE?

Med-MoE is a lightweight medical VLM that replaces each transformer block's standard FFN with a **Mixture-of-Experts (MoE) FFN** — a set of parallel expert networks where a gating mechanism routes each token to the top-k most relevant experts. A separate **domain router** (a linear probe on the model's hidden states) classifies the medical imaging domain at inference time (X-Ray, CT, MRI, Pathology).

The key insight: instead of one large FFN per layer, you have N smaller specialized FFNs. Each expert can specialize for a different domain without explicit supervision — only the gating and the load-balancing loss guide the specialization.

---

## Architecture

```
Image → CLIP ViT-B/32 (frozen) → CLS token (512-dim)
                                        ↓
                              Projector MLP (512 → 768)
                                        ↓
                        prepend as image token to text sequence
                                        ↓
                    GPT-2 (124M) with MoE FFN in every block
                    ├── each block: gate → top-2 of 4 experts → weighted sum
                    └── output_hidden_states=True
                                        ↓
                    mean all hidden states (all layers, all positions)
                                        ↓
                    Router: nn.Linear(768, 4) → domain label
```

---

## Components

### 1. Vision Encoder — CLIP ViT-B/32 (frozen)

Extracts a single CLS token (1×512) as a global image summary. Frozen entirely — we borrow its pretrained representations without fine-tuning.

### 2. Projector — 2-layer MLP

Maps the 512-dim CLIP CLS vector to 768-dim (GPT-2 hidden dim) so the image can be treated as a single token in the LLM's input sequence.

```
Linear(512, 768) → GELU → Linear(768, 768)
```

### 3. LLM Backbone — GPT-2 (124M)

Standard GPT-2 with every FFN block replaced by a MoE FFN block. Chosen for its small size (fits on 6GB VRAM) and simple architecture.

### 4. MoE FFN Layer

Replaces the standard `Linear → GELU → Linear` FFN with N parallel expert FFNs and a gating network:

- **N = 4 experts** (one per medical domain: X-Ray, CT, MRI, Pathology)
- **top-k = 2** (each token is routed to 2 experts)
- **Output** = weighted sum of the top-2 expert outputs, weights from softmax gating

### 5. Router — Linear Probe

A single linear layer trained to classify the medical imaging domain from the model's hidden states:

```
nn.Linear(768, 4)   # 768 = GPT-2 hidden dim, 4 = number of domains
```

Input: mean of all hidden states across all layers and all token positions. Trained with cross-entropy on domain labels, independently from the LM objective.

---

## Dataset

200 samples total, 50 per modality, loaded from HuggingFace — no manual download required.

| Modality | Source | HuggingFace ID | Filter |
|----------|--------|----------------|--------|
| X-Ray | VQA-RAD | `flaviagiammarino/vqa-rad` | `image_organ == "CHEST"` |
| CT | VQA-RAD | `flaviagiammarino/vqa-rad` | `image_organ == "ABDOMEN"` |
| MRI | VQA-RAD | `flaviagiammarino/vqa-rad` | `image_organ == "HEAD"` |
| Pathology | PathVQA | `flaviagiammarino/path-vqa` | sample 50 from train split |

---

## Loss Functions

Three losses are combined during training:

```
L_total = L_LM + λ_router * L_router + λ_aux * L_aux
```

| Term | Weight | Description |
|------|--------|-------------|
| `L_LM` | 1.0 | Autoregressive cross-entropy on answer tokens |
| `L_router` | 0.1 | Cross-entropy on domain label from linear router |
| `L_aux` | 0.01 | Switch Transformer load-balancing loss |

### L_LM — Language Modeling Loss

```
L_LM = -1/T * Σ log P(y_t | y_<t, x_image, x_question)
```

Computed only over answer tokens (not question or image tokens).

### L_router — Router Classification Loss

```
L_router = CrossEntropy(W_router * mean(all_hidden_states), y_domain)
```

### L_aux — Load-Balancing Loss (Switch Transformer)

```
L_aux = N * Σ_i (f_i * p_i)
```

- `f_i` = fraction of tokens hard-dispatched to expert `i` (no gradient)
- `p_i` = mean soft routing probability for expert `i` (has gradient)
- `N` = number of experts

Prevents all tokens from collapsing to one or two experts.

---

## Implementation Plan

### Phase 0 — Setup
- [ ] Create conda env, install: `torch`, `transformers`, `open_clip_torch`, `datasets`, `pillow`
- [ ] Prepare dataset: sample 50 per modality from VQA-RAD + PathVQA

### Phase 1 — Build the pieces
- [ ] `model/clip_encoder.py` — wrap CLIP ViT-B/32, expose `encode_image()` returning CLS token
- [ ] `model/projector.py` — 2-layer MLP, 512 → 768
- [ ] `model/moe_ffn.py` — MoE FFN: 4 experts, top-2 gating, auxiliary loss computation
- [ ] `model/router.py` — linear probe on mean hidden state

### Phase 2 — Wire up the model
- [ ] `model/medmoe.py` — patch GPT-2 FFN blocks with MoE FFN, attach router head
- [ ] Verify forward pass shapes end-to-end with a single dummy sample

### Phase 3 — Training
- [ ] `data/dataset.py` — returns `(image, question, answer, domain_label)` per sample
- [ ] `train.py` — training loop with all three losses, 3 epochs

### Phase 4 — Inspect routing
- [ ] `inference.py` — run on a sample image, print expert activation per layer
- [ ] `notebooks/routing_analysis.ipynb` — visualize router confidence, expert load, gating weights per modality

---

## What Success Looks Like

- All three losses decrease over 3 epochs
- Router accuracy > 25% (above random for 4 classes)
- No single expert handles > 80% of tokens (load-balancing is working)
- You can point at a line of code and explain exactly where routing happens

---

## Known Gotcha: Expert Collapse

Without `L_aux`, the gating network quickly learns to always route to the same 1–2 experts. Other experts receive no gradients and never specialize. This is the most important failure mode to understand in MoE training — intentionally run without `L_aux` first, observe the collapse, then turn it on.

---

## Repo Structure

```
med-moe-reimplement/
├── model/
│   ├── clip_encoder.py
│   ├── projector.py
│   ├── moe_ffn.py
│   ├── router.py
│   └── medmoe.py
├── data/
│   └── dataset.py
├── notebooks/
│   └── routing_analysis.ipynb
├── train.py
├── inference.py
└── README.md
```

---

## References

- Jiang et al. (2024). *Med-MoE: Mixture of Domain-Specific Experts for Lightweight Medical Vision-Language Models.* arXiv:2404.10237
- Fedus et al. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR.
- MoE-LLaVA: [PKU-YuanGroup/MoE-LLaVA](https://github.com/PKU-YuanGroup/MoE-LLaVA)
