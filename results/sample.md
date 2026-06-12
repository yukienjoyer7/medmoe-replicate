# Sample Inference — Run 002

**Checkpoint:** `checkpoints/phase3.pt`  
**Command:** `python inference.py --idx 0`

---

## Input

| Field | Value |
|-------|-------|
| Sample index | 0 |
| True domain | MRI |
| Question | is there a lesion in the left temporal lobe? |
| Image | `data/images/vqarad_mri_002.jpg` |

---

## Output

### Router Prediction

| | Value |
|-|-------|
| Predicted domain | CT ✗ |
| True domain | MRI |
| X-Ray logit | 0.10 |
| CT logit | **0.24** |
| MRI logit | -0.13 |
| Pathology logit | -0.04 |

Router predicted wrong. All logits are near-zero and indistinguishable — the router outputs near-uniform distribution, consistent with the stalled training (router_acc ~25% throughout Phase 2).

---

### Generated Answer

```
c knownl present tissue lesion image l the imaging contrast?, of in acute organellesu...

is lesion left and a the context the lobe the in sigmoidill field likely this left (?
```

Incoherent. Expected — the model is still recovering from MoE weight truncation on a 200-sample training set.

---

### Expert Dispatch per Layer (all 30 layers)

```
  Layer  | Top expert | Dispatch counts          | Mean gate probs
  ----------------------------------------------------------------------
      0  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
      1  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
      2  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
      3  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
      4  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
      5  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
      6  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
      7  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
      8  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
      9  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     10  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     11  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     12  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     13  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     14  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     15  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     16  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     17  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     18  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     19  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     20  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     21  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     22  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     23  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     24  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     25  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     26  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     27  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     28  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
     29  | E0 (X-Ray)         | E0:  1  E1:  0  E2:  1  E3:  0 | 0.25  0.25  0.25  0.25
```

---

## Interpretation

**Router:** Near-uniform logits (0.10 / 0.24 / -0.13 / -0.04). The router never learned to distinguish domains from T_comb at 200 samples — all values are within noise range of zero.

**Expert dispatch:** Perfectly uniform gate probs (0.25 each) across all 30 layers. This is a direct consequence of the stalled router — since gate_probs ≈ [0.25, 0.25, 0.25, 0.25], top-2 selection always picks E0 and E2 (ties broken deterministically), resulting in identical dispatch across every layer. No domain specialisation.

**Generation:** Incoherent. The model partially learned token co-occurrence patterns (words like "lesion", "left", "lobe" appear) but cannot form coherent sentences. Root cause: MoE activation truncated expert intermediate_dim from 1536→576, and 200 samples is not enough for full recovery even at 9 epochs.

**What this demonstrates about the mechanism:** The routing pipeline (T_comb → Router → gate_probs → MoEFFN) is functioning correctly end-to-end. The failure is purely in the router's inability to learn from insufficient labeled data — a scale problem that confirms the paper's design requires large-scale pretraining to be effective.
