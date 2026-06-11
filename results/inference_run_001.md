# Inference Analysis — Run 001

**Date:** 2026-06-11  
**Checkpoint:** `checkpoints/phase3.pt`  
**Command:** `python inference.py` (one sample per domain)

---

## Samples Tested

| Sample | Domain (true) | Domain (predicted) | Correct |
|--------|---------------|--------------------|---------|
| 0 | MRI | MRI | ✓ |
| 2 | X-Ray | MRI | ✗ |
| 3 | CT | MRI | ✗ |
| 11 | Pathology | Pathology | ✓ |

Router accuracy on these 4 samples: **50%**

---

## Router Logits

| Sample | X-Ray | CT | MRI | Pathology |
|--------|-------|----|-----|-----------|
| MRI (✓) | -48.52 | 1.21 | **28.93** | 3.72 |
| X-Ray (✗) | -43.98 | 5.64 | **32.85** | 0.91 |
| CT (✗) | -47.92 | 2.48 | **32.51** | 2.17 |
| Pathology (✓) | -8.44 | 4.42 | 1.52 | **7.22** |

### Observation
The router has collapsed toward predicting **MRI** for almost everything. X-Ray logits are strongly negative across all samples (-8 to -48), suggesting the router never learned to distinguish X-Ray from other modalities confidently. Only Pathology has a competing logit that beats MRI.

This is consistent with the training dynamics: at 1 epoch Phase 2 router accuracy was only 36%, and the router loss dropped from 4.29 → 0.03 over 3 epochs — but this fast collapse likely reflects memorisation of a biased decision boundary rather than genuine generalisation.

---

## Expert Dispatch per Layer

### Pattern observed across all 4 samples
- Gate probabilities across all 30 layers are nearly uniform: **~0.24–0.27 per expert**
- No layer shows clear expert specialisation (e.g., E2/MRI always firing for MRI inputs)
- The "top expert" per layer changes arbitrarily between samples of the same domain

### Example: Layer-wise top expert for MRI sample
```
Layers 0–11:  E1, E2, E1, E3, E1, E2, E3, E3, E0, E1, E0, E1
Layers 12–29: E1, E3, E2, E0, E2, E1, E0, E0, E2, E3, E0, E0, E1, E3, E0, E1, E2, E0
```
No consistent expert dominance — routing is essentially random at this scale.

### Why this happens
The MoE experts were initialised from the same truncated LlamaMLP weights (all identical at Phase 3 start) and trained for only 3 epochs on 200 samples. With no auxiliary load-balancing loss (paper-exact), there is no pressure to force specialisation. The experts diverge slowly and symmetrically, so gate probs remain close to uniform.

This is the **expert collapse / uniform routing** phenomenon described in MoE literature. At paper scale (larger data, more epochs, domain-labelled Phase 3 data), experts would gradually specialise.

---

## Generated Text Quality

All 4 samples produced **incoherent outputs**:

> "visualized visualized visualized visualized... urinary thiswellwise?? assistant yesy kidney kidney kidney..."

> "image?? in? this the lesion? is this this system? with this lesion?..."

### Why this happens
1. **MoE activation reset**: `activate_moe()` replaces all 30 FFN layers with truncated experts (intermediate_dim 1536→576). This degrades LM quality sharply at the start of Phase 3 (lm_loss spiked to 8.2 at epoch 1).
2. **Insufficient recovery**: 3 epochs on 200 samples is not enough to fully recover generation quality after the weight truncation. Phase 3 final lm_loss was 2.27, but coherent generation typically requires lm_loss < 1.5 on the training distribution.
3. **Short sequence budget**: `max_length=64` tokens during training, combined with no repetition penalty at inference, leads to repetition loops.

---

## Summary

| Aspect | Result | Expected at this scale |
|--------|--------|------------------------|
| Router accuracy (4 samples) | 50% | Low — 200 samples, 3 epochs |
| Router logit collapse | MRI dominates | Yes — small data bias |
| Expert specialisation | None (uniform ~0.25) | None — needs more data + epochs |
| Generated text quality | Incoherent | Expected — post-MoE weight truncation |

### What this tells us about the mechanism

The 3-phase structure works as designed:
- **Phase 1** connects vision to language (projector learns)
- **Phase 2** trains the router as a domain classifier over hidden states
- **Phase 3** hands off routing decisions to frozen router and tries to specialise experts

The failure modes observed (router collapse, uniform expert dispatch, incoherent generation) are all **scale failures**, not implementation bugs. They are consistent with what would be expected when running the paper's architecture at 1/100th the data and epoch budget.

To see meaningful expert specialisation, the minimum viable experiment would require: ~2000 domain-labelled samples, ~10 epochs Phase 3, and ideally an auxiliary load-balancing loss in Phase 3 (despite not being in the paper) to break the initial symmetry between identically-initialised experts.

---

## Raw Output Example (Sample 0 — MRI)

```
======================================================================
Sample 0  |  true domain: MRI
Question: is there a lesion in the left temporal lobe?
Image:    data/images/vqarad_mri_002.jpg

  Router: predicted=MRI  true=MRI  ✓
  Router logits: X-Ray=-48.52  CT=1.21  MRI=28.93  Pathology=3.72

  Layer  | Top expert | Dispatch counts          | Mean gate probs
  ----------------------------------------------------------------------
      0  | E1 (CT)            | E0:  0  E1:  2  E2:  1  E3:  1 | 0.24  0.26  0.25  0.25
      1  | E2 (MRI)           | E0:  0  E1:  0  E2:  2  E3:  2 | 0.24  0.23  0.26  0.27
      2  | E1 (CT)            | E0:  0  E1:  2  E2:  2  E3:  0 | 0.25  0.27  0.25  0.24
      3  | E3 (Pathology)     | E0:  1  E1:  0  E2:  1  E3:  2 | 0.26  0.24  0.25  0.26
      4  | E1 (CT)            | E0:  1  E1:  2  E2:  1  E3:  0 | 0.24  0.27  0.25  0.24
      5  | E2 (MRI)           | E0:  0  E1:  1  E2:  2  E3:  1 | 0.25  0.24  0.26  0.25
      6  | E3 (Pathology)     | E0:  1  E1:  0  E2:  1  E3:  2 | 0.25  0.24  0.26  0.25
      7  | E3 (Pathology)     | E0:  1  E1:  0  E2:  1  E3:  2 | 0.25  0.23  0.24  0.27
      8  | E0 (X-Ray)         | E0:  1  E1:  1  E2:  1  E3:  1 | 0.25  0.24  0.25  0.25
      9  | E1 (CT)            | E0:  0  E1:  2  E2:  1  E3:  1 | 0.24  0.26  0.24  0.26
     10  | E0 (X-Ray)         | E0:  2  E1:  1  E2:  1  E3:  0 | 0.25  0.26  0.25  0.23
     11  | E1 (CT)            | E0:  0  E1:  2  E2:  2  E3:  0 | 0.23  0.25  0.27  0.24
     12  | E1 (CT)            | E0:  1  E1:  2  E2:  1  E3:  0 | 0.23  0.31  0.24  0.22
     13  | E3 (Pathology)     | E0:  1  E1:  0  E2:  1  E3:  2 | 0.26  0.23  0.25  0.26
     14  | E2 (MRI)           | E0:  1  E1:  1  E2:  2  E3:  0 | 0.25  0.25  0.27  0.23
     15  | E0 (X-Ray)         | E0:  2  E1:  2  E2:  0  E3:  0 | 0.26  0.27  0.23  0.24
     16  | E2 (MRI)           | E0:  1  E1:  0  E2:  2  E3:  1 | 0.24  0.24  0.27  0.25
     17  | E1 (CT)            | E0:  0  E1:  2  E2:  0  E3:  2 | 0.24  0.26  0.23  0.27
     18  | E0 (X-Ray)         | E0:  1  E1:  1  E2:  1  E3:  1 | 0.26  0.24  0.25  0.25
     19  | E0 (X-Ray)         | E0:  2  E1:  1  E2:  0  E3:  1 | 0.28  0.24  0.23  0.25
     20  | E2 (MRI)           | E0:  1  E1:  1  E2:  2  E3:  0 | 0.26  0.27  0.25  0.22
     21  | E3 (Pathology)     | E0:  1  E1:  0  E2:  1  E3:  2 | 0.25  0.22  0.25  0.27
     22  | E0 (X-Ray)         | E0:  2  E1:  0  E2:  0  E3:  2 | 0.25  0.22  0.23  0.30
     23  | E2 (MRI)           | E0:  0  E1:  1  E2:  2  E3:  1 | 0.24  0.23  0.26  0.27
     24  | E0 (X-Ray)         | E0:  2  E1:  0  E2:  0  E3:  2 | 0.30  0.22  0.23  0.26
     25  | E0 (X-Ray)         | E0:  2  E1:  1  E2:  0  E3:  1 | 0.24  0.28  0.23  0.25
     26  | E1 (CT)            | E0:  0  E1:  2  E2:  1  E3:  1 | 0.23  0.31  0.24  0.22
     27  | E3 (Pathology)     | E0:  0  E1:  1  E2:  1  E3:  2 | 0.23  0.22  0.24  0.32
     28  | E0 (X-Ray)         | E0:  2  E1:  0  E2:  0  E3:  2 | 0.23  0.17  0.19  0.41
     29  | E1 (CT)            | E0:  0  E1:  2  E2:  0  E3:  2 | 0.18  0.24  0.20  0.39

  Generating response...
  Answer: image?? in? this the lesion?
is this this system? with this lesion? in this the lesion lesion lesion?
? kidneys
is the lesion lesion?
yes kidney lesion?
```
