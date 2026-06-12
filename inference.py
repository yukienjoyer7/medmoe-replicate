"""
Inference — run a sample through the trained Med-MoE model and inspect routing.

Usage:
    python inference.py                          # runs all 4 domain samples from dataset
    python inference.py --idx 0                  # run a specific sample index
    python inference.py --checkpoint checkpoints/phase3.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
from PIL import Image
from transformers import AutoTokenizer
import open_clip

from model.medmoe import MedMoE, MoEWrapper
from model.moe_ffn import MoEFFN

CHECKPOINT   = "checkpoints/phase3.pt"
DATA_JSON    = "data/dataset.json"
MODEL_ID     = "HuggingFaceTB/SmolLM-135M-Instruct"
DOMAIN_NAMES = ["X-Ray", "CT", "MRI", "Pathology"]
MAX_NEW_TOKENS = 64
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(checkpoint: str) -> MedMoE:
    model = MedMoE().to(DEVICE)
    model.activate_moe(num_experts=4, top_k=2)   # must happen before load_state_dict
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    return model


def get_clip_preprocess():
    _, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    return preprocess


def collect_routing(model: MedMoE) -> list[dict]:
    """Read last_routing from every MoEFFN layer after a forward pass."""
    routing = []
    for i, layer in enumerate(model.lm.model.layers):
        if isinstance(layer.mlp, MoEWrapper):
            moe: MoEFFN = layer.mlp.moe
            if moe.last_routing is not None:
                r = moe.last_routing
                # mean gate probs across all token positions
                mean_probs = r["gate_probs"].mean(dim=0)  # (num_experts,)
                # most dispatched expert (by token count)
                dispatch_counts = torch.zeros(moe.num_experts)
                dispatch_counts.scatter_add_(
                    0, r["indices"].view(-1),
                    torch.ones(r["indices"].numel())
                )
                routing.append({
                    "layer": i,
                    "mean_gate_probs": mean_probs.tolist(),
                    "dispatch_counts": dispatch_counts.tolist(),
                    "top_expert": dispatch_counts.argmax().item(),
                })
    return routing


def print_routing(routing: list[dict], domain_names: list[str]):
    print("\n  Layer  | Top expert | Dispatch counts          | Mean gate probs")
    print("  " + "-" * 70)
    for r in routing:
        counts = "  ".join(f"E{i}:{int(c):>3}" for i, c in enumerate(r["dispatch_counts"]))
        probs  = "  ".join(f"{p:.2f}" for p in r["mean_gate_probs"])
        top    = f"E{r['top_expert']} ({domain_names[r['top_expert']]})"
        print(f"  {r['layer']:>5}  | {top:<18} | {counts} | {probs}")


def generate_response(model: MedMoE, tokenizer, image_tensor: torch.Tensor, question: str) -> str:
    import torch.nn.functional as F
    model_dtype = next(model.lm.parameters()).dtype

    image_emb   = model.clip(image_tensor)
    image_token = model.proj(image_emb).to(model_dtype)  # (1, 1, 576)

    messages = [{"role": "user", "content": f"<image>\n{question}"}]
    prompt   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    text_embeds   = model.lm.model.embed_tokens(input_ids)
    inputs_embeds = torch.cat([image_token, text_embeds], dim=1)

    # set gate_probs on all MoEWrappers so MoEFFN has routing weights during generation
    if model._moe_active:
        router_logits = model.router(inputs_embeds)
        gate_probs = F.softmax(router_logits, dim=-1)
        for layer in model.lm.model.layers:
            if isinstance(layer.mlp, MoEWrapper):
                layer.mlp.current_gate_probs = gate_probs

    with torch.no_grad():
        output_ids = model.lm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if model._moe_active:
        for layer in model.lm.model.layers:
            if isinstance(layer.mlp, MoEWrapper):
                layer.mlp.current_gate_probs = None

    # generated tokens only (strip the prompt length)
    generated = output_ids[0][inputs_embeds.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── main ─────────────────────────────────────────────────────────────────────

def run_sample(model, tokenizer, preprocess, sample: dict, sample_idx: int):
    question      = sample["question"]
    image_path    = sample["image"]
    true_domain   = sample["modality"]

    print(f"\n{'='*70}")
    print(f"Sample {sample_idx}  |  true domain: {true_domain}")
    print(f"Question: {question}")
    print(f"Image:    {image_path}")

    # --- preprocess image ---
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # router prediction — reads T_comb directly (paper eq. 3)
        image_emb   = model.clip(image_tensor)
        image_token = model.proj(image_emb).to(next(model.lm.parameters()).dtype)
        dummy_ids   = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
        text_embeds = model.lm.model.embed_tokens(dummy_ids)
        inputs_embeds = torch.cat([image_token, text_embeds], dim=1)

        router_logits = model.router(inputs_embeds)
        pred_domain   = router_logits.argmax(-1).item()

    print(f"\n  Router: predicted={DOMAIN_NAMES[pred_domain]}  true={true_domain}  "
          f"{'✓' if DOMAIN_NAMES[pred_domain] == true_domain else '✗'}")
    print(f"  Router logits: " + "  ".join(
        f"{DOMAIN_NAMES[i]}={router_logits[0,i].item():.2f}" for i in range(4)
    ))

    # --- generate response (also populates last_routing on each MoEFFN) ---
    print("\n  Generating response...")
    response = generate_response(model, tokenizer, image_tensor, question)
    print(f"  Answer: {response}")

    # --- expert routing per layer (read after generation so last_routing is populated) ---
    routing = collect_routing(model)
    print_routing(routing, DOMAIN_NAMES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--idx", type=int, default=None,
                        help="Run a single sample by index. Omit to run one per domain.")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model     = load_model(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    preprocess = get_clip_preprocess()

    with open(DATA_JSON) as f:
        dataset = json.load(f)

    if args.idx is not None:
        run_sample(model, tokenizer, preprocess, dataset[args.idx], args.idx)
    else:
        # one sample per domain for a representative overview
        seen = set()
        for i, sample in enumerate(dataset):
            d = sample["modality"]
            if d not in seen:
                seen.add(d)
                run_sample(model, tokenizer, preprocess, sample, i)
            if len(seen) == len(DOMAIN_NAMES):
                break

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
