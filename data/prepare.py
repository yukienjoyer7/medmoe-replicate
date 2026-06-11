"""
Download and sample 200 examples (50 per modality) from VQA-RAD and PathVQA.
Saves to data/dataset.json as a list of:
  { "image": <PIL-compatible path or url>, "question": str, "answer": str, "label": int, "modality": str }

Label mapping:
  0 = X-Ray   (VQA-RAD, chest anatomy keywords)
  1 = CT      (VQA-RAD, abdomen anatomy keywords)
  2 = MRI     (VQA-RAD, brain/head anatomy keywords)
  3 = Pathology (PathVQA)

Note: HuggingFace VQA-RAD strips image_organ metadata, so modality is inferred
from anatomy keywords in the question text — matches the original organ mapping.
"""

import json
import random
import os
from datasets import load_dataset
from PIL import Image

SAMPLES_PER_CLASS = 50
SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
OUT_JSON = os.path.join(os.path.dirname(__file__), "dataset.json")

LABEL_MAP = {"X-Ray": 0, "CT": 1, "MRI": 2, "Pathology": 3}

XRAY_KEYWORDS  = {"chest", "lung", "pleural", "pneumothorax", "pneumonia",
                  "cardiac", "heart", "rib", "trachea", "x-ray", "x ray", "xray"}
CT_KEYWORDS    = {"abdomen", "abdominal", "liver", "spleen", "kidney", "pancreas",
                  "gallbladder", "aorta", "bowel", "colon", "ct"}
MRI_KEYWORDS   = {"brain", "cerebr", "infarct", "edema", "lesion", "tumor",
                  "ventricle", "cortex", "mri", "signal intensity", "hyperintens",
                  "hypointens", "white matter", "gray matter"}

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def infer_modality(question: str) -> str | None:
    q = question.lower()
    if any(k in q for k in MRI_KEYWORDS):
        return "MRI"
    if any(k in q for k in XRAY_KEYWORDS):
        return "X-Ray"
    if any(k in q for k in CT_KEYWORDS):
        return "CT"
    return None


def save_image(img: Image.Image, name: str) -> str:
    path = os.path.join(OUT_DIR, name)
    img.convert("RGB").save(path)
    return path


def sample_vqa_rad():
    print("Loading VQA-RAD...")
    ds = load_dataset("flaviagiammarino/vqa-rad", split="train")

    buckets = {"X-Ray": [], "CT": [], "MRI": []}

    for item in ds:
        modality = infer_modality(item["question"])
        if modality and len(buckets[modality]) < SAMPLES_PER_CLASS * 3:
            buckets[modality].append(item)

    records = []
    for modality, items in buckets.items():
        sampled = random.sample(items, min(SAMPLES_PER_CLASS, len(items)))
        for i, item in enumerate(sampled):
            img_name = f"vqarad_{modality.lower().replace('-', '')}_{i:03d}.jpg"
            img_path = save_image(item["image"], img_name)
            records.append({
                "image": img_path,
                "question": item["question"],
                "answer": str(item["answer"]),
                "modality": modality,
                "label": LABEL_MAP[modality],
            })
        print(f"  {modality}: {len(sampled)} samples")

    return records


def sample_path_vqa():
    print("Loading PathVQA...")
    ds = load_dataset("flaviagiammarino/path-vqa", split="train")

    # PathVQA answers are yes/no or short phrases — filter for non-trivial answers
    candidates = [
        item for item in ds
        if item.get("answer", "").lower() not in ("yes", "no", "")
    ]

    if len(candidates) < SAMPLES_PER_CLASS:
        # fall back to all items if filtered set is too small
        candidates = list(ds)

    sampled = random.sample(candidates, SAMPLES_PER_CLASS)
    records = []
    for i, item in enumerate(sampled):
        img_name = f"pathvqa_{i:03d}.jpg"
        img_path = save_image(item["image"], img_name)
        records.append({
            "image": img_path,
            "question": item["question"],
            "answer": str(item["answer"]),
            "modality": "Pathology",
            "label": LABEL_MAP["Pathology"],
        })

    print(f"  Pathology: {len(sampled)} samples")
    return records


if __name__ == "__main__":
    records = sample_vqa_rad() + sample_path_vqa()
    random.shuffle(records)

    with open(OUT_JSON, "w") as f:
        json.dump(records, f, indent=2)

    counts = {}
    for r in records:
        counts[r["modality"]] = counts.get(r["modality"], 0) + 1

    print(f"\nSaved {len(records)} samples to {OUT_JSON}")
    for modality, count in sorted(counts.items()):
        print(f"  {modality}: {count}")
