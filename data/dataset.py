"""
MedMoE dataset — loads data/dataset.json and returns
(image_tensor, input_ids, labels, domain_label) per sample.
"""

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import GPT2Tokenizer
import open_clip


class MedMoEDataset(Dataset):
    def __init__(self, json_path: str, max_length: int = 64):
        with open(json_path) as f:
            self.data = json.load(f)

        self.max_length = max_length

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        _, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        item = self.data[idx]

        # --- image ---
        image = Image.open(item["image"]).convert("RGB")
        image_tensor = self.preprocess(image)           # (3, 224, 224)

        # --- text: format as "Q: {question} A: {answer}" ---
        text = f"Q: {item['question']} A: {item['answer']}"
        enc  = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)         # (max_length,)

        # labels: same as input_ids, pad tokens masked with -100
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        domain_label = torch.tensor(item["label"], dtype=torch.long)

        return image_tensor, input_ids, labels, domain_label
