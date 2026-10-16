import sys

import torch
from transformers import AutoTokenizer

from albert.model import ModernBertForMaskedLM

REPOSITORY_ID = sys.argv[1] if len(sys.argv) > 1 else "answerdotai/ModernBERT-base"
TEXTS = ["The capital of France is [MASK].", "[MASK] is considered the 3rd [MASK] from the sun."]

tokenizer = AutoTokenizer.from_pretrained(REPOSITORY_ID)
model = ModernBertForMaskedLM.from_pretrained(REPOSITORY_ID)

batch = tokenizer(TEXTS, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(batch["input_ids"], batch["attention_mask"])

# one (row, position) pair for each [MASK] in the batch
mask_positions = (batch["input_ids"] == tokenizer.mask_token_id).nonzero().tolist()

print(f"model: {REPOSITORY_ID}")
for row, position in mask_positions:
    probabilities = logits[row, position].softmax(dim=-1)
    print(f"\n{TEXTS[row]}  (position {position})")
    for probability, token_id in zip(*probabilities.topk(3)):
        print(f"   {tokenizer.decode(token_id):<12} {probability:.3f}")
