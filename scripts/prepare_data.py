import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

TOKENIZER_ID = "answerdotai/ModernBERT-base"
SOURCE = "datasets/HuggingFaceFW/finewiki/data/enwiki/000_00000.parquet"
DATA_DIR = Path("data")
VALIDATION_ROW_GROUPS = 2

CITATION = re.compile(r"\[\d+\]")
HEADING = re.compile(r"^#+ ")


def clean(text: str) -> str:
    lines = (line for line in text.splitlines() if not line.startswith("|"))
    return CITATION.sub("", "\n".join(HEADING.sub("", line) for line in lines)).strip()


def write_split(name: str, parquet: pq.ParquetFile, row_groups: range, tokenizer):
    path = DATA_DIR / f"{name}.bin"

    articles = 0
    tokens = 0

    with open(path, "wb") as file:
        for row in row_groups:
            texts = parquet.read_row_group(row, columns=["text"]).column("text").to_pylist()
            texts = [text for text in map(clean, texts) if text]

            batch = tokenizer(texts, split_special_tokens=True)["input_ids"]
            ids = np.concatenate([np.array(article, dtype=np.uint16) for article in batch])
            ids.tofile(file)

            articles += len(batch)
            tokens += len(ids)

    print(f"{name}: {articles:,} articles, {tokens:,} tokens -> {path}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    DATA_DIR.mkdir(exist_ok=True)

    parquet = pq.ParquetFile(HfFileSystem().open(SOURCE))
    train_row_groups = parquet.num_row_groups - VALIDATION_ROW_GROUPS

    write_split("train", parquet, range(train_row_groups), tokenizer)
    write_split("validation", parquet, range(train_row_groups, parquet.num_row_groups), tokenizer)


if __name__ == "__main__":
    main()
