from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}\n")


def show_tokens(text: str) -> None:
    ids = tokenizer(text)["input_ids"]
    print(f"{text!r}  ->  {len(ids)} tokens")
    for token_id, token in zip(ids, tokenizer.convert_ids_to_tokens(ids)):
        print(f"   {token_id:>6}  {token}")


def text_to_ids() -> None:
    section("1. Text -> tokens -> ids")
    show_tokens("The capital of France is Paris.")


def ids_to_text() -> None:
    section("2. Ids -> text")
    ids = tokenizer("The capital of France is Paris.")["input_ids"]
    print("decode(ids)                          :", repr(tokenizer.decode(ids)))
    print("decode(ids, skip_special_tokens=True):", repr(tokenizer.decode(ids, skip_special_tokens=True)))


def special_tokens() -> None:
    section("3. Special tokens")
    for name in ("cls_token", "sep_token", "pad_token", "mask_token", "unk_token"):
        token = getattr(tokenizer, name)
        print(f"   {token:<7} id {tokenizer.convert_tokens_to_ids(token)}")

    print(f"Vocabulary size: {len(tokenizer)}")


def batch_padding() -> None:
    section("4. Batches: padding and attention_mask")
    batch = tokenizer(["A short text.", "A much longer text with more tokens."], padding=True)
    for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
        print("   ids :", ids)
        print("   mask:", mask)


def truncation() -> None:
    section("5. Truncation")
    ids = tokenizer("one two three four five six seven", truncation=True, max_length=5)["input_ids"]
    print("   max_length=5:", tokenizer.convert_ids_to_tokens(ids))


def pairs() -> None:
    section("6. Pairs of texts")
    ids = tokenizer("what is rope?", "RoPE rotates vectors.")["input_ids"]
    print("  ", tokenizer.convert_ids_to_tokens(ids))


if __name__ == "__main__":
    text_to_ids()
    ids_to_text()
    special_tokens()
    batch_padding()
    truncation()
    pairs()
