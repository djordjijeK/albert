import torch
import torch.nn.functional as F

from albert.model import ModernBertForMaskedLM

IGNORE_INDEX = -100


def mask_tokens(input_ids: torch.Tensor, tokenizer, probability: float = 0.3, generator: torch.Generator | None = None):

    def coin(p: float) -> torch.Tensor:
        return torch.rand(input_ids.shape, generator=generator) < p

    is_special_token_mask = torch.isin(input_ids, torch.tensor(tokenizer.all_special_ids))
    selected_tokens = coin(probability) & ~is_special_token_mask

    labels = input_ids.masked_fill(~selected_tokens, IGNORE_INDEX)

    to_mask = selected_tokens & coin(0.8)
    to_random = selected_tokens & ~to_mask & coin(0.5)

    random_ids = torch.randint(tokenizer.vocab_size, input_ids.shape, generator=generator)

    inputs = input_ids.masked_fill(to_mask, tokenizer.mask_token_id)
    inputs = torch.where(to_random, random_ids, inputs)

    return inputs, labels


def mlm_loss(model: ModernBertForMaskedLM, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    hidden = model.model(inputs)

    selected_mask = labels != IGNORE_INDEX
    logits = model.decoder(model.head(hidden[selected_mask]))

    return F.cross_entropy(logits, labels[selected_mask])
