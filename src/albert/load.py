import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file


def load_checkpoint(repository_id: str) -> dict[str, torch.Tensor]:
    files = list_repo_files(repository_id)
    if "model.safetensors" in files:
        return load_file(hf_hub_download(repository_id, "model.safetensors"))

    path = hf_hub_download(repository_id, "pytorch_model.bin")
    return torch.load(path, map_location="cpu", weights_only=True)
