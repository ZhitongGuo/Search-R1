#!/usr/bin/env python3
"""Merge verl FSDP checkpoint shards into a single HuggingFace model."""

import argparse
import torch
from pathlib import Path
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_fsdp_checkpoint(checkpoint_dir: str, output_dir: str, base_model: str = None):
    checkpoint_path = Path(checkpoint_dir) / "actor"
    hf_config_path = checkpoint_path / "huggingface"

    shard_files = sorted(checkpoint_path.glob("model_world_size_*_rank_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No model shards found in {checkpoint_path}")

    num_shards = len(shard_files)
    print(f"Found {num_shards} shards, loading...")

    shards = []
    for f in shard_files:
        shards.append(torch.load(f, map_location="cpu", weights_only=False))

    keys = list(shards[0].keys())
    print(f"Merging {len(keys)} parameters...")

    merged_state_dict = OrderedDict()
    for k in keys:
        v0 = shards[0][k]
        if hasattr(v0, "_local_tensor"):
            # DTensor: concatenate local shards along the shard dimension
            placements = v0.placements
            if len(placements) == 1 and hasattr(placements[0], "dim"):
                shard_dim = placements[0].dim
                local_tensors = [s[k]._local_tensor for s in shards]
                merged_state_dict[k] = torch.cat(local_tensors, dim=shard_dim)
            else:
                # Replicated — just take rank 0
                merged_state_dict[k] = v0._local_tensor
        elif isinstance(v0, torch.Tensor):
            merged_state_dict[k] = v0
        else:
            merged_state_dict[k] = v0

    del shards

    model_path = base_model or str(hf_config_path)
    print(f"Loading model architecture from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.load_state_dict(merged_state_dict, strict=True)
    del merged_state_dict

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)

    tokenizer = AutoTokenizer.from_pretrained(hf_config_path, trust_remote_code=True)
    tokenizer.save_pretrained(output)

    print(f"Saved merged model to {output}")
    total_size = sum(f.stat().st_size for f in output.glob("*.safetensors"))
    print(f"Model size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", help="Path to global_step_N directory")
    parser.add_argument("output_dir", help="Path to save merged HF model")
    parser.add_argument("--base-model", default=None, help="Base model for architecture")
    args = parser.parse_args()
    merge_fsdp_checkpoint(args.checkpoint_dir, args.output_dir, args.base_model)
