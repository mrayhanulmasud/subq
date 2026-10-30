"""Read single weight tensors from a merged model folder without loading the model.

Merged models are saved by merge.py with save_pretrained: either one `model.safetensors` or several shards
plus `model.safetensors.index.json`. `load_weight(path, "model.layers.3.mlp.up_proj")` returns that
layer's weight as a CPU tensor in its stored dtype (bf16).
"""
import json
import os
from functools import lru_cache

import torch
from safetensors import safe_open


@lru_cache(maxsize=None)
def _weight_map(path):
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return {k: os.path.join(path, v) for k, v in json.load(f)["weight_map"].items()}
    single = os.path.join(path, "model.safetensors")
    assert os.path.exists(single), f"no safetensors weights in {path}"
    with safe_open(single, framework="pt", device="cpu") as f:
        return {k: single for k in f.keys()}


def load_weight(path, module_name):
    """module_name like 'model.layers.0.self_attn.q_proj' or 'lm_head'."""
    key = module_name + ".weight"
    wm = _weight_map(path)
    assert key in wm, f"{key} not in {path}"
    with safe_open(wm[key], framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def linear_names(path):
    """All block linears (196 for Qwen2.5-7B) plus lm_head, in sorted order."""
    names = sorted(k[:-len(".weight")] for k in _weight_map(path)
                   if k.startswith("model.layers.") and k.endswith("_proj.weight"))
    assert len(names) == 196, f"expected 196 block linears, found {len(names)}"
    return names + ["lm_head"]
