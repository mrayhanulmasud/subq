"""Merge a LoRA adapter into the bf16 base and save a plain HF checkpoint on local disk.

python merge.py --adapter <adapter dir>|none --out <dir> [--base unsloth/Qwen2.5-7B-Instruct]
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the untouched base")
ap.add_argument("--out", required=True)
ap.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
args = ap.parse_args()

done_flag = os.path.join(args.out, "MERGE_DONE")  # written last, so a half-saved folder is never skipped
if os.path.exists(done_flag):
    print(f"[merge] {args.out} already complete, skipping")
    raise SystemExit(0)

print(f"[merge] loading base {args.base}")
model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
tok = AutoTokenizer.from_pretrained(args.base)

if args.adapter != "none":
    from peft import PeftModel

    ref = model.model.layers[14].self_attn.q_proj.weight.detach().clone()
    print(f"[merge] applying adapter {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    d = (model.model.layers[14].self_attn.q_proj.weight - ref).abs().max().item()
    print(f"[merge] max |dW| layer14 q_proj = {d:.3e}")
    assert d > 0, "adapter had no effect on the weights; check adapter_config target_modules"
    del ref

tied = getattr(model.config, "tie_word_embeddings", None)
assert tied is False, f"expected untied lm_head for 7B, got tie_word_embeddings={tied}; stop and report"
assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr(), "lm_head shares storage with embed_tokens"

n_params = sum(p.numel() for p in model.parameters())
print(f"[merge] params={n_params/1e9:.3f}B  dtype={next(model.parameters()).dtype}")
model.save_pretrained(args.out, safe_serialization=True)
tok.save_pretrained(args.out)
open(done_flag, "w").close()
print(f"[merge] saved -> {args.out}")
