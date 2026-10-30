"""NF4 per-layer rounding error, in the same format as the HQQ qerr files.

For every model: load the bitsandbytes NF4 model (exactly as nf4_eval.py does), dequantize each of the 196
Linear4bit weights, read the matching bf16 weight from the merged safetensors on disk, and record
||W_nf4 - W_bf16||_F per layer to results/qerr_<model>_nf4.json. Prints the body total next to HQQ 4-bit's
body total (from qerr_<model>_4.json, lm_head excluded) so "bigger error" and "different error" can be told apart.

python nf4_qerr.py --models /content/merged/* --qerr-dir $ROOT/results/
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import free, load_model_nf4, model_tag  # noqa: E402
from weights import load_weight  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--qerr-dir", required=True)
args = ap.parse_args()

import bitsandbytes as bnb  # noqa: E402
from bitsandbytes.functional import dequantize_4bit  # noqa: E402

for path in args.models:
    tag = model_tag(path)
    out = os.path.join(args.qerr_dir, f"qerr_{tag}_nf4.json")
    if os.path.exists(out):
        print(f"[nf4qerr] skip {tag} (exists)")
        continue
    model, tok = load_model_nf4(path)
    qerr, n = {}, 0
    with torch.no_grad():
        for name, mod in model.model.layers.named_modules():
            if not isinstance(mod, bnb.nn.Linear4bit):
                continue
            full = f"model.layers.{name}"
            w_q = dequantize_4bit(mod.weight.data, mod.weight.quant_state).to(torch.float32)
            w = load_weight(path, full).to(w_q.device, torch.float32)
            assert w.shape == w_q.shape, (full, w.shape, w_q.shape)
            qerr[full] = float((w_q - w).norm())
            n += 1
            del w_q, w
    assert n == 196, f"{tag}: {n} Linear4bit layers, expected 196"
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(qerr, f, indent=1)
    os.replace(tmp, out)
    total = math.sqrt(sum(v * v for v in qerr.values()))
    line = f"[nf4qerr] {tag}: body total Frobenius {total:.2f} over {n} layers"
    hqq_path = os.path.join(args.qerr_dir, f"qerr_{tag}_4.json")
    if os.path.exists(hqq_path):
        with open(hqq_path) as f:
            h = json.load(f)
        hb = math.sqrt(sum(v * v for k, v in h.items() if k != "lm_head"))
        line += f" | HQQ 4-bit body total {hb:.2f} | ratio NF4/HQQ {total / hb:.3f}"
    print(line)
    del model, tok
    free()
print("[nf4qerr] done")
